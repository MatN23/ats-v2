"""Mixture-of-Experts layer.

Primary path: deepspeed.moe.layer.MoE, which handles top-k gating, capacity,
expert parallelism (ep_size) and the load-balancing auxiliary loss internally.
ats does not reimplement routing kernels.

Fallback: if deepspeed (or its MoE submodule) is not importable, a plain
PyTorch top-k router is used. It only works within a single process (no
expert parallelism) and prints a clear warning, since it is not the
production path.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ats.model.ffn import SwiGLU
from ats.model.initialization import init_residual_projection

logger = logging.getLogger("ats.model.moe")

try:
    from deepspeed.moe.layer import MoE as DeepSpeedMoE
    _DEEPSPEED_MOE_AVAILABLE = True
except ImportError:
    DeepSpeedMoE = None
    _DEEPSPEED_MOE_AVAILABLE = False


class _PyTorchMoEFallback(nn.Module):
    """Single-process top-k MoE fallback. Not expert-parallel. Used only when
    DeepSpeed's MoE module cannot be imported."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, num_experts: int,
        top_k: int, capacity_factor: float, load_balancing_weight: float,
        num_layers: int, quantization: str = "none",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balancing_weight = load_balancing_weight
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(hidden_size, intermediate_size, quantization=quantization) for _ in range(num_experts)]
        )
        # Each expert's down_proj writes directly into the residual stream,
        # exactly like the dense FFN path's down_proj -- it needs the same
        # depth-scaled init (see ats.model.initialization), not the generic
        # one a later blanket init_weights() pass would otherwise give it.
        for expert in self.experts:
            init_residual_projection(expert.down_proj, num_layers)
        self.last_expert_utilization: Optional[Dict[int, float]] = None

    def compute_routing(self, flat_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (top_k_probs, top_k_idx, router_probs) for a flattened
        [num_tokens, hidden_size] input: top_k_probs are the (renormalized,
        sum-to-1-per-token) gating weights for the selected experts,
        top_k_idx are their indices, router_probs is the full softmax
        distribution over all experts (used for the load-balancing aux
        loss). Factored out of forward() so it can be exercised directly by
        tests without duplicating the gating math."""
        router_logits = self.gate(flat_x)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        return top_k_probs, top_k_idx, router_probs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"MoE fallback expected hidden_size={self.hidden_size}, got {hidden_size}."
            )
        flat_x = x.reshape(-1, hidden_size)
        num_tokens = flat_x.shape[0]

        top_k_probs, top_k_idx, router_probs = self.compute_routing(flat_x)

        capacity = max(1, int(self.capacity_factor * num_tokens * self.top_k / self.num_experts))
        output = torch.zeros_like(flat_x)
        total_dropped = 0

        for expert_id in range(self.num_experts):
            token_mask = (top_k_idx == expert_id).any(dim=-1)
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            if token_indices.numel() > capacity:
                total_dropped += token_indices.numel() - capacity
                token_indices = token_indices[:capacity]
            expert_input = flat_x[token_indices]
            expert_output = self.experts[expert_id](expert_input)
            slot_weight = top_k_probs[token_indices]
            slot_mask = (top_k_idx[token_indices] == expert_id)
            weight = (slot_weight * slot_mask).sum(dim=-1, keepdim=True)
            output.index_add_(0, token_indices, expert_output * weight)

        if total_dropped > 0:
            logger.warning(
                "MoE fallback capacity exceeded: dropped %d of %d token-expert assignments "
                "(capacity=%d per expert). Dropped tokens get zero gradient from this MoE "
                "layer this step. Fix: increase moe_capacity_factor, or install deepspeed "
                "for the expert-parallel MoE path, which doesn't have this single-process "
                "capacity limitation in the same way.",
                total_dropped, num_tokens * self.top_k, capacity,
            )

        # Standard load-balancing auxiliary loss (Switch Transformer style):
        # encourages uniform routing probability mass and uniform dispatch fraction.
        router_prob_mean = router_probs.mean(dim=0)  # [num_experts]
        dispatch_mask = F.one_hot(top_k_idx, self.num_experts).float().sum(dim=1)  # [tokens, experts]
        dispatch_fraction = dispatch_mask.mean(dim=0)  # [num_experts]
        aux_loss = self.num_experts * torch.sum(router_prob_mean * dispatch_fraction)
        aux_loss = aux_loss * self.load_balancing_weight

        # Expose per-expert utilization (fraction of tokens dispatched to
        # each expert, normalized to sum to 1.0) so callers (MoELayer ->
        # ATSTransformer -> Trainer) can surface it to AdaptiveController's
        # expert-collapse detection, which otherwise never receives this
        # signal. Normalized separately from aux_loss's dispatch_fraction
        # (which is intentionally NOT sum-to-1 -- it's the raw per-token
        # mean dispatch indicator the Switch Transformer aux-loss formula
        # expects) so that this fallback backend reports utilization on the
        # same 0..1-summing-to-1 scale as the DeepSpeed backend's
        # counts/total normalization, rather than two backends silently
        # reporting the same metric on different scales.
        normalized_utilization = dispatch_fraction / dispatch_fraction.sum().clamp(min=1e-8)
        self.last_expert_utilization = {
            i: float(normalized_utilization[i].item()) for i in range(self.num_experts)
        }

        return output.reshape(batch, seq_len, hidden_size), aux_loss


class MoELayer(nn.Module):
    """Thin wrapper that dispatches to DeepSpeed's MoE if available, else the
    pure-PyTorch fallback above. Callers always get back (output, aux_loss)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_layers: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        load_balancing_weight: float = 0.01,
        ep_size: int = 1,
        quantization: str = "none",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.uses_deepspeed = _DEEPSPEED_MOE_AVAILABLE
        self.last_expert_utilization: Optional[Dict[int, float]] = None
        # Only used on the DeepSpeed path (the fallback scales its own
        # aux_loss internally) -- see the load_balancing_weight comment in
        # forward() for why this is applied here rather than passed into
        # DeepSpeedMoE's constructor.
        self.load_balancing_weight = load_balancing_weight

        if self.uses_deepspeed:
            expert = SwiGLU(hidden_size, intermediate_size, quantization=quantization)
            # Applied to the template BEFORE DeepSpeedMoE constructs its
            # per-expert copies, so every expert inherits the correct
            # depth-scaled residual-projection init on down_proj (matching
            # the dense FFN path) rather than the generic one.
            init_residual_projection(expert.down_proj, num_layers)
            # NOTE: deepspeed.moe.layer.MoE's constructor does not accept a
            # load_balancing_weight kwarg (checked against deepspeed>=0.12.0
            # through 0.19.x) -- it was never part of MoE.__init__'s API in
            # any version this project targets. MoE.forward() returns the
            # raw, unscaled load-balancing loss as l_aux; the weighting is
            # applied by this class's forward() instead, matching how
            # _PyTorchMoEFallback already scales its own aux_loss.
            self.moe = DeepSpeedMoE(
                hidden_size=hidden_size,
                expert=expert,
                num_experts=num_experts,
                ep_size=ep_size,
                k=top_k,
                capacity_factor=capacity_factor,
            )
        else:
            logger.warning(
                "deepspeed.moe.layer.MoE could not be imported; using a single-process "
                "PyTorch MoE fallback with no expert parallelism. Install deepspeed for "
                "production multi-GPU MoE training."
            )
            self.moe = _PyTorchMoEFallback(
                hidden_size, intermediate_size, num_experts, top_k,
                capacity_factor, load_balancing_weight, num_layers=num_layers, quantization=quantization,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.uses_deepspeed:
            raw_output = self.moe(x)
            try:
                output, aux_loss, exp_counts = raw_output
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "deepspeed.moe.layer.MoE.forward() did not return the expected "
                    "(output, aux_loss, exp_counts) 3-tuple this installed DeepSpeed "
                    f"version actually returned: {type(raw_output).__name__} "
                    f"{'of length ' + str(len(raw_output)) if hasattr(raw_output, '__len__') else ''}. "
                    "This usually means your installed DeepSpeed version's MoE API "
                    "differs from what ats-v2 was written against. "
                    "Fix: check `pip show deepspeed` against ats-v2's requirements.txt "
                    "pin (deepspeed>=0.12.0), or open an issue with your DeepSpeed version."
                ) from exc
            # exp_counts (previously discarded as `_exp_counts`) is DeepSpeed's
            # per-expert token count for this forward pass; normalize into the
            # same {expert_id: fraction} shape the PyTorch fallback exposes,
            # so AdaptiveController's expert-collapse detection actually
            # receives a signal regardless of which MoE backend is in use.
            try:
                counts = exp_counts.detach().float()
                total = counts.sum()
                if total > 0:
                    self.last_expert_utilization = {
                        i: float((counts[i] / total).item()) for i in range(counts.shape[0])
                    }
            except (AttributeError, TypeError, IndexError) as exc:
                logger.warning(
                    "Could not derive expert_utilization from DeepSpeed's exp_counts "
                    "(got %r): %s. Expert-collapse detection will be skipped this step.",
                    type(exp_counts).__name__, exc,
                )
            # DeepSpeedMoE.forward() returns the raw, unscaled load-balancing
            # loss (see the constructor comment on why load_balancing_weight
            # is applied here rather than passed into DeepSpeedMoE's ctor).
            return output, aux_loss * self.load_balancing_weight
        output, aux_loss = self.moe(x)
        self.last_expert_utilization = self.moe.last_expert_utilization
        return output, aux_loss
