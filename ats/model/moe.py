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
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ats.model.ffn import SwiGLU

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balancing_weight = load_balancing_weight
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"MoE fallback expected hidden_size={self.hidden_size}, got {hidden_size}."
            )
        flat_x = x.reshape(-1, hidden_size)
        num_tokens = flat_x.shape[0]

        router_logits = self.gate(flat_x)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        capacity = max(1, int(self.capacity_factor * num_tokens * self.top_k / self.num_experts))
        output = torch.zeros_like(flat_x)

        for expert_id in range(self.num_experts):
            token_mask = (top_k_idx == expert_id).any(dim=-1)
            token_indices = token_mask.nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            if token_indices.numel() > capacity:
                token_indices = token_indices[:capacity]
            expert_input = flat_x[token_indices]
            expert_output = self.experts[expert_id](expert_input)
            slot_weight = top_k_probs[token_indices]
            slot_mask = (top_k_idx[token_indices] == expert_id)
            weight = (slot_weight * slot_mask).sum(dim=-1, keepdim=True)
            output.index_add_(0, token_indices, expert_output * weight)

        # Standard load-balancing auxiliary loss (Switch Transformer style):
        # encourages uniform routing probability mass and uniform dispatch fraction.
        router_prob_mean = router_probs.mean(dim=0)  # [num_experts]
        dispatch_mask = F.one_hot(top_k_idx, self.num_experts).float().sum(dim=1)  # [tokens, experts]
        dispatch_fraction = dispatch_mask.mean(dim=0)  # [num_experts]
        aux_loss = self.num_experts * torch.sum(router_prob_mean * dispatch_fraction)
        aux_loss = aux_loss * self.load_balancing_weight

        return output.reshape(batch, seq_len, hidden_size), aux_loss


class MoELayer(nn.Module):
    """Thin wrapper that dispatches to DeepSpeed's MoE if available, else the
    pure-PyTorch fallback above. Callers always get back (output, aux_loss)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        load_balancing_weight: float = 0.01,
        ep_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.uses_deepspeed = _DEEPSPEED_MOE_AVAILABLE

        if self.uses_deepspeed:
            expert = SwiGLU(hidden_size, intermediate_size)
            self.moe = DeepSpeedMoE(
                hidden_size=hidden_size,
                expert=expert,
                num_experts=num_experts,
                ep_size=ep_size,
                k=top_k,
                capacity_factor=capacity_factor,
                load_balancing_weight=load_balancing_weight,
            )
        else:
            logger.warning(
                "deepspeed.moe.layer.MoE could not be imported; using a single-process "
                "PyTorch MoE fallback with no expert parallelism. Install deepspeed for "
                "production multi-GPU MoE training."
            )
            self.moe = _PyTorchMoEFallback(
                hidden_size, intermediate_size, num_experts, top_k,
                capacity_factor, load_balancing_weight,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.uses_deepspeed:
            output, aux_loss, _exp_counts = self.moe(x)
            return output, aux_loss
        return self.moe(x)
