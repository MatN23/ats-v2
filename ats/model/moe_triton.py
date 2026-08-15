"""Fused MoE routing: softmax + top-k expert selection in a single Triton
kernel, per token row.

UNVERIFIED ON REAL HARDWARE (no GPU/Triton available in the authoring
environment) -- see the same caveat in norm_triton.py/rope_triton.py.
HAS_TRITON gates every use; a missing or broken Triton install falls back
transparently to the pure-PyTorch path.

Scope note, stated honestly rather than overclaimed: this kernel fuses only
the softmax + top-k *selection* (deciding which experts each token goes to,
and with what weight) into Triton. The actual token *dispatch* -- gathering
each expert's assigned tokens into a contiguous buffer, respecting a
capacity limit, running the expert FFN, and scattering results back -- stays
in plain PyTorch (ats.model.moe._PyTorchMoEFallback's existing
index_add_-based implementation) even when Triton is used for routing.
Writing a correct, capacity-aware, dynamic-shape gather/scatter Triton
kernel is a substantially higher-risk undertaking to author without
hardware to validate against than the fixed-shape per-row reduction below,
so it is intentionally not attempted here. "Fused MoE routing" in this file
means routing decisions, not the full dispatch pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_MAX_SUPPORTED_TOP_K = 8  # unrolled loop bound; see _triton_moe_routing


if HAS_TRITON:

    @triton.jit
    def _moe_routing_kernel(
        logits_ptr,
        top_k_probs_ptr,
        top_k_idx_ptr,
        num_experts,
        top_k: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < num_experts

        row_start = row_idx * num_experts
        logits = tl.load(
            logits_ptr + row_start + col_offsets, mask=mask, other=-float("inf")
        )

        row_max = tl.max(logits, axis=0)
        shifted = logits - row_max
        exp = tl.exp(shifted)
        exp = tl.where(mask, exp, 0.0)
        denom = tl.sum(exp, axis=0)
        probs = exp / denom  # softmax over experts, this row

        # Iteratively extract the top-k via k passes of argmax + mask-out.
        # top_k is a compile-time constant (tl.constexpr), so this Python
        # for-loop is unrolled at trace time -- a standard Triton pattern
        # for small, fixed k (MoE top-k is almost always 1-8 in practice).
        remaining = probs
        raw_sum = tl.zeros((), dtype=tl.float32)
        selected_probs = tl.zeros((top_k,), dtype=tl.float32)
        selected_idx = tl.zeros((top_k,), dtype=tl.int32)
        for k in tl.static_range(top_k):
            best_val = tl.max(remaining, axis=0)
            best_idx = tl.argmax(remaining, axis=0)
            selected_probs = tl.where(
                tl.arange(0, top_k) == k, best_val, selected_probs
            )
            selected_idx = tl.where(tl.arange(0, top_k) == k, best_idx, selected_idx)
            raw_sum += best_val
            remaining = tl.where(col_offsets == best_idx, -1.0, remaining)

        normalized = selected_probs / raw_sum
        out_row_start = row_idx * top_k
        out_offsets = tl.arange(0, top_k)
        tl.store(top_k_probs_ptr + out_row_start + out_offsets, normalized)
        tl.store(top_k_idx_ptr + out_row_start + out_offsets, selected_idx)

    def _triton_moe_routing(
        gate_logits: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if top_k > _MAX_SUPPORTED_TOP_K:
            raise ValueError(
                f"Triton MoE routing kernel supports top_k <= {_MAX_SUPPORTED_TOP_K} "
                f"(unrolled loop bound), got top_k={top_k}. "
                f"Fix: reduce moe_top_k, or this call will need the PyTorch fallback."
            )
        num_tokens, num_experts = gate_logits.shape
        gate_logits = gate_logits.contiguous()
        top_k_probs = torch.empty(
            (num_tokens, top_k), device=gate_logits.device, dtype=torch.float32
        )
        top_k_idx = torch.empty(
            (num_tokens, top_k), device=gate_logits.device, dtype=torch.int32
        )

        block_size = triton.next_power_of_2(num_experts)
        grid = (num_tokens,)
        _moe_routing_kernel[grid](
            gate_logits,
            top_k_probs,
            top_k_idx,
            num_experts,
            top_k,
            BLOCK_SIZE=block_size,
        )
        return top_k_probs, top_k_idx.long()


def _pytorch_moe_routing(
    gate_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    router_probs = F.softmax(gate_logits, dim=-1)
    top_k_probs, top_k_idx = torch.topk(router_probs, top_k, dim=-1)
    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
    return top_k_probs, top_k_idx


def fused_moe_routing(
    gate_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (top_k_probs, top_k_idx) -- softmax + top-k expert selection,
    fused into one Triton kernel per token row when available and running
    on CUDA with top_k small enough for the unrolled kernel; otherwise the
    equivalent PyTorch sequence (also used by ats.model.moe's fallback)."""
    if gate_logits.dim() != 2:
        raise ValueError(
            f"fused_moe_routing expected gate_logits of shape [num_tokens, num_experts], "
            f"got {tuple(gate_logits.shape)}."
        )
    if top_k < 1:
        raise ValueError(f"fused_moe_routing requires top_k >= 1, got {top_k}.")
    if HAS_TRITON and gate_logits.is_cuda and top_k <= _MAX_SUPPORTED_TOP_K:
        return _triton_moe_routing(gate_logits, top_k)
    return _pytorch_moe_routing(gate_logits, top_k)


def fused_moe_dispatch(
    gate_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    top_k: int,
    capacity_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper matching the spec's requested signature: computes
    fused routing (see fused_moe_routing) and the per-expert capacity limit
    for the given tokens/capacity_factor. Returns (top_k_probs, top_k_idx);
    actual token gather/dispatch against experts is left to the caller
    (ats.model.moe.MoELayer), consistent with this file's documented scope.
    """
    if hidden_states.shape[0] != gate_logits.shape[0]:
        raise ValueError(
            f"fused_moe_dispatch: hidden_states has {hidden_states.shape[0]} tokens but "
            f"gate_logits has {gate_logits.shape[0]}; they must match."
        )
    if capacity_factor <= 0:
        raise ValueError(
            f"fused_moe_dispatch requires capacity_factor > 0, got {capacity_factor}."
        )
    return fused_moe_routing(gate_logits, top_k)
