"""Tests for ats/model/*_triton.py.

Honesty note on what these tests can actually prove: in any environment
without a CUDA GPU and a working Triton install, every *_triton kernel path
is skipped by its own HAS_TRITON/is_cuda guard, and these tests instead
verify the PyTorch fallback matches the existing, already-tested reference
implementations (RMSNorm, apply_rotary_pos_emb, MoE routing math). The
"Triton output matches PyTorch fallback within 1e-5" parity tests only run
(and only mean anything) on a machine with Triton and a CUDA GPU; elsewhere
they are skipped rather than silently passing on unexercised code.
"""

from __future__ import annotations

import pytest
import torch

from ats.model.mla_triton import fused_mla_kv_decompress
from ats.model.moe_triton import HAS_TRITON as MOE_HAS_TRITON
from ats.model.moe_triton import fused_moe_dispatch, fused_moe_routing
from ats.model.norm_triton import HAS_TRITON as NORM_HAS_TRITON
from ats.model.norm_triton import fused_rmsnorm_residual
from ats.model.rope import RotaryEmbedding, apply_rotary_pos_emb
from ats.model.rope_triton import HAS_TRITON as ROPE_HAS_TRITON
from ats.model.rope_triton import fused_apply_rope

_CUDA_AND_TRITON_AVAILABLE = torch.cuda.is_available() and NORM_HAS_TRITON


# --- Fallback correctness (always runs, no GPU required) ---

def test_fused_rmsnorm_residual_fallback_matches_manual_computation():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    residual = torch.randn(2, 5, 16)
    weight = torch.randn(16)
    eps = 1e-6

    out = fused_rmsnorm_residual(x, residual, weight, eps)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    expected_norm = (x * torch.rsqrt(variance + eps)) * weight
    expected = residual + expected_norm
    assert torch.allclose(out, expected, atol=1e-5)


def test_fused_rmsnorm_residual_rejects_shape_mismatch():
    x = torch.randn(2, 5, 16)
    residual = torch.randn(2, 5, 8)
    weight = torch.randn(16)
    with pytest.raises(ValueError):
        fused_rmsnorm_residual(x, residual, weight)


def test_fused_apply_rope_fallback_matches_reference_implementation():
    torch.manual_seed(0)
    head_dim = 8
    q = torch.randn(1, 2, 6, head_dim)
    k = torch.randn(1, 2, 6, head_dim)
    rope = RotaryEmbedding(head_dim, max_seq_len=16, theta=10000.0)
    cos, sin = rope(seq_len=6, device=torch.device("cpu"), dtype=torch.float32)

    q_fused = fused_apply_rope(q, cos, sin)
    k_fused = fused_apply_rope(k, cos, sin)
    q_ref, k_ref = apply_rotary_pos_emb(q, k, cos, sin)

    assert torch.allclose(q_fused, q_ref, atol=1e-5)
    assert torch.allclose(k_fused, k_ref, atol=1e-5)


def test_fused_moe_routing_fallback_matches_manual_topk():
    torch.manual_seed(0)
    gate_logits = torch.randn(5, 4)
    top_k = 2

    probs, idx = fused_moe_routing(gate_logits, top_k)

    router_probs = torch.softmax(gate_logits, dim=-1)
    expected_probs, expected_idx = torch.topk(router_probs, top_k, dim=-1)
    expected_probs = expected_probs / expected_probs.sum(dim=-1, keepdim=True)

    assert torch.allclose(probs, expected_probs, atol=1e-5)
    assert torch.equal(idx, expected_idx)


def test_fused_moe_dispatch_returns_valid_routing():
    torch.manual_seed(0)
    gate_logits = torch.randn(6, 4)
    hidden_states = torch.randn(6, 16)
    probs, idx = fused_moe_dispatch(gate_logits, hidden_states, top_k=2, capacity_factor=1.25)
    assert probs.shape == (6, 2)
    assert idx.shape == (6, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(6), atol=1e-5)


def test_fused_moe_dispatch_rejects_mismatched_token_count():
    gate_logits = torch.randn(6, 4)
    hidden_states = torch.randn(5, 16)  # mismatched token count on purpose
    with pytest.raises(ValueError):
        fused_moe_dispatch(gate_logits, hidden_states, top_k=2, capacity_factor=1.25)


def test_fused_mla_kv_decompress_fallback_matches_manual_matmul():
    torch.manual_seed(0)
    latent_dim, out_dim = 8, 16
    c = torch.randn(2, 5, latent_dim)
    w_uk = torch.randn(out_dim, latent_dim)
    w_uv = torch.randn(out_dim, latent_dim)

    k, v = fused_mla_kv_decompress(c, w_uk, w_uv)

    expected_k = c @ w_uk.t()
    expected_v = c @ w_uv.t()
    assert torch.allclose(k, expected_k, atol=1e-5)
    assert torch.allclose(v, expected_v, atol=1e-5)


def test_fused_mla_kv_decompress_rejects_dim_mismatch():
    c = torch.randn(2, 5, 8)
    w_uk = torch.randn(16, 8)
    w_uv = torch.randn(16, 4)  # mismatched latent_dim on purpose
    with pytest.raises(ValueError):
        fused_mla_kv_decompress(c, w_uk, w_uv)


# --- Triton-vs-fallback parity (only meaningful with real CUDA + Triton) ---

@pytest.mark.skipif(
    not _CUDA_AND_TRITON_AVAILABLE,
    reason="Requires a CUDA GPU and a working Triton install; not available here.",
)
def test_triton_rmsnorm_residual_matches_pytorch_within_tolerance():
    torch.manual_seed(0)
    x = torch.randn(4, 32, device="cuda")
    residual = torch.randn(4, 32, device="cuda")
    weight = torch.randn(32, device="cuda")

    from ats.model.norm_triton import _pytorch_rmsnorm_residual, _triton_rmsnorm_residual

    triton_out = _triton_rmsnorm_residual(x, residual, weight, eps=1e-6)
    pytorch_out = _pytorch_rmsnorm_residual(x, residual, weight, eps=1e-6)
    assert torch.allclose(triton_out, pytorch_out, atol=1e-5)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and ROPE_HAS_TRITON),
    reason="Requires a CUDA GPU and a working Triton install; not available here.",
)
def test_triton_rope_matches_pytorch_within_tolerance():
    torch.manual_seed(0)
    head_dim = 16
    q = torch.randn(2, 4, 8, head_dim, device="cuda")
    rope = RotaryEmbedding(head_dim, max_seq_len=16, theta=10000.0)
    cos, sin = rope(seq_len=8, device=torch.device("cuda"), dtype=torch.float32)

    from ats.model.rope_triton import _pytorch_apply_rope, _triton_apply_rope

    triton_out = _triton_apply_rope(q, cos, sin)
    pytorch_out = _pytorch_apply_rope(q, cos, sin)
    assert torch.allclose(triton_out, pytorch_out, atol=1e-5)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and MOE_HAS_TRITON),
    reason="Requires a CUDA GPU and a working Triton install; not available here.",
)
def test_triton_moe_routing_matches_pytorch_within_tolerance():
    torch.manual_seed(0)
    gate_logits = torch.randn(32, 8, device="cuda")

    from ats.model.moe_triton import _pytorch_moe_routing, _triton_moe_routing

    triton_probs, triton_idx = _triton_moe_routing(gate_logits, top_k=2)
    pytorch_probs, pytorch_idx = _pytorch_moe_routing(gate_logits, top_k=2)
    assert torch.equal(triton_idx.sort(dim=-1).values, pytorch_idx.sort(dim=-1).values)
    assert torch.allclose(
        triton_probs.sort(dim=-1).values, pytorch_probs.sort(dim=-1).values, atol=1e-5,
    )


@pytest.mark.skipif(
    not _CUDA_AND_TRITON_AVAILABLE,
    reason="Requires a CUDA GPU and a working Triton install; not available here.",
)
def test_triton_mla_matmul_matches_pytorch_within_tolerance():
    torch.manual_seed(0)
    c = torch.randn(2, 8, 16, device="cuda")
    w_uk = torch.randn(32, 16, device="cuda")
    w_uv = torch.randn(32, 16, device="cuda")

    from ats.model.mla_triton import _triton_matmul

    k_triton = _triton_matmul(c.reshape(-1, 16), w_uk.t()).reshape(2, 8, 32)
    k_pytorch = c @ w_uk.t()
    assert torch.allclose(k_triton, k_pytorch, atol=1e-5)
