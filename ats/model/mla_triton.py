"""Fused MLA KV decompression: computes `k = W_UK @ c` and `v = W_UV @ c`
from the shared compressed latent `c` in a single matmul instead of two.

UNVERIFIED ON REAL HARDWARE (no GPU/Triton available in the authoring
environment) -- see the same caveat in norm_triton.py/rope_triton.py/
moe_triton.py. HAS_TRITON gates every use; a missing or broken Triton
install falls back transparently to two separate (but numerically
identical) PyTorch matmuls.

Fusion strategy: W_UK and W_UV are concatenated once (at module-construction
time, not per-forward-call) into a single [latent_dim, 2*hidden_size] weight
matrix, so the "fusion" is doing one matmul against the concatenated weight
instead of two matmuls against separate weights -- then splitting the output
in half. The Triton kernel itself is a standard block-tiled GEMM (the same
structure as Triton's own canonical matmul tutorial kernel); this is the
best-understood, most heavily-precedented Triton kernel pattern, which is
why it's the one attempted here for the actual matmul rather than a novel
kernel shape.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] + k0 < K)
            b_mask = (offs_k[:, None] + k0 < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)

    def _triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a: [M, K], b: [K, N] -> [M, N], accumulated in fp32."""
        M, K = a.shape
        K2, N = b.shape
        if K != K2:
            raise ValueError(f"_triton_matmul: inner dims must match, got {K} and {K2}.")
        a = a.contiguous()
        b = b.contiguous()
        out = torch.empty((M, N), device=a.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            a, b, out, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out.to(a.dtype)


def fused_mla_kv_decompress(
    c: torch.Tensor, w_uk: torch.Tensor, w_uv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes (k, v) = (c @ w_uk.T, c @ w_uv.T) from the shared compressed
    latent `c`. When Triton is available and c is on CUDA, does this as a
    single matmul against the concatenated [w_uk; w_uv] weight; otherwise
    two ordinary PyTorch matmuls (numerically identical result either way).

    c: [..., latent_dim]. w_uk, w_uv: [out_dim, latent_dim] (nn.Linear
    weight layout, i.e. already transposed relative to the matmul).
    """
    if c.shape[-1] != w_uk.shape[-1] or c.shape[-1] != w_uv.shape[-1]:
        raise ValueError(
            f"fused_mla_kv_decompress: c's last dim ({c.shape[-1]}) must match "
            f"w_uk/w_uv's last dim ({w_uk.shape[-1]}, {w_uv.shape[-1]})."
        )
    if w_uk.shape[0] != w_uv.shape[0]:
        raise ValueError(
            f"fused_mla_kv_decompress: w_uk and w_uv must have the same output dim, "
            f"got {w_uk.shape[0]} and {w_uv.shape[0]}."
        )

    orig_shape = c.shape
    latent_dim = orig_shape[-1]
    c_flat = c.reshape(-1, latent_dim)
    out_dim = w_uk.shape[0]

    if HAS_TRITON and c.is_cuda:
        concat_weight = torch.cat([w_uk, w_uv], dim=0)  # [2*out_dim, latent_dim]
        combined = _triton_matmul(c_flat, concat_weight.t())  # [num_rows, 2*out_dim]
        k_flat, v_flat = combined.split(out_dim, dim=-1)
    else:
        k_flat = c_flat @ w_uk.t()
        v_flat = c_flat @ w_uv.t()

    k = k_flat.reshape(*orig_shape[:-1], out_dim)
    v = v_flat.reshape(*orig_shape[:-1], out_dim)
    return k, v
