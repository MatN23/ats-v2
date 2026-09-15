"""Fused RoPE rotation kernel.

UNVERIFIED ON REAL HARDWARE (no GPU/Triton available to compile or run this
in the authoring environment) -- see the same caveat in norm_triton.py.
HAS_TRITON gates every use, so a missing or broken Triton install falls
back transparently to the pure-PyTorch path (identical to
ats.model.rope.apply_rotary_pos_emb) rather than crashing.

Scope note: this fuses the elementwise rotate_half/cos/sin application for
Q and K in a single kernel launch each. It does NOT fuse the preceding
Q/K linear projection (input @ W_q, input @ W_k) into the same kernel --
that would require a full block-tiled matmul kernel with its own
correctness surface (tile sizes, shared-memory staging, boundary masking on
non-power-of-2 dimensions), which is a much larger risk to author blind
than an elementwise op. Fusing "matmul -> reshape -> rope" end-to-end, as an
ambitious version of this kernel could in principle do, is left as future
work rather than claimed here.

EXPERIMENTAL -- NOT ON ANY PRODUCTION EXECUTION PATH.

Audit status (BUG-119): nothing under ats/ imports this module. The only
callers anywhere in the repository are tests/test_triton.py. Production
training, evaluation, export and generation all go through the PyTorch
implementations instead (ats.model.rope.apply_rotary_pos_emb), so this code cannot execute during
a real run no matter what hardware is present.

It is also not usable as-is for training: the Triton path launches a raw
kernel that writes into a torch.empty() buffer. Raw kernel launches are
invisible to autograd, so the returned tensor carries no grad_fn and
gradients stop dead at this call. The PyTorch fallback in the same function
IS differentiable -- which means the function would silently be
differentiable on CPU and silently NOT differentiable on CUDA. To prevent
that from ever becoming a silent training bug, the Triton path now refuses
to run on inputs that require gradients (see the guard in the wrapper
below) rather than returning a detached result.

Wiring any of this into the model would require, at minimum, wrapping each
kernel in a torch.autograd.Function with a hand-written backward, and
validating it on real hardware. None of that has been done, and none of it
can be validated in an environment without a GPU.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        seq_len,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Grid: one program per (batch*heads, seq_position) pair, flattened
        # over the leading dims by the caller before launch.
        row_idx = tl.program_id(0)
        half = head_dim // 2
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < half

        seq_pos = row_idx % seq_len
        row_start = row_idx * head_dim

        x1 = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        x2 = tl.load(x_ptr + row_start + half + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        cos1 = tl.load(
            cos_ptr + seq_pos * head_dim + col_offsets, mask=mask, other=1.0
        ).to(tl.float32)
        sin1 = tl.load(
            sin_ptr + seq_pos * head_dim + col_offsets, mask=mask, other=0.0
        ).to(tl.float32)

        # rotate_half convention: out1 = x1*cos - x2*sin, out2 = x2*cos + x1*sin
        # (cos/sin are duplicated across the two halves, per rope.py's cache
        # layout, so cos1 read from the first half slice is valid for both.)
        out1 = x1 * cos1 - x2 * sin1
        out2 = x2 * cos1 + x1 * sin1

        tl.store(out_ptr + row_start + col_offsets, out1, mask=mask)
        tl.store(out_ptr + row_start + half + col_offsets, out2, mask=mask)

    def _triton_apply_rope(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # x: [batch, heads, seq_len, head_dim]; cos/sin: [seq_len, head_dim]
        batch, heads, seq_len, head_dim = x.shape
        x_flat = x.reshape(-1, head_dim).contiguous()
        out = torch.empty_like(x_flat)
        num_rows = x_flat.shape[0]
        block_size = triton.next_power_of_2(head_dim // 2)
        grid = (num_rows,)
        _rope_kernel[grid](
            x_flat, cos, sin, out, seq_len, head_dim, BLOCK_SIZE=block_size
        )
        return out.reshape(batch, heads, seq_len, head_dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _pytorch_apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos_b) + (_rotate_half(x) * sin_b)


def fused_apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Applies RoPE rotation to `x` ([batch, heads, seq_len, head_dim]) given
    `cos`/`sin` ([seq_len, head_dim]), matching
    ats.model.rope.apply_rotary_pos_emb's math exactly. Uses a fused Triton
    kernel when available and running on CUDA; otherwise the equivalent
    PyTorch sequence."""
    if x.dim() != 4:
        raise ValueError(
            f"fused_apply_rope expected x of shape [batch, heads, seq_len, head_dim], "
            f"got {tuple(x.shape)}."
        )
    if x.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"fused_apply_rope: x head_dim ({x.shape[-1]}) must match cos/sin dim "
            f"({cos.shape[-1]})."
        )
    use_triton = HAS_TRITON and x.is_cuda
    if use_triton and torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError(
            "fused_apply_rope: the Triton path is not differentiable (it launches a raw "
            "kernel into a torch.empty buffer, which autograd cannot see), and "
            "an input requires grad. Refusing to return a silently detached "
            "result. This module is experimental and is not used by any "
            "production path in ats -- see its module docstring."
        )
    if use_triton:
        return _triton_apply_rope(x, cos, sin)
    return _pytorch_apply_rope(x, cos, sin)
