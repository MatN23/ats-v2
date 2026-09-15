"""Fused RMSNorm + residual-add, as a single Triton kernel (one memory pass
instead of two separate elementwise passes for the norm and the add).

UNVERIFIED ON REAL HARDWARE: this kernel was authored without access to a
GPU or a Triton installation to compile/run/benchmark it. The math mirrors
ats.model.norm.RMSNorm exactly (same reduction, same epsilon placement), and
the block-size/masking logic follows Triton's standard row-wise-reduction
kernel pattern, but "should be correct by inspection" is not the same
guarantee as "has been run." HAS_TRITON gates every use of this module, so
if Triton either isn't installed or turns out to be buggy here, callers
transparently fall back to fused_rmsnorm_residual's pure-PyTorch path
(functionally identical, just not fused into one kernel launch), and no
caller crashes.

EXPERIMENTAL -- NOT ON ANY PRODUCTION EXECUTION PATH.

Audit status (BUG-119): nothing under ats/ imports this module. The only
callers anywhere in the repository are tests/test_triton.py. Production
training, evaluation, export and generation all go through the PyTorch
implementations instead (ats.model.norm.RMSNorm plus an ordinary residual add), so this code cannot execute during
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
except Exception as exc:  # noqa: BLE001 -- deliberately broad: this
    # is an optional-dependency import guard (see the comment below)
    # and must catch whatever a broken/mismatched native install
    # raises, not just ImportError, or the fallback this guard exists
    # for never triggers.
    # BUG-123 (same class as ats.model.moe's DeepSpeed import guard): triton
    # is a compiled, torch/CUDA-version-sensitive dependency. An
    # incompatible install can raise something other than ImportError deep
    # inside its own import machinery, which previously crashed this module
    # (and everything that imports it) instead of degrading to the
    # documented PyTorch fallback every function in this file already has.
    import logging

    logging.getLogger(__name__).warning(
        "triton is installed but failed to import cleanly (%s: %s). Falling "
        "back to the pure-PyTorch implementation. Fix: reinstall triton "
        "matching your installed torch/CUDA version.",
        type(exc).__name__,
        exc,
    )
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_residual_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row_start = row_idx * n_cols
        x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        residual = tl.load(
            residual_ptr + row_start + col_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)

        variance = tl.sum(x * x, axis=0) / n_cols
        inv_rms = 1.0 / tl.sqrt(variance + eps)
        normed = x * inv_rms * weight
        out = residual + normed

        tl.store(out_ptr + row_start + col_offsets, out, mask=mask)

    def _triton_rmsnorm_residual(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        orig_shape = x.shape
        hidden_size = orig_shape[-1]
        x_flat = x.reshape(-1, hidden_size).contiguous()
        residual_flat = residual.reshape(-1, hidden_size).contiguous()
        num_rows = x_flat.shape[0]

        out = torch.empty_like(x_flat)
        block_size = triton.next_power_of_2(hidden_size)
        grid = (num_rows,)
        _rmsnorm_residual_kernel[grid](
            x_flat,
            residual_flat,
            weight,
            out,
            hidden_size,
            eps,
            BLOCK_SIZE=block_size,
        )
        return out.reshape(orig_shape)


def _pytorch_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    normed = (x_fp32 * torch.rsqrt(variance + eps)).to(input_dtype) * weight
    return residual + normed


def fused_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes `residual + rms_norm(x) * weight` in as few memory passes as
    possible. Uses a fused Triton kernel when available and running on CUDA;
    otherwise falls back to the equivalent (unfused, but numerically
    identical) PyTorch sequence."""
    if x.shape != residual.shape:
        raise ValueError(
            f"fused_rmsnorm_residual: x shape {tuple(x.shape)} must match "
            f"residual shape {tuple(residual.shape)}."
        )
    if weight.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"fused_rmsnorm_residual: weight last dim {weight.shape[-1]} must match "
            f"x last dim {x.shape[-1]}."
        )
    use_triton = HAS_TRITON and x.is_cuda
    if (
        use_triton
        and torch.is_grad_enabled()
        and (x.requires_grad or residual.requires_grad or weight.requires_grad)
    ):
        raise RuntimeError(
            "fused_rmsnorm_residual: the Triton path is not differentiable (it launches a raw "
            "kernel into a torch.empty buffer, which autograd cannot see), and "
            "an input requires grad. Refusing to return a silently detached "
            "result. This module is experimental and is not used by any "
            "production path in ats -- see its module docstring."
        )
    if use_triton:
        return _triton_rmsnorm_residual(x, residual, weight, eps)
    return _pytorch_rmsnorm_residual(x, residual, weight, eps)
