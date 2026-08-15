"""RMSNorm. Uses torch's native fused implementation when available (PT >= 2.4),
falls back to a plain, correct nn.Module implementation otherwise. No custom kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(
                f"RMSNorm hidden_size must be positive, got {hidden_size}."
            )
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self._has_native_rms_norm = hasattr(F, "rms_norm")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"RMSNorm expected last dimension {self.hidden_size}, got {x.shape[-1]} "
                f"(full shape {tuple(x.shape)}). Fix: check that the layer feeding this "
                f"RMSNorm produces hidden_size={self.hidden_size}."
            )
        if self._has_native_rms_norm:
            return F.rms_norm(x, (self.hidden_size,), self.weight, self.eps)
        # Manual fallback, computed in fp32 for numerical stability then cast back.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)
