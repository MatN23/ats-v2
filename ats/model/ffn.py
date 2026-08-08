"""SwiGLU feed-forward network. No custom activation kernels; torch.compile fuses
the elementwise ops automatically when the model is compiled."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ats.model.quantization import make_linear


class SwiGLU(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, dropout: float = 0.0,
        quantization: str = "none",
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError(
                f"SwiGLU requires positive hidden_size and intermediate_size, got "
                f"hidden_size={hidden_size}, intermediate_size={intermediate_size}."
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = make_linear(hidden_size, 2 * intermediate_size, quantization, bias=False)
        self.down_proj = make_linear(intermediate_size, hidden_size, quantization, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"SwiGLU expected last dimension {self.hidden_size}, got {x.shape[-1]}."
            )
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        out = self.down_proj(F.silu(gate) * up)
        return self.dropout(out)
