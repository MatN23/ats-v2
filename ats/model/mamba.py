"""A pure-PyTorch selective state-space (Mamba-style) block.

This is a real, sequential selective-scan implementation, not a renamed
transformer block: there is no attention here, no QKV projections, and the
recurrence explicitly carries a state tensor across the sequence dimension.
It is not the fused CUDA selective-scan kernel from the original Mamba
paper/repo (that would violate the "no custom CUDA kernels" rule); this is
the reference-style O(seq_len) sequential recurrence written in plain
PyTorch ops, which is correct but slower than a fused kernel.

MambaBlock itself is a sub-layer (like GroupedQueryAttention or SwiGLU
elsewhere in this codebase): it does not apply its own residual connection.
Callers (see ats.model.transformer.MambaLayer) are responsible for the
pre-norm residual wrapping: `x + MambaBlock(norm(x))`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"MambaBlock hidden_size must be positive, got {hidden_size}.")
        if d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError(
                f"MambaBlock d_state, d_conv, expand must all be positive, got "
                f"d_state={d_state}, d_conv={d_conv}, expand={expand}."
            )
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * hidden_size

        # Input projection produces both the main branch and the gating branch.
        self.in_proj = nn.Linear(hidden_size, 2 * self.d_inner, bias=False)

        # Causal depthwise 1D convolution over the sequence dimension, applied
        # per-channel (groups=d_inner), giving each position a short local
        # receptive field before the selective scan.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )

        # Input-dependent (selective) SSM parameters: dt, B, C are all
        # functions of the current activations, not fixed weights — this is
        # what makes it "selective" rather than a plain linear SSM.
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # -> (B, C, dt_raw)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A is a learned, per-channel, per-state negative-definite matrix
        # (stored as log for positivity via -exp(A_log)), the continuous-time
        # SSM's state transition parameter.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"MambaBlock expected input of shape [batch, seq_len, hidden_size], "
                f"got shape {tuple(x.shape)}."
            )
        batch, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"MambaBlock expected hidden_size={self.hidden_size}, got {hidden_size}."
            )

        x_and_gate = self.in_proj(x)  # [batch, seq_len, 2*d_inner]
        x_main, gate = x_and_gate.chunk(2, dim=-1)

        # Causal depthwise conv: transpose to [batch, d_inner, seq_len], pad
        # is already causal via padding=d_conv-1 on the left+right, so trim
        # the extra right-side outputs to keep it strictly causal.
        x_conv = self.conv1d(x_main.transpose(1, 2))[..., :seq_len]
        x_conv = F.silu(x_conv.transpose(1, 2))  # [batch, seq_len, d_inner]

        # Selective parameters, input-dependent per position.
        proj = self.x_proj(x_conv)  # [batch, seq_len, 2*d_state + 1]
        B, C, dt_raw = torch.split(proj, [self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))  # [batch, seq_len, d_inner], always positive

        A = -torch.exp(self.A_log)  # [d_inner, d_state], negative for stability

        # Discretize (zero-order hold) and run the sequential selective scan.
        # state: [batch, d_inner, d_state]
        state = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            dt_t = dt[:, t, :]  # [batch, d_inner]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [batch, d_inner, d_state]
            dB = dt_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # [batch, d_inner, d_state]
            state = state * dA + dB * x_conv[:, t, :].unsqueeze(-1)
            y_t = torch.einsum("bdn,bn->bd", state, C[:, t, :])  # [batch, d_inner]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # [batch, seq_len, d_inner]
        y = y + x_conv * self.D  # skip connection (D is a per-channel scalar)

        y = y * F.silu(gate)  # gating
        out = self.out_proj(y)
        return out
