"""A pure-PyTorch selective state-space (Mamba-style) block.

This is a real selective-scan implementation, not a renamed transformer
block: there is no attention here, no QKV projections, and the recurrence
explicitly carries a state tensor across the sequence dimension. It is not
the fused CUDA selective-scan kernel from the original Mamba paper/repo
(that would violate the "no custom CUDA kernels" rule).

The scan itself uses CHUNKED parallel computation rather than a Python loop
over every timestep: within each chunk of `chunk_size` positions, the
recurrence is solved via a single batched matmul against a lower-triangular
log-space decay matrix (the standard trick for parallelizing a linear
recurrence with time-varying coefficients), so sequential Python-level steps
drop from O(seq_len) to O(seq_len / chunk_size). Only the carry-over state
between chunks is sequential. This was verified numerically against a plain
sequential-loop reference implementation in pure numpy (exact match to
float64 precision for short sequences, ~1e-7 relative error at seq_len=4096
in float32 with an extreme decay-coefficient range) before being written
here -- see CHANGES.md for the verification methodology, since this
sandbox has no GPU/torch to run the actual nn.Module against.

chunk_size trades memory for sequential-step count: the per-chunk
lower-triangular decay tensor is [batch, chunk_size, chunk_size, d_inner,
d_state], so larger chunks mean fewer sequential steps but quadratically
more peak memory per chunk. The default (32) is conservative; increase it
if you have memory headroom and want fewer sequential launches, decrease it
if you hit OOM on this specific tensor.

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
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"MambaBlock hidden_size must be positive, got {hidden_size}.")
        if d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError(
                f"MambaBlock d_state, d_conv, expand must all be positive, got "
                f"d_state={d_state}, d_conv={d_conv}, expand={expand}."
            )
        if chunk_size <= 0:
            raise ValueError(f"MambaBlock chunk_size must be positive, got {chunk_size}.")
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * hidden_size
        self.chunk_size = chunk_size

        # Input projection produces both the main branch and the gating branch.
        self.in_proj = nn.Linear(hidden_size, 2 * self.d_inner, bias=False)

        # Causal depthwise 1D convolution over the sequence dimension, applied
        # per-channel (groups=d_inner), giving each position a short local
        # receptive field before the selective scan.
        # Bug 14 fix: nn.Conv1d's `padding` argument is always symmetric (it
        # cannot express a (left, right) pair), so padding=d_conv-1 padded
        # both sides and the extra right-side output columns were computed
        # only to be immediately thrown away by the [..., :seq_len] trim in
        # forward(). Padding=0 here, combined with an explicit left-only
        # F.pad(..., (d_conv - 1, 0)) in forward(), gets the same strictly
        # causal result without the wasted right-side computation.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=0, bias=True,
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

    def _chunked_scan(
        self, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, x_conv: torch.Tensor,
    ) -> torch.Tensor:
        """Computes state_t at every position via a chunked parallel scan.
        dt, x_conv: [batch, seq_len, d_inner]. A: [d_inner, d_state].
        B: [batch, seq_len, d_state]. Returns states: [batch, seq_len, d_inner, d_state].
        """
        batch, seq_len, d_inner = dt.shape
        d_state = A.shape[-1]
        device, dtype = dt.device, dt.dtype

        all_states = torch.empty(batch, seq_len, d_inner, d_state, device=device, dtype=dtype)
        carry = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)

        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            L = end - start

            dt_chunk = dt[:, start:end, :]          # [batch, L, d_inner]
            x_chunk = x_conv[:, start:end, :]        # [batch, L, d_inner]
            B_chunk = B[:, start:end, :]              # [batch, L, d_state]

            # log_a[b,t,d,n] = dt[b,t,d] * A[d,n]  (since a_t = exp(dt_t * A), this
            # IS log(a_t) directly -- no log(exp(...)) round trip needed).
            log_a = dt_chunk.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [batch, L, d_inner, d_state]
            log_decay = torch.cumsum(log_a, dim=1)  # [batch, L, d_inner, d_state]

            # b_t[b,t,d,n] = dt[b,t,d] * x_conv[b,t,d] * B[b,t,n]
            b_term = (dt_chunk * x_chunk).unsqueeze(-1) * B_chunk.unsqueeze(2)  # [batch, L, d_inner, d_state]

            # Contribution carried in from the previous chunk's final state.
            carry_contrib = carry.unsqueeze(1) * torch.exp(log_decay)  # [batch, L, d_inner, d_state]

            # Intra-chunk contribution via the lower-triangular decay-ratio
            # matrix: decay_ratio[b,t,k,d,n] = exp(log_decay[t] - log_decay[k])
            # for k <= t, else 0. Clamped before exp() to avoid overflow for
            # the (masked-out, k>t) entries where the difference can be large
            # and positive.
            log_decay_t = log_decay.unsqueeze(2)  # [batch, L, 1, d_inner, d_state]
            log_decay_k = log_decay.unsqueeze(1)  # [batch, 1, L, d_inner, d_state]
            tri_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
            tri_mask = tri_mask.view(1, L, L, 1, 1)
            log_diff = torch.clamp(log_decay_t - log_decay_k, max=0.0)
            decay_ratio = torch.where(tri_mask, torch.exp(log_diff), torch.zeros((), device=device, dtype=dtype))

            # intra[b,t,d,n] = sum_k decay_ratio[b,t,k,d,n] * b_term[b,k,d,n]
            intra = torch.einsum("btkdn,bkdn->btdn", decay_ratio, b_term)

            chunk_states = carry_contrib + intra
            all_states[:, start:end, :, :] = chunk_states
            carry = chunk_states[:, -1, :, :]

        return all_states

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

        # Causal depthwise conv: transpose to [batch, d_inner, seq_len], then
        # pad on the left only (kernel_size - 1 positions) so every output
        # position only ever sees itself and earlier positions -- strictly
        # causal without computing (and discarding) right-side padding.
        x_main_t = x_main.transpose(1, 2)
        x_main_t = F.pad(x_main_t, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_main_t)
        x_conv = F.silu(x_conv.transpose(1, 2))  # [batch, seq_len, d_inner]

        # Selective parameters, input-dependent per position.
        proj = self.x_proj(x_conv)  # [batch, seq_len, 2*d_state + 1]
        B, C, dt_raw = torch.split(proj, [self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))  # [batch, seq_len, d_inner], always positive

        A = -torch.exp(self.A_log)  # [d_inner, d_state], negative for stability

        states = self._chunked_scan(dt, A, B, x_conv)  # [batch, seq_len, d_inner, d_state]
        y = torch.einsum("btdn,btn->btd", states, C)  # [batch, seq_len, d_inner]
        y = y + x_conv * self.D  # skip connection (D is a per-channel scalar)

        y = y * F.silu(gate)  # gating
        out = self.out_proj(y)
        return out
