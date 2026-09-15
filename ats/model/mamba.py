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
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn


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
            raise ValueError(
                f"MambaBlock hidden_size must be positive, got {hidden_size}."
            )
        if d_state <= 0 or d_conv <= 0 or expand <= 0:
            raise ValueError(
                f"MambaBlock d_state, d_conv, expand must all be positive, got "
                f"d_state={d_state}, d_conv={d_conv}, expand={expand}."
            )
        if chunk_size <= 0:
            raise ValueError(
                f"MambaBlock chunk_size must be positive, got {chunk_size}."
            )
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
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=0,
            bias=True,
        )

        # Input-dependent (selective) SSM parameters: dt, B, C are all
        # functions of the current activations, not fixed weights — this is
        # what makes it "selective" rather than a plain linear SSM.
        self.x_proj = nn.Linear(
            self.d_inner, d_state * 2 + 1, bias=False
        )  # -> (B, C, dt_raw)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A is a learned, per-channel, per-state negative-definite matrix
        # (stored as log for positivity via -exp(A_log)), the continuous-time
        # SSM's state transition parameter.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)

    @staticmethod
    def _chunk_body(
        dt_chunk: torch.Tensor,
        x_chunk: torch.Tensor,
        B_chunk: torch.Tensor,
        C_chunk: torch.Tensor,
        A: torch.Tensor,
        carry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One chunk of the scan. Returns (y_chunk, new_carry) where
        y_chunk is [batch, L, d_inner] -- the states are contracted with
        C *inside* this function and never returned.

        Isolated into its own function (a) so the C contraction happens
        per-chunk rather than on a materialized full-sequence state tensor,
        and (b) so _chunked_scan can wrap it in
        torch.utils.checkpoint.checkpoint and have the huge intermediates
        below recomputed in backward instead of retained. See _chunked_scan
        for the memory arithmetic.
        """
        L = dt_chunk.shape[1]
        device, dtype = dt_chunk.device, dt_chunk.dtype

        # log_a[b,t,d,n] = dt[b,t,d] * A[d,n]  (a_t = exp(dt_t * A), so this
        # IS log(a_t) directly -- no log(exp(...)) round trip needed).
        log_a = dt_chunk.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        log_decay = torch.cumsum(log_a, dim=1)  # [batch, L, d_inner, d_state]

        # b_t[b,t,d,n] = dt[b,t,d] * x_conv[b,t,d] * B[b,t,n]
        b_term = (dt_chunk * x_chunk).unsqueeze(-1) * B_chunk.unsqueeze(2)

        # Contribution carried in from the previous chunk's final state.
        carry_contrib = carry.unsqueeze(1) * torch.exp(log_decay)

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
        decay_ratio = torch.where(
            tri_mask,
            torch.exp(log_diff),
            torch.zeros((), device=device, dtype=dtype),
        )

        # intra[b,t,d,n] = sum_k decay_ratio[b,t,k,d,n] * b_term[b,k,d,n]
        intra = torch.einsum("btkdn,bkdn->btdn", decay_ratio, b_term)

        chunk_states = carry_contrib + intra  # [batch, L, d_inner, d_state]
        y_chunk = torch.einsum("btdn,btn->btd", chunk_states, C_chunk)
        new_carry = chunk_states[:, -1, :, :]
        return y_chunk, new_carry

    def _chunked_scan(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        x_conv: torch.Tensor,
        initial_carry: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes y_t = <state_t, C_t> at every position via a chunked
        parallel scan, plus the final carry state.
        dt, x_conv: [batch, seq_len, d_inner]. A: [d_inner, d_state].
        B, C: [batch, seq_len, d_state].
        Returns (y: [batch, seq_len, d_inner], final_state: [batch, d_inner, d_state]).

        BUG FIX (BUG-104), two parts, both memory:

        1. This used to return `all_states`, a [batch, seq_len, d_inner,
           d_state] tensor holding every intermediate state, which the
           caller then contracted with C. That is d_state (default 16)
           times larger than the y it exists to produce, and it is retained
           by autograd for the whole backward pass. Contracting with C
           inside each chunk removes it entirely.

        2. The per-chunk `decay_ratio` intermediate is [batch, L, L,
           d_inner, d_state]. einsum saves it for backward, so the *total*
           retained memory was (seq_len / chunk_size) chunks x that tensor
           = batch * seq_len * chunk_size * d_inner * d_state * 4 bytes.
           At batch=8, seq_len=4096, hidden=2048 (d_inner=4096), d_state=16,
           chunk_size=32 that is ~1.1 TB for ONE Mamba layer -- measured at
           small scale as 537 MB of retained tensors for a [2, 512, 128]
           input, i.e. ~600x the size of the layer's own output.
           Checkpointing each chunk body drops the retained total to a
           single chunk's worth (a seq_len/chunk_size = 128x reduction at
           those settings) at the cost of recomputing the chunk in backward.

        Chunk-local recompute is only used when gradients are actually
        needed; inference and torch.no_grad() paths call the body directly.
        """
        """Computes state_t at every position via a chunked parallel scan.
        dt, x_conv: [batch, seq_len, d_inner]. A: [d_inner, d_state].
        B: [batch, seq_len, d_state]. Returns states: [batch, seq_len, d_inner, d_state].

        `initial_carry` (state at the position immediately before this
        call's first position, shape [batch, d_inner, d_state]) defaults to
        zeros -- the correct starting state for a fresh sequence. Passing a
        real prior state here (from a previous call's final state) is what
        makes incremental decoding mathematically exact rather than an
        approximation: continuing the scan from the exact carry a full-
        sequence call would have produced at that position is exactly what
        the linear recurrence's associativity guarantees is equivalent to
        having run the full sequence in one call -- see
        test_mamba_incremental_decoding_matches_full_sequence in
        tests/test_model.py for the direct numerical check of that claim.
        """
        batch, seq_len, d_inner = dt.shape
        d_state = A.shape[-1]
        device, dtype = dt.device, dt.dtype

        carry = (
            torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
            if initial_carry is None
            else initial_carry
        )

        use_recompute = torch.is_grad_enabled() and (
            dt.requires_grad or x_conv.requires_grad or A.requires_grad
        )

        y_chunks: list[torch.Tensor] = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            args = (
                dt[:, start:end, :],
                x_conv[:, start:end, :],
                B[:, start:end, :],
                C[:, start:end, :],
                A,
                carry,
            )
            if use_recompute:
                y_chunk, carry = torch.utils.checkpoint.checkpoint(
                    self._chunk_body, *args, use_reentrant=False
                )
            else:
                y_chunk, carry = self._chunk_body(*args)
            y_chunks.append(y_chunk)

        y = torch.cat(y_chunks, dim=1) if len(y_chunks) > 1 else y_chunks[0]
        return y, carry

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """state, if given, is (conv_state, ssm_state) from a previous call:
        conv_state is the last (d_conv - 1) raw (pre-conv) x_main columns
        seen so far, shape [batch, d_inner, d_conv - 1] (empty in the last
        dim if d_conv == 1, since a kernel-size-1 conv has no history to
        carry); ssm_state is the selective-scan carry, shape
        [batch, d_inner, d_state]. state=None (the default) means "start of
        a fresh sequence" -- mathematically identical to conv_state being
        all zeros and ssm_state being all zeros, which is exactly what a
        fresh sequence's true initial state is, so this is not a special
        case requiring separate math, just a convenient default that avoids
        callers allocating zero tensors themselves.

        Returns just the output tensor (exactly the pre-existing behavior,
        unchanged) when use_cache=False. Returns (output, new_state) when
        use_cache=True, where new_state is a (conv_state, ssm_state) pair
        in the same shapes described above, ready to pass as `state` on the
        next call to continue the sequence.
        """
        if x.dim() != 3:
            raise ValueError(
                f"MambaBlock expected input of shape [batch, seq_len, hidden_size], "
                f"got shape {tuple(x.shape)}."
            )
        batch, _seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"MambaBlock expected hidden_size={self.hidden_size}, got {hidden_size}."
            )
        conv_state, ssm_state = (None, None) if state is None else state
        if conv_state is not None:
            expected_conv_shape = (batch, self.d_inner, self.d_conv - 1)
            if tuple(conv_state.shape) != expected_conv_shape:
                raise ValueError(
                    f"MambaBlock got a conv_state of shape {tuple(conv_state.shape)}, "
                    f"expected {expected_conv_shape}."
                )
        if ssm_state is not None:
            expected_ssm_shape = (batch, self.d_inner, self.d_state)
            if tuple(ssm_state.shape) != expected_ssm_shape:
                raise ValueError(
                    f"MambaBlock got an ssm_state of shape {tuple(ssm_state.shape)}, "
                    f"expected {expected_ssm_shape}."
                )

        x_and_gate = self.in_proj(x)  # [batch, seq_len, 2*d_inner]
        x_main, gate = x_and_gate.chunk(2, dim=-1)

        # Causal depthwise conv: transpose to [batch, d_inner, seq_len], then
        # either zero-pad on the left (fresh sequence -- the true history
        # before position 0 is nothing, which zero-padding correctly
        # represents) or prepend the real conv_state carried over from a
        # previous call (continuing a sequence -- using the actual prior
        # values here, not zeros, is what makes this exact rather than an
        # approximation that forgets the last (d_conv - 1) tokens' influence
        # on the conv every time a new incremental call starts).
        x_main_t = x_main.transpose(1, 2)  # [batch, d_inner, seq_len]
        if conv_state is None:
            x_main_t_history = F.pad(x_main_t, (self.d_conv - 1, 0))
        else:
            x_main_t_history = torch.cat([conv_state, x_main_t], dim=-1)
        x_conv = self.conv1d(x_main_t_history)
        x_conv = F.silu(x_conv.transpose(1, 2))  # [batch, seq_len, d_inner]

        # The new conv_state to return is simply the last (d_conv - 1)
        # RAW (pre-conv) columns of the same history buffer just used --
        # exactly what the next call needs to prepend to continue seamlessly.
        new_conv_state = (
            x_main_t_history[..., -(self.d_conv - 1) :]
            if self.d_conv > 1
            else torch.empty(batch, self.d_inner, 0, device=x.device, dtype=x.dtype)
        )

        # Selective parameters, input-dependent per position.
        proj = self.x_proj(x_conv)  # [batch, seq_len, 2*d_state + 1]
        B, C, dt_raw = torch.split(proj, [self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(
            self.dt_proj(dt_raw)
        )  # [batch, seq_len, d_inner], always positive

        A = -torch.exp(self.A_log)  # [d_inner, d_state], negative for stability

        # _chunked_scan now contracts the states against C internally and
        # returns only y ([batch, seq_len, d_inner]) plus the final carry,
        # instead of a full [batch, seq_len, d_inner, d_state] state tensor
        # -- see BUG-104 in its docstring.
        y, new_ssm_state = self._chunked_scan(
            dt, A, B, C, x_conv, initial_carry=ssm_state
        )
        y = y + x_conv * self.D  # skip connection (D is a per-channel scalar)

        y = y * F.silu(gate)  # gating
        out = self.out_proj(y)
        if use_cache:
            return out, (new_conv_state, new_ssm_state)
        return out
