"""Rotary Position Embeddings. Standard, correct implementation, no custom kernels.

Uses the "rotate_half" (non-interleaved-pair) convention, which is the one used
by Llama / Mistral / GPT-NeoX style models and is mathematically equivalent to
the interleaved formulation for the purposes of relative-position encoding, as
long as the same convention is used consistently between q/k construction and
apply_rotary_pos_emb. Dynamic cache extension is supported: if a sequence
longer than the cached max_seq_len is requested, cos/sin are recomputed.
"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"RotaryEmbedding dim must be even, got {dim}. "
                f"Fix: head_dim (hidden_size // num_heads) must be even."
            )
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len = 0
        self._cached_cos: torch.Tensor
        self._cached_sin: torch.Tensor
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("_cached_cos", emb.cos(), persistent=False)
        self.register_buffer("_cached_sin", emb.sin(), persistent=False)
        self._cached_seq_len = seq_len

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            raise ValueError(f"RotaryEmbedding requires seq_len > 0, got {seq_len}.")
        if seq_len > self._cached_seq_len:
            # Bug 11 fix: growing to exactly `seq_len` every time means
            # incremental decoding (seq_len creeping up one token at a time)
            # rebuilds the whole cache on every single step. Doubling
            # instead amortizes that rebuild cost across many steps.
            new_len = max(
                seq_len,
                self._cached_seq_len * 2 if self._cached_seq_len > 0 else seq_len,
            )
            self._build_cache(new_len)
        cos = self._cached_cos[:seq_len].to(device=device, dtype=dtype)
        sin = self._cached_sin[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k: [batch, num_heads, seq_len, head_dim]. cos, sin: [seq_len, head_dim]."""
    if q.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"apply_rotary_pos_emb: q head_dim ({q.shape[-1]}) does not match "
            f"cos/sin dim ({cos.shape[-1]})."
        )
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
