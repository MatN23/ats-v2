"""Multi-Head Latent Attention (MLA), DeepSeek-V2 style.

Keys and values are derived from a shared, compressed low-dimensional latent
`c` (dimension `latent_dim`, typically hidden_size // 4 or // 8) rather than
from full per-head projections. This is what shrinks the KV cache: instead of
caching (num_kv_heads * head_dim) values per token for K and again for V, we
cache only `latent_dim` (+ a small shared RoPE slice) values per token, total.

RoPE is decoupled: it is applied only to a small extra set of
`rope_head_dim` dimensions per head that are concatenated onto the
content (non-positional) part of q/k, following DeepSeek-V2. The main
compressed latent itself carries no explicit position information, which is
what makes it safe to cache and reuse across positions.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ats.model.rope import RotaryEmbedding, apply_rotary_pos_emb

logger = logging.getLogger("ats.model.mla")

# Cache tuple for MLA holds ONLY the compressed latent (and the shared
# decoupled-RoPE key slice), never full per-head K/V tensors.
MLAPastState = Tuple[torch.Tensor, torch.Tensor]  # (latent_cache, rope_k_cache)


class MLAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        latent_dim: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rope_head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"MLAAttention: hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if latent_dim <= 0 or latent_dim >= hidden_size:
            raise ValueError(
                f"MLAAttention: latent_dim ({latent_dim}) must be in (0, hidden_size) "
                f"— it is meant to compress hidden_size ({hidden_size}). "
                f"Typical choice: hidden_size // 4 or // 8."
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_dim = latent_dim
        self.dropout_p = dropout
        self.rope_head_dim = rope_head_dim if rope_head_dim is not None else max(2, self.head_dim // 4)
        if self.rope_head_dim % 2 != 0:
            self.rope_head_dim += 1

        # Down-projection: hidden_size -> latent_dim. This latent is what's cached.
        self.w_dkv = nn.Linear(hidden_size, latent_dim, bias=False)
        # Up-projections from the shared latent to per-head K and V content.
        self.w_uk = nn.Linear(latent_dim, num_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(latent_dim, num_heads * self.head_dim, bias=False)
        # Query path: its own down/up compression.
        self.w_dq = nn.Linear(hidden_size, latent_dim, bias=False)
        self.w_uq = nn.Linear(latent_dim, num_heads * self.head_dim, bias=False)
        # Decoupled RoPE: a small position-aware slice appended to q/k content,
        # computed directly from x so no position info enters the cached latent.
        self.w_qr = nn.Linear(hidden_size, num_heads * self.rope_head_dim, bias=False)
        self.w_kr = nn.Linear(hidden_size, self.rope_head_dim, bias=False)  # shared across heads

        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.rope_head_dim, max_seq_len, rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[MLAPastState] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[MLAPastState]]:
        if x.dim() != 3:
            raise ValueError(
                f"MLAAttention expected input of shape [batch, seq_len, hidden_size], "
                f"got {tuple(x.shape)}."
            )
        batch, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"MLAAttention expected hidden_size={self.hidden_size}, got {hidden_size}."
            )

        # --- Compressed latent: this, plus k_rope, is the ENTIRE KV cache. ---
        c_kv = self.w_dkv(x)  # [batch, seq_len, latent_dim]
        k_rope = self.w_kr(x)  # [batch, seq_len, rope_head_dim]

        past_len = 0 if past_key_value is None else past_key_value[0].shape[1]
        if past_key_value is not None:
            c_kv_full = torch.cat([past_key_value[0], c_kv], dim=1)
            k_rope_full = torch.cat([past_key_value[1], k_rope], dim=1)
        else:
            c_kv_full = c_kv
            k_rope_full = k_rope
        new_past_key_value: Optional[MLAPastState] = (c_kv_full, k_rope_full) if use_cache else None
        total_len = past_len + seq_len

        # --- Up-project the (full, cached) latent to per-head K/V content ---
        k_content = self.w_uk(c_kv_full).view(batch, total_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_kv_full).view(batch, total_len, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Query path: compress then up-project, current tokens only ---
        c_q = self.w_dq(x)
        q_content = self.w_uq(c_q).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_rope = self.w_qr(x).view(batch, seq_len, self.num_heads, self.rope_head_dim).transpose(1, 2)

        # --- Decoupled RoPE applied only to the small rope slice ---
        cos, sin = self.rotary_emb(total_len, device=x.device, dtype=x.dtype)
        k_rope_expanded = k_rope_full.unsqueeze(1).expand(
            batch, self.num_heads, total_len, self.rope_head_dim
        )
        q_rope_full_pad = F.pad(q_rope, (0, 0, past_len, 0))  # align seq axis for the shared apply call
        q_rope_rotated, k_rope_rotated = apply_rotary_pos_emb(q_rope_full_pad, k_rope_expanded, cos, sin)
        q_rope_rotated = q_rope_rotated[:, :, past_len:total_len, :]

        q = torch.cat([q_content, q_rope_rotated], dim=-1)
        k = torch.cat([k_content, k_rope_rotated], dim=-1)

        is_causal = past_key_value is None and attention_mask is None and seq_len > 1
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(attn_out)
        return out, new_past_key_value

    def cache_size_per_token(self) -> int:
        """Scalars cached per token: compressed latent + shared decoupled-RoPE
        key slice. Used by tests to confirm MLA's cache is smaller than
        standard GQA's 2 * num_kv_heads * head_dim per token."""
        return self.latent_dim + self.rope_head_dim
