"""Grouped Query Attention (GQA) with RoPE. Uses flash_attn if installed and the
tensors are on CUDA in fp16/bf16; otherwise falls back to torch's native scaled
dot product attention (SDPA) with a causal mask. No custom CUDA kernels are
written by ats itself in either path."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ats.model.quantization import make_linear
from ats.model.rope import RotaryEmbedding, apply_rotary_pos_emb
from ats.model.swa import generate_swa_mask

logger = logging.getLogger("ats.model.attention")

try:
    from flash_attn import flash_attn_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_func = None
    _FLASH_ATTN_AVAILABLE = False

PastKeyValue = Tuple[torch.Tensor, torch.Tensor]


def build_incremental_causal_mask(
    seq_len: int, past_len: int, device: torch.device, window_size: Optional[int] = None,
) -> torch.Tensor:
    """Mask for attending `seq_len` new query positions against a KV cache
    of `past_len` prior positions plus the `seq_len` new key/value positions
    (total_len = past_len + seq_len). New tokens attend to cached/new
    positions strictly at or before their own absolute position (causal),
    additionally restricted to the last `window_size` positions if given.
    Returns a boolean [seq_len, total_len] mask (True = attend), matching
    torch.nn.functional.scaled_dot_product_attention's convention.

    Neither is_causal=True nor is_causal=False expresses this pattern:
    is_causal=True assumes query position i and key position i are the SAME
    absolute position (wrong here, since queries start at past_len, not 0);
    is_causal=False would incorrectly let new tokens attend to each other
    non-causally (including "future" new tokens), leaking information
    during multi-token continuation decoding.
    """
    total_len = past_len + seq_len
    query_positions = torch.arange(past_len, total_len, device=device).unsqueeze(1)  # [seq_len, 1]
    key_positions = torch.arange(total_len, device=device).unsqueeze(0)  # [1, total_len]
    distance = query_positions - key_positions  # [seq_len, total_len]
    mask = distance >= 0  # causal: key at or before query's absolute position
    if window_size is not None:
        mask = mask & (distance < window_size)
    return mask


def build_padding_causal_mask(
    attention_mask: torch.Tensor, seq_len: int, is_causal: bool, device: torch.device,
) -> torch.Tensor:
    """Combines a [batch, seq_len] padding mask (1 = attend, 0 = pad) with an
    optional causal mask into the [batch, 1, seq_len, seq_len] boolean shape
    SDPA expects for attn_mask (True = attend). Passing attention_mask to
    SDPA on its own -- as a raw key-only mask -- silently drops causality:
    SDPA does not add causal masking on its own just because a mask is
    present, so without this the model would attend to future positions
    whenever a padding mask is supplied, with no error to signal it."""
    batch = attention_mask.shape[0]
    key_mask = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]  # [batch,1,1,seq_len]
    mask = key_mask.expand(batch, 1, seq_len, seq_len)
    if is_causal:
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        mask = mask & causal
    return mask


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 4096,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_swa: bool = False,
        swa_window_size: int = 4096,
        quantization: str = "none",
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"GroupedQueryAttention: hidden_size ({hidden_size}) must be divisible "
                f"by num_heads ({num_heads})."
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"GroupedQueryAttention: num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads}) for grouped-query attention."
            )
        if use_swa and swa_window_size <= 0:
            raise ValueError(
                f"swa_window_size must be positive when use_swa=True, got {swa_window_size}."
            )
        self.use_swa = use_swa
        self.swa_window_size = swa_window_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_p = dropout
        self.use_flash_attention = use_flash_attention and _FLASH_ATTN_AVAILABLE
        if use_flash_attention and not _FLASH_ATTN_AVAILABLE:
            logger.warning(
                "model.use_flash_attention=True but the flash_attn package is not "
                "installed; falling back to torch.nn.functional.scaled_dot_product_attention."
            )

        self.q_proj = make_linear(hidden_size, num_heads * self.head_dim, quantization, bias=False)
        self.k_proj = make_linear(hidden_size, num_kv_heads * self.head_dim, quantization, bias=False)
        self.v_proj = make_linear(hidden_size, num_kv_heads * self.head_dim, quantization, bias=False)
        self.o_proj = make_linear(num_heads * self.head_dim, hidden_size, quantization, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len, rope_theta)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        # x: [batch, num_kv_heads, seq_len, head_dim] -> [batch, num_kv_heads * n_rep, seq_len, head_dim]
        if n_rep == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
        force_full_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PastKeyValue]]:
        if x.dim() != 3:
            raise ValueError(
                f"GroupedQueryAttention expected input of shape "
                f"[batch, seq_len, hidden_size], got shape {tuple(x.shape)}."
            )
        batch, seq_len, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"GroupedQueryAttention expected hidden_size={self.hidden_size}, "
                f"got {hidden_size} (full shape {tuple(x.shape)})."
            )

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if past_key_value is None else past_key_value[0].shape[2]
        total_len = past_len + seq_len
        cos, sin = self.rotary_emb(total_len, device=x.device, dtype=x.dtype)
        cos, sin = cos[past_len:total_len], sin[past_len:total_len]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_past_key_value = (k, v) if use_cache else None

        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)

        is_causal = past_key_value is None and attention_mask is None and seq_len > 1
        apply_swa = self.use_swa and not force_full_attention

        # Multi-token continuation against an existing KV cache (seq_len>1
        # with past_key_value set) needs an explicit mask: new tokens must
        # always see every cached position, and be causal only among
        # themselves. Neither is_causal=True (assumes queries start at
        # position 0) nor is_causal=False (would let new tokens see each
        # other non-causally) expresses this correctly, and flash_attn's
        # causal/window_size flags can't express it either, so this case
        # always uses SDPA with an explicit mask regardless of
        # use_flash_attention.
        needs_incremental_mask = (
            past_key_value is not None and seq_len > 1 and attention_mask is None
        )

        if needs_incremental_mask:
            incremental_mask = build_incremental_causal_mask(
                seq_len, past_len, x.device, window_size=self.swa_window_size if apply_swa else None,
            )
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=incremental_mask,
                dropout_p=self.dropout_p if self.training else 0.0, is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        elif self.use_flash_attention and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            q_bshd = q.transpose(1, 2)
            k_bshd = k.transpose(1, 2)
            v_bshd = v.transpose(1, 2)
            flash_kwargs = dict(
                dropout_p=self.dropout_p if self.training else 0.0, causal=is_causal,
            )
            if apply_swa:
                # flash_attn>=2.2 accepts window_size=(left, right); (-1, -1) means
                # unbounded. We restrict the left (past) window and leave right
                # unbounded since attention is causal (right is masked by `causal`).
                try:
                    attn_out = flash_attn_func(
                        q_bshd, k_bshd, v_bshd,
                        window_size=(self.swa_window_size - 1, 0), **flash_kwargs,
                    )
                except TypeError:
                    logger.warning(
                        "Installed flash_attn version does not support the window_size "
                        "argument; falling back to SDPA with a manual banded mask for SWA."
                    )
                    swa_mask = generate_swa_mask(seq_len, self.swa_window_size, x.device)
                    attn_out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=swa_mask,
                        dropout_p=self.dropout_p if self.training else 0.0, is_causal=False,
                    ).transpose(1, 2)
            else:
                attn_out = flash_attn_func(q_bshd, k_bshd, v_bshd, **flash_kwargs)
            attn_out = attn_out.reshape(batch, seq_len, self.num_heads * self.head_dim)
        else:
            attn_mask = None
            use_is_causal = is_causal
            if attention_mask is not None:
                # attention_mask is a [batch, seq_len] padding mask (long or
                # bool, 1 = attend / 0 = pad) straight from the dataloader.
                # SDPA rejects long dtype outright, and -- independent of
                # dtype -- passing a key-only mask on its own silently drops
                # causal masking (is_causal above was already forced False
                # the moment attention_mask is not None), so the causal
                # component has to be folded back in here explicitly.
                attn_mask = build_padding_causal_mask(
                    attention_mask, seq_len, is_causal=(past_key_value is None and seq_len > 1),
                    device=x.device,
                )
                use_is_causal = False
            if apply_swa and past_key_value is None:
                swa_mask = generate_swa_mask(seq_len, self.swa_window_size, x.device)
                attn_mask = swa_mask if attn_mask is None else (attn_mask & swa_mask)
                use_is_causal = False
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=use_is_causal,
            )
            attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)

        out = self.o_proj(attn_out)
        return out, new_past_key_value