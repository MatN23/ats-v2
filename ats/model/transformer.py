"""ATSTransformer: a pre-norm, GQA + RoPE transformer that optionally routes
its FFN through MoE and/or wraps blocks with Mixture-of-Depths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ats.config.schema import ModelConfig
from ats.model.attention import GroupedQueryAttention, PastKeyValue
from ats.model.ffn import SwiGLU
from ats.model.initialization import init_residual_projection, init_weights
from ats.model.mla import MLAAttention
from ats.model.mamba import MambaBlock
from ats.model.mod import MixtureOfDepths
from ats.model.moe import MoELayer
from ats.model.mtp import MultiTokenPredictionHead
from ats.model.norm import RMSNorm
from ats.model.swa import is_full_attention_layer


@dataclass
class TransformerOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor
    past_key_values: Optional[List[Optional[PastKeyValue]]] = None
    mtp_logits: Optional[List[torch.Tensor]] = None


class MambaLayer(nn.Module):
    """Adapts MambaBlock to the same (x, attention_mask, past_key_value,
    use_cache) -> (x, aux_loss, new_past_key_value) call signature used by
    TransformerBlock, so ATSTransformer can mix the two freely. This
    reference implementation does not support KV-cache-based incremental
    decoding (the sequential scan is recomputed over the full sequence each
    call); attention_mask/past_key_value/use_cache are accepted for
    interface compatibility but past_key_value must be None."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = MambaBlock(
            hidden_size=config.hidden_size,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[PastKeyValue]]:
        if past_key_value is not None:
            raise ValueError(
                "MambaLayer does not support KV-cache-based incremental decoding in "
                "this implementation. Fix: run full-sequence forward passes, or disable "
                "use_cache for Mamba layers."
            )
        h = self.input_norm(x)
        mamba_out = self.mamba(h)
        out = x + mamba_out
        aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        return out, aux_loss, None


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        if not config.is_resolved():
            raise ValueError(
                "TransformerBlock received an unresolved ModelConfig (architecture "
                "fields are None). Call ats.config.defaults.apply_size_preset() first."
            )
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.uses_mla = config.use_mla
        self.force_full_attention = config.use_swa and is_full_attention_layer(
            layer_idx, config.swa_full_attention_interval
        )

        if config.use_mla:
            self.attention: nn.Module = MLAAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                latent_dim=config.resolved_mla_latent_dim,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                dropout=config.dropout,
            )
        else:
            self.attention = GroupedQueryAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                dropout=config.dropout,
                use_flash_attention=config.use_flash_attention,
                use_swa=config.use_swa,
                swa_window_size=config.swa_window_size,
            )
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.use_moe:
            self.ffn: nn.Module = MoELayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                top_k=config.moe_top_k,
                capacity_factor=config.moe_capacity_factor,
                load_balancing_weight=config.moe_load_balancing_weight,
            )
            self.ffn_is_moe = True
        else:
            self.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.dropout)
            self.ffn_is_moe = False

        init_residual_projection(self.attention.o_proj, config.num_layers)
        if not self.ffn_is_moe:
            init_residual_projection(self.ffn.down_proj, config.num_layers)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[PastKeyValue] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[PastKeyValue]]:
        residual = x
        h = self.input_norm(x)
        if self.uses_mla:
            attn_out, new_past_key_value = self.attention(
                h, attention_mask=attention_mask, past_key_value=past_key_value, use_cache=use_cache
            )
        else:
            attn_out, new_past_key_value = self.attention(
                h, attention_mask=attention_mask, past_key_value=past_key_value,
                use_cache=use_cache, force_full_attention=self.force_full_attention,
            )
        x = residual + attn_out

        residual = x
        h = self.post_attention_norm(x)
        if self.ffn_is_moe:
            ffn_out, aux_loss = self.ffn(h)
        else:
            ffn_out = self.ffn(h)
            aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        x = residual + ffn_out

        return x, aux_loss, new_past_key_value


class ATSTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not config.is_resolved():
            raise ValueError(
                "ATSTransformer received an unresolved ModelConfig. "
                "Call ats.config.defaults.apply_size_preset(config.model) first, "
                "or pass model.size in the YAML config."
            )
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_scale = math.sqrt(config.hidden_size)

        blocks: List[nn.Module] = []
        for layer_idx in range(config.num_layers):
            if config.use_mamba and (layer_idx + 1) % config.mamba_every_n_layers == 0:
                block: nn.Module = MambaLayer(config)
            else:
                block = TransformerBlock(config, layer_idx)
            if config.use_mod:
                block = MixtureOfDepths(config.hidden_size, block, config.mod_capacity_factor)
            blocks.append(block)
        self.layers = nn.ModuleList(blocks)

        self.uses_mtp = config.use_mtp
        if config.use_mtp:
            self.mtp_head = MultiTokenPredictionHead(
                hidden_size=config.hidden_size, vocab_size=config.vocab_size,
                num_future_tokens=config.mtp_num_tokens,
            )

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(lambda m: init_weights(m, config.num_layers))
        if config.tie_word_embeddings:
            # Re-tie after init since apply() re-initializes embed_tokens.weight
            # in place, which lm_head.weight already aliases (no-op, kept for clarity).
            self.lm_head.weight = self.embed_tokens.weight

    def _run_layers(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List[Optional[PastKeyValue]]],
        use_cache: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[PastKeyValue]]]:
        total_aux_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        new_past_key_values: List[Optional[PastKeyValue]] = []

        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx] if past_key_values is not None else None
            if isinstance(layer, MixtureOfDepths):
                x, aux_loss, new_kv = layer(
                    x, attention_mask=attention_mask, past_key_value=past_kv, use_cache=use_cache,
                )
            else:
                x, aux_loss, new_kv = layer(
                    x, attention_mask=attention_mask, past_key_value=past_kv, use_cache=use_cache,
                )
            total_aux_loss = total_aux_loss + aux_loss
            new_past_key_values.append(new_kv)

        return x, total_aux_loss, new_past_key_values

    def forward_hidden(
        self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the transformer stack directly on precomputed embeddings
        (e.g. noised embeddings from DiffusionLM), skipping the token
        embedding lookup and the LM head, and returns final normed hidden
        states rather than logits. No aux-loss/MoE/MoD routing state is
        threaded through here since diffusion training does not use KV
        caching or autoregressive generation."""
        if inputs_embeds.dim() != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"ATSTransformer.forward_hidden expected inputs_embeds of shape "
                f"[batch, seq_len, {self.config.hidden_size}], got {tuple(inputs_embeds.shape)}."
            )
        x, _aux_loss, _past = self._run_layers(
            inputs_embeds, attention_mask, past_key_values=None, use_cache=False,
        )
        return self.final_norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[PastKeyValue]]] = None,
        use_cache: bool = False,
    ) -> TransformerOutput:
        if input_ids.dim() != 2:
            raise ValueError(
                f"ATSTransformer expected input_ids of shape [batch, seq_len], "
                f"got shape {tuple(input_ids.shape)}."
            )
        if input_ids.max().item() >= self.config.vocab_size or input_ids.min().item() < 0:
            raise ValueError(
                f"input_ids contains token ids outside [0, {self.config.vocab_size}). "
                f"Got min={input_ids.min().item()}, max={input_ids.max().item()}. "
                f"Fix: check your tokenizer's vocab_size matches model.vocab_size."
            )

        x = self.embed_tokens(input_ids) * self.embed_scale
        x, total_aux_loss, new_past_key_values = self._run_layers(
            x, attention_mask, past_key_values, use_cache,
        )
        x = self.final_norm(x)
        logits = self.lm_head(x)

        mtp_logits = self.mtp_head(x) if self.uses_mtp else None

        return TransformerOutput(
            logits=logits,
            aux_loss=total_aux_loss,
            past_key_values=new_past_key_values if use_cache else None,
            mtp_logits=mtp_logits,
        )
