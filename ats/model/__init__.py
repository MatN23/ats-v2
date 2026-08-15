"""Transformer model definition: attention, MoE, MoD, SWA, MLA, Mamba, MTP,
quantization, diffusion, FFN, norm, RoPE, init."""

from ats.model.attention import GroupedQueryAttention, build_incremental_causal_mask
from ats.model.diffusion import DiffusionLM, DiffusionOutput
from ats.model.ffn import SwiGLU
from ats.model.mamba import MambaBlock
from ats.model.mla import MLAAttention
from ats.model.mod import MixtureOfDepths
from ats.model.moe import MoELayer
from ats.model.mtp import MultiTokenPredictionHead
from ats.model.norm import RMSNorm
from ats.model.quantization import QuantizedLinear
from ats.model.rope import RotaryEmbedding, apply_rotary_pos_emb
from ats.model.swa import generate_swa_mask, is_full_attention_layer
from ats.model.transformer import (
    ATSTransformer,
    MambaLayer,
    TransformerBlock,
    TransformerOutput,
)

__all__ = [
    "ATSTransformer",
    "DiffusionLM",
    "DiffusionOutput",
    "GroupedQueryAttention",
    "MLAAttention",
    "MambaBlock",
    "MambaLayer",
    "MixtureOfDepths",
    "MoELayer",
    "MultiTokenPredictionHead",
    "QuantizedLinear",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerOutput",
    "apply_rotary_pos_emb",
    "build_incremental_causal_mask",
    "generate_swa_mask",
    "is_full_attention_layer",
]
