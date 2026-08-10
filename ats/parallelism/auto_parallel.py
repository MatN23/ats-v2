"""Selects a concrete DeepSpeed parallelism strategy from
ParallelismConfig.strategy == "auto", based on the actual GPU count, node
count, MoE flag, and an estimated parameter count derived from the resolved
ModelConfig. This is real arithmetic on the config, not a hardcoded lookup.
"""

from __future__ import annotations

from ats.config.schema import ATSConfig, ConfigError

_THIRTEEN_BILLION = 13_000_000_000


def estimate_param_count(model) -> int:
    """Rough dense-transformer parameter count from architecture fields.
    Used only for parallelism-strategy selection, not for exact reporting.

    Per layer: attention (GQA q/k/v/o, or MLA's compressed-latent
    projections if model.use_mla) + FFN (SwiGLU: gate_up + down) params.
    Plus embedding (and untied lm_head, if applicable).
    """
    if not model.is_resolved():
        raise ConfigError(
            "estimate_param_count requires a resolved ModelConfig "
            "(hidden_size/num_layers/etc must not be None)."
        )
    h = model.hidden_size

    if model.use_mla:
        # MLA's parameter structure is completely different from GQA's: no
        # separate per-head K/V projections, instead a shared compressed
        # latent (w_dkv/w_dq down-projections, w_uk/w_uv/w_uq up-projections)
        # plus small decoupled-RoPE projections (w_qr/w_kr). See
        # ats/model/mla.py for the exact layer definitions this mirrors.
        latent_dim = model.resolved_mla_latent_dim
        head_dim = h // model.num_heads
        rope_head_dim = max(2, head_dim // 4)
        if rope_head_dim % 2 != 0:
            rope_head_dim += 1
        attn_params_per_layer = (
            (h * latent_dim)      # w_dkv
            + (latent_dim * h)    # w_uk
            + (latent_dim * h)    # w_uv
            + (h * latent_dim)    # w_dq
            + (latent_dim * h)    # w_uq
            + (h * model.num_heads * rope_head_dim)  # w_qr
            + (h * rope_head_dim)  # w_kr
            + (h * h)              # o_proj
        )
    else:
        kv_dim = (model.hidden_size // model.num_heads) * model.num_kv_heads
        attn_params_per_layer = (h * h) + (h * kv_dim) + (h * kv_dim) + (h * h)

    ffn_params_per_layer = (h * 2 * model.intermediate_size) + (model.intermediate_size * h)
    per_layer = attn_params_per_layer + ffn_params_per_layer
    total = per_layer * model.num_layers

    embedding_params = model.vocab_size * h
    total += embedding_params
    if not model.tie_word_embeddings:
        total += embedding_params

    if model.use_moe:
        # Only count the additional experts beyond the one already priced into
        # ffn_params_per_layer above, across every layer.
        extra_experts = max(0, model.num_experts - 1)
        total += ffn_params_per_layer * extra_experts * model.num_layers
        # The gate/router itself (hidden_size -> num_experts) is small
        # relative to expert FFN params but previously omitted entirely.
        total += h * model.num_experts * model.num_layers

    return total


def resolve_strategy(config: ATSConfig) -> str:
    """Return a concrete strategy string, resolving "auto" using GPU count,
    node count, MoE flag, and estimated parameter count."""
    strategy = config.parallelism.strategy
    if strategy != "auto":
        return strategy

    gpus = config.parallelism.gpus
    nodes = config.parallelism.nodes

    if config.model.use_moe and nodes > 1:
        return "deepspeed_moe"

    if gpus == 1 and nodes == 1:
        return "deepspeed_zero0"

    param_count = estimate_param_count(config.model)

    if gpus <= 8 and param_count <= _THIRTEEN_BILLION:
        return "deepspeed_zero2"

    return "deepspeed_zero3"
