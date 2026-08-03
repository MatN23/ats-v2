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

    Per layer: attention (q,k,v,o proj) + FFN (SwiGLU: gate_up + down) params.
    Plus embedding (and untied lm_head, if applicable).
    """
    if not model.is_resolved():
        raise ConfigError(
            "estimate_param_count requires a resolved ModelConfig "
            "(hidden_size/num_layers/etc must not be None)."
        )
    h = model.hidden_size
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
