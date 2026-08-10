"""Tests for ats.parallelism.auto_parallel: parameter count estimation
(dense, MLA, MoE) and strategy resolution."""

from __future__ import annotations

import pytest

from ats.config.schema import ATSConfig, DataConfig, DataSource, ModelConfig, ParallelismConfig, TrainingConfig
from ats.parallelism.auto_parallel import estimate_param_count, resolve_strategy


def _model_config(**overrides) -> ModelConfig:
    defaults = dict(
        hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=8,
        intermediate_size=2048, vocab_size=32000, max_seq_len=2048,
    )
    return ModelConfig(**{**defaults, **overrides})


def _ats_config(model: ModelConfig, **parallelism_overrides) -> ATSConfig:
    return ATSConfig(
        model=model,
        training=TrainingConfig(max_steps=100, learning_rate=1e-4, warmup_steps=10),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=2048),
        parallelism=ParallelismConfig(**parallelism_overrides),
    )


def test_estimate_param_count_requires_resolved_config():
    unresolved = ModelConfig()
    with pytest.raises(ValueError):
        estimate_param_count(unresolved)


def test_estimate_param_count_positive_and_scales_with_layers():
    small = estimate_param_count(_model_config(num_layers=4))
    large = estimate_param_count(_model_config(num_layers=8))
    assert small > 0
    assert large > small
    assert large == pytest.approx(2 * small, rel=0.05)


def test_estimate_param_count_mla_uses_mla_formula_not_gqa():
    """Regression test: estimate_param_count previously applied the GQA
    attention formula unconditionally, even to MLA models, which have a
    completely different (compressed-latent) parameter structure. An MLA
    model's estimated param count must differ from what the same
    architecture fields would give under the GQA formula."""
    mla_config = _model_config(use_mla=True, mla_latent_dim=128)
    dense_config = _model_config(use_mla=False)

    mla_count = estimate_param_count(mla_config)
    dense_count = estimate_param_count(dense_config)

    assert mla_count != dense_count
    # MLA's compressed attention should use noticeably fewer attention
    # parameters than full GQA for a reasonable compression ratio.
    assert mla_count < dense_count


def test_estimate_param_count_mla_scales_with_latent_dim():
    small_latent = estimate_param_count(_model_config(use_mla=True, mla_latent_dim=32))
    large_latent = estimate_param_count(_model_config(use_mla=True, mla_latent_dim=256))
    assert large_latent > small_latent


def test_estimate_param_count_moe_scales_with_num_experts():
    few_experts = estimate_param_count(_model_config(use_moe=True, num_experts=2, moe_top_k=1))
    many_experts = estimate_param_count(_model_config(use_moe=True, num_experts=16, moe_top_k=1))
    assert many_experts > few_experts


def test_estimate_param_count_moe_includes_gate_params():
    """Regression test: the MoE gate/router's own parameters
    (hidden_size * num_experts) were previously omitted entirely."""
    no_moe = estimate_param_count(_model_config(use_moe=False))
    with_moe_one_expert_equivalent = estimate_param_count(
        _model_config(use_moe=True, num_experts=1, moe_top_k=1)
    )
    # With num_experts=1, the "extra experts" FFN contribution is zero, so
    # any remaining difference from the dense case must come from the gate.
    assert with_moe_one_expert_equivalent > no_moe


def test_resolve_strategy_single_gpu_is_zero0():
    config = _ats_config(_model_config(), strategy="auto", gpus=1, nodes=1)
    assert resolve_strategy(config) == "deepspeed_zero0"


def test_resolve_strategy_moe_multi_node_is_deepspeed_moe():
    config = _ats_config(
        _model_config(use_moe=True, num_experts=8, moe_top_k=2),
        strategy="auto", gpus=8, nodes=2,
    )
    assert resolve_strategy(config) == "deepspeed_moe"


def test_resolve_strategy_small_model_multi_gpu_is_zero2():
    config = _ats_config(_model_config(num_layers=8), strategy="auto", gpus=4, nodes=1)
    assert resolve_strategy(config) == "deepspeed_zero2"


def test_resolve_strategy_large_model_is_zero3():
    huge = _model_config(hidden_size=8192, num_layers=80, num_heads=64, num_kv_heads=8, intermediate_size=28672)
    config = _ats_config(huge, strategy="auto", gpus=16, nodes=2)
    assert resolve_strategy(config) == "deepspeed_zero3"


def test_resolve_strategy_explicit_strategy_is_passed_through():
    config = _ats_config(_model_config(), strategy="deepspeed_zero1", gpus=1, nodes=1)
    assert resolve_strategy(config) == "deepspeed_zero1"
