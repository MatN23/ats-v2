"""Tests for ats.utils.memory.estimate_memory: sane numbers for known configs,
monotonicity in the directions that matter, and the MemoryReport contract."""

from __future__ import annotations

import pytest

from ats.config.schema import ATSConfig, DataConfig, DataSource, ModelConfig, ParallelismConfig, TrainingConfig
from ats.utils.memory import estimate_memory


def _config(**model_overrides) -> ATSConfig:
    defaults = dict(
        hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=8,
        intermediate_size=2048, vocab_size=32000, max_seq_len=2048,
    )
    model = ModelConfig(**{**defaults, **model_overrides})
    return ATSConfig(
        model=model,
        training=TrainingConfig(max_steps=100, learning_rate=1e-4, warmup_steps=10, micro_batch_size=4),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=2048),
        parallelism=ParallelismConfig(gpus=1, nodes=1),
    )


def test_estimate_memory_returns_positive_components():
    report = estimate_memory(_config())
    assert report.model_gb > 0
    assert report.optimizer_gb > 0
    assert report.activation_gb > 0
    assert report.total_gb == pytest.approx(report.model_gb + report.optimizer_gb + report.activation_gb)


def test_estimate_memory_scales_with_parameter_count():
    small = estimate_memory(_config(hidden_size=512, intermediate_size=2048))
    large = estimate_memory(_config(hidden_size=2048, intermediate_size=8192, num_heads=16, num_kv_heads=16))
    assert large.model_gb > small.model_gb
    assert large.optimizer_gb > small.optimizer_gb


def test_estimate_memory_scales_with_batch_size():
    small_batch = estimate_memory(_config(), target_batch_size=1)
    large_batch = estimate_memory(_config(), target_batch_size=16)
    assert large_batch.activation_gb > small_batch.activation_gb
    # Activation memory should scale roughly linearly with batch size.
    assert large_batch.activation_gb == pytest.approx(small_batch.activation_gb * 16, rel=0.05)


def test_estimate_memory_gradient_checkpointing_reduces_activations():
    without_ckpt = estimate_memory(_config(gradient_checkpointing=False))
    with_ckpt = estimate_memory(_config(gradient_checkpointing=True))
    assert with_ckpt.activation_gb < without_ckpt.activation_gb


def test_estimate_memory_zero_stage_shards_optimizer_across_gpus():
    model = ModelConfig(
        hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=8,
        intermediate_size=2048, vocab_size=32000, max_seq_len=2048,
    )
    single_gpu = ATSConfig(
        model=model,
        training=TrainingConfig(max_steps=100, learning_rate=1e-4, warmup_steps=10, micro_batch_size=4),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=2048),
        parallelism=ParallelismConfig(strategy="deepspeed_zero1", gpus=1, nodes=1),
    )
    multi_gpu = single_gpu.model_copy(
        update={"parallelism": ParallelismConfig(strategy="deepspeed_zero1", gpus=8, nodes=1)}
    )
    report_single = estimate_memory(single_gpu)
    report_multi = estimate_memory(multi_gpu)
    # ZeRO stage 1 shards optimizer state across GPUs, so per-GPU optimizer
    # memory should shrink substantially with more GPUs.
    assert report_multi.optimizer_gb < report_single.optimizer_gb


def test_estimate_memory_rejects_unresolved_model_config():
    model = ModelConfig()  # architecture fields all None
    config = ATSConfig(
        model=model,
        training=TrainingConfig(max_steps=100, learning_rate=1e-4, warmup_steps=10),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=2048),
        parallelism=ParallelismConfig(),
    )
    with pytest.raises(ValueError):
        estimate_memory(config)


def test_estimate_memory_report_has_suggested_fields():
    report = estimate_memory(_config())
    assert isinstance(report.suggested_batch_size, int)
    assert report.suggested_batch_size >= 1
    assert isinstance(report.suggested_grad_accum, int)
    assert report.suggested_grad_accum >= 1
    assert 0 <= report.suggested_zero_stage <= 3


def test_estimate_memory_checkpointing_reduction_is_constant_not_depth_scaled():
    """Regression test for a formula bug: gradient checkpointing's memory
    reduction factor must be roughly constant regardless of num_layers, not
    scale with depth (an earlier version divided by sqrt(num_layers), and a
    since-reverted fix divided by num_layers directly -- both produce
    reduction factors that grow implausibly with depth; the correct
    heuristic here is a small constant factor)."""
    shallow = estimate_memory(_config(gradient_checkpointing=True))
    shallow_no_ckpt = estimate_memory(_config(gradient_checkpointing=False))
    deep = estimate_memory(_config(num_layers=64, gradient_checkpointing=True))
    deep_no_ckpt = estimate_memory(_config(num_layers=64, gradient_checkpointing=False))

    shallow_reduction = shallow_no_ckpt.activation_gb / shallow.activation_gb
    deep_reduction = deep_no_ckpt.activation_gb / deep.activation_gb

    assert shallow_reduction == pytest.approx(deep_reduction, rel=0.01)
    assert 2.0 <= shallow_reduction <= 4.0
