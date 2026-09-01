"""Tests for parallelism.offload_optimizer/offload_param -> DeepSpeed's
zero_optimization JSON block, and the ZeRO-stage compatibility checks that
gate them.

build_deepspeed_config never calls deepspeed.initialize() or requires
deepspeed to be installed -- it only builds the config dict -- so these
tests run on plain CPU with no GPU/deepspeed dependency, exactly like the
rest of the config-construction tests in this suite.
"""

from __future__ import annotations

import pytest

from ats.config.schema import (
    ATSConfig,
    ConfigError,
    DataConfig,
    DataSource,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)
from ats.parallelism.deepspeed_utils import build_deepspeed_config


def _config(**parallelism_overrides) -> ATSConfig:
    model_config = ModelConfig(
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        vocab_size=20,
    )
    return ATSConfig(
        model=model_config,
        training=TrainingConfig(max_steps=10, learning_rate=1e-3, warmup_steps=1),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=8),
        parallelism=ParallelismConfig(**parallelism_overrides),
    )


def test_no_offload_by_default():
    config = _config(strategy="deepspeed_zero2", gpus=2)
    ds_config = build_deepspeed_config(config, micro_batch_size=1)
    zero_opt = ds_config["zero_optimization"]
    assert "offload_optimizer" not in zero_opt
    assert "offload_param" not in zero_opt


def test_offload_optimizer_with_zero2():
    config = _config(strategy="deepspeed_zero2", gpus=2, offload_optimizer=True)
    ds_config = build_deepspeed_config(config, micro_batch_size=1)
    zero_opt = ds_config["zero_optimization"]
    assert zero_opt["stage"] == 2
    assert zero_opt["offload_optimizer"] == {"device": "cpu", "pin_memory": True}
    assert "offload_param" not in zero_opt


def test_offload_param_with_zero3():
    config = _config(strategy="deepspeed_zero3", gpus=2, offload_param=True)
    ds_config = build_deepspeed_config(config, micro_batch_size=1)
    zero_opt = ds_config["zero_optimization"]
    assert zero_opt["stage"] == 3
    assert zero_opt["offload_param"] == {"device": "cpu", "pin_memory": True}


def test_offload_both_with_zero3():
    config = _config(
        strategy="deepspeed_zero3", gpus=2, offload_optimizer=True, offload_param=True
    )
    ds_config = build_deepspeed_config(config, micro_batch_size=1)
    zero_opt = ds_config["zero_optimization"]
    assert "offload_optimizer" in zero_opt
    assert "offload_param" in zero_opt


def test_offload_param_rejected_below_zero3():
    config = _config(strategy="deepspeed_zero2", gpus=2, offload_param=True)
    with pytest.raises(ConfigError, match="offload_param"):
        build_deepspeed_config(config, micro_batch_size=1)


def test_offload_param_rejected_with_zero1():
    config = _config(strategy="deepspeed_zero1", gpus=2, offload_param=True)
    with pytest.raises(ConfigError, match="ZeRO stage 3"):
        build_deepspeed_config(config, micro_batch_size=1)


def test_offload_optimizer_rejected_with_zero0():
    config = _config(strategy="deepspeed_zero0", gpus=1, offload_optimizer=True)
    with pytest.raises(ConfigError, match="offload_optimizer"):
        build_deepspeed_config(config, micro_batch_size=1)


def test_offload_optimizer_allowed_with_zero1():
    # Boundary check: stage 1 (not just 2/3) should be accepted for
    # offload_optimizer -- only stage 0 is rejected.
    config = _config(strategy="deepspeed_zero1", gpus=2, offload_optimizer=True)
    ds_config = build_deepspeed_config(config, micro_batch_size=1)
    assert ds_config["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
