"""Tests for ats.config: validation, auto-tuning, YAML loading, error messages."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ats.config.defaults import MODEL_SIZE_PRESETS, apply_size_preset
from ats.config.loader import load_config
from ats.config.schema import (
    ATSConfig, ConfigError, DataConfig, DataSource, ModelConfig, TrainingConfig,
)


def test_model_config_requires_positive_dropout():
    with pytest.raises(ValidationError):
        ModelConfig(
            hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
            intermediate_size=128, dropout=1.5,
        )


def test_model_config_heads_must_divide_hidden_size():
    with pytest.raises(ValidationError):
        ModelConfig(hidden_size=65, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=128)


def test_model_config_kv_heads_must_divide_heads():
    with pytest.raises(ValidationError):
        ModelConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=3, intermediate_size=128)


def test_training_config_warmup_cannot_exceed_max_steps():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=10, learning_rate=1e-4, warmup_steps=20)


def test_apply_size_preset_fills_unset_fields():
    cfg = ModelConfig(size="1b")
    resolved = apply_size_preset(cfg)
    assert resolved.hidden_size == MODEL_SIZE_PRESETS["1b"]["hidden_size"]
    assert resolved.num_layers == MODEL_SIZE_PRESETS["1b"]["num_layers"]
    assert resolved.is_resolved()


def test_apply_size_preset_explicit_field_wins():
    cfg = ModelConfig(size="1b", hidden_size=99999)
    resolved = apply_size_preset(cfg)
    assert resolved.hidden_size == 99999
    assert resolved.num_layers == MODEL_SIZE_PRESETS["1b"]["num_layers"]


def test_apply_size_preset_unknown_size_raises():
    cfg = ModelConfig(size="999b")
    with pytest.raises(ConfigError):
        apply_size_preset(cfg)


def test_apply_size_preset_missing_fields_without_size_raises():
    cfg = ModelConfig(hidden_size=64)  # num_layers etc still None
    with pytest.raises(ConfigError):
        apply_size_preset(cfg)


def test_load_config_debug_yaml():
    config = load_config("configs/debug.yaml")
    assert isinstance(config, ATSConfig)
    assert config.model.hidden_size == 128
    assert config.model.num_layers == 4
    assert config.model.use_swa is False
    assert config.model.use_mla is False


def test_load_config_7b_yaml_resolves_via_preset():
    config = load_config("configs/7b.yaml")
    assert config.model.hidden_size == 4096
    assert config.model.num_layers == 32


def test_load_config_missing_file_raises_actionable_error():
    with pytest.raises(ConfigError, match="not found"):
        load_config("configs/does_not_exist.yaml")


def test_data_config_requires_at_least_one_source():
    with pytest.raises(ValidationError):
        DataConfig(sources=[], seq_length=128)


def test_moe_num_experts_must_be_at_least_top_k():
    with pytest.raises(ValidationError):
        ATSConfig(
            model=ModelConfig(
                hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
                intermediate_size=128, use_moe=True, num_experts=1, moe_top_k=2,
            ),
            training=TrainingConfig(max_steps=10, learning_rate=1e-4, warmup_steps=1),
            data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=16),
        )


def test_config_hash_changes_when_architecture_changes():
    base = ModelConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, intermediate_size=128)
    common = dict(
        training=TrainingConfig(max_steps=10, learning_rate=1e-4, warmup_steps=1),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=16),
    )
    cfg_a = ATSConfig(model=base, **common)
    cfg_b = ATSConfig(model=base.model_copy(update={"hidden_size": 128, "num_heads": 8}), **common)
    assert cfg_a.config_hash() != cfg_b.config_hash()


# --- CLI override merge logic (ats.cli.train) ---

def _cli_config(argv):
    """Helper: parse argv (excluding --config) against the debug config and
    return the merged ATSConfig, using the real ats.cli.train module."""
    from ats.cli import train as train_module

    parser = train_module.build_arg_parser()
    args = parser.parse_args(["--config", "configs/debug.yaml"] + argv)
    base_config = load_config(args.config)
    return train_module.apply_cli_overrides(base_config, args)


def test_cli_use_swa_flag_sets_config():
    config = _cli_config(["--use-swa"])
    assert config.model.use_swa is True


def test_cli_no_use_swa_overrides_preset():
    config = _cli_config(["--architecture", "all", "--no-use-swa"])
    assert config.model.use_swa is False
    # Everything else from "all" should still be enabled.
    assert config.model.use_mla is True
    assert config.model.use_moe is True


def test_cli_architecture_all_enables_everything_compatible():
    config = _cli_config(["--architecture", "all"])
    assert config.model.use_swa is True
    assert config.model.use_mla is True
    assert config.model.use_mamba is True
    assert config.model.use_moe is True
    assert config.model.use_mod is True
    assert config.model.use_mtp is True


def test_cli_architecture_dense_disables_everything():
    config = _cli_config(["--architecture", "dense"])
    assert config.model.use_swa is False
    assert config.model.use_mla is False
    assert config.model.use_mamba is False
    assert config.model.use_moe is False
    assert config.model.use_mod is False
    assert config.model.use_mtp is False


def test_cli_mtp_plus_diffusion_raises():
    with pytest.raises(ConfigError):
        _cli_config(["--use-mtp", "--model-type", "diffusion"])


def test_cli_numeric_override_applies():
    config = _cli_config(["--learning-rate", "5e-5", "--max-steps", "42"])
    assert config.training.learning_rate == pytest.approx(5e-5)
    assert config.training.max_steps == 42


def test_cli_hidden_size_override_reruns_validation():
    # num_kv_heads=4 from debug.yaml; overriding num_heads to something not
    # divisible by it must still be caught by re-validation after merge.
    with pytest.raises(ConfigError):
        _cli_config(["--num-heads", "5"])


def test_cli_moe_flags_apply_numeric_overrides():
    config = _cli_config(["--use-moe", "--moe-num-experts", "16", "--moe-top-k", "4"])
    assert config.model.use_moe is True
    assert config.model.num_experts == 16
    assert config.model.moe_top_k == 4


# --- micro_batch_size: config field + CLI precedence regression tests ---

def test_training_config_micro_batch_size_default_is_one():
    training = TrainingConfig(max_steps=10, learning_rate=1e-4, warmup_steps=1)
    assert training.micro_batch_size == 1


def test_training_config_rejects_non_positive_micro_batch_size():
    with pytest.raises(ValidationError):
        TrainingConfig(max_steps=10, learning_rate=1e-4, warmup_steps=1, micro_batch_size=0)


def test_load_config_debug_yaml_micro_batch_size_is_read_from_file():
    config = load_config("configs/debug.yaml")
    # debug.yaml explicitly sets micro_batch_size: 4 -- this must NOT be
    # silently overwritten by any CLI-layer default.
    assert config.training.micro_batch_size == 4


def test_cli_without_micro_batch_size_flag_preserves_yaml_value():
    """Regression test: --micro-batch-size must default to None at the
    argparse layer so apply_cli_overrides leaves the YAML's
    training.micro_batch_size untouched when the flag isn't passed."""
    config = _cli_config([])
    assert config.model.hidden_size == 128  # sanity: still the debug config
    assert config.training.micro_batch_size == 4  # from configs/debug.yaml, not clobbered to 1


def test_cli_micro_batch_size_flag_overrides_yaml():
    config = _cli_config(["--micro-batch-size", "16"])
    assert config.training.micro_batch_size == 16


def test_load_config_350m_yaml_resolves_via_preset():
    """Regression test: configs/350m.yaml previously referenced a model.size
    preset ('350m') that did not exist in MODEL_SIZE_PRESETS, so loading it
    raised ConfigError immediately. Now covered directly."""
    config = load_config("configs/350m.yaml")
    assert config.model.hidden_size == 1024
    assert config.model.num_layers == 24


def test_all_size_configs_load_without_error():
    """Every configs/*.yaml file must actually be loadable -- this would
    have caught the missing-350m-preset bug immediately for ALL size
    configs, not just the ones with an explicit test."""
    import pathlib

    for path in sorted(pathlib.Path("configs").glob("*.yaml")):
        config = load_config(str(path))
        assert config.model.is_resolved(), f"{path}: model config did not resolve"
