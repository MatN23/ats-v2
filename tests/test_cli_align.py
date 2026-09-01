"""Tests for ats.cli.align's argument parsing and the error paths reachable
without a real checkpoint/DeepSpeed/GPU: --method rlhf's explicit rejection,
--checkpoint required, --preference-data required, and
model_type='diffusion' being rejected before any weights are loaded.

Mirrors tests/test_cli_breeding.py's approach: stop short of a real
training job (needs DeepSpeed + real tokenized data), test exactly what's
reachable without one.
"""

from __future__ import annotations

from ats.cli.align import _check_architecture_supported, build_arg_parser, main
from ats.config.schema import ConfigError, ModelConfig


def test_align_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/debug.yaml",
            "--checkpoint",
            "/some/checkpoint",
            "--preference-data",
            "/some/prefs.jsonl",
        ]
    )
    assert args.method == "dpo"
    assert args.dpo_beta == 0.1
    assert args.output_dir is None


def test_align_parser_base_checkpoint_is_alias_for_checkpoint():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/debug.yaml",
            "--base-checkpoint",
            "/some/checkpoint",
            "--preference-data",
            "/some/prefs.jsonl",
        ]
    )
    assert args.checkpoint == "/some/checkpoint"


def test_align_main_rejects_rlhf_method():
    exit_code = main(
        [
            "--config",
            "configs/debug.yaml",
            "--checkpoint",
            "/does/not/matter",
            "--preference-data",
            "/does/not/matter.jsonl",
            "--method",
            "rlhf",
        ]
    )
    assert exit_code == 1


def test_align_main_rejects_rlhf_before_touching_config():
    # The rlhf rejection must fire even when --config points nowhere real,
    # proving it happens before any config loading -- same "fail fast, in
    # order" property test_cli_breeding.py checks for ats-train.
    exit_code = main(
        [
            "--config",
            "/definitely/does/not/exist.yaml",
            "--checkpoint",
            "/does/not/matter",
            "--preference-data",
            "/does/not/matter.jsonl",
            "--method",
            "rlhf",
        ]
    )
    assert exit_code == 1


def test_align_main_requires_checkpoint():
    exit_code = main(
        [
            "--config",
            "configs/debug.yaml",
            "--preference-data",
            "/does/not/matter.jsonl",
        ]
    )
    assert exit_code == 1


def test_check_architecture_supported_rejects_diffusion():
    config_dict = {
        "model": {
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "num_kv_heads": 2,
            "intermediate_size": 32,
            "vocab_size": 30,
            "model_type": "diffusion",
        },
        "training": {"max_steps": 10, "learning_rate": 1e-3, "warmup_steps": 1},
        "data": {"sources": [{"path": "x.jsonl"}], "seq_length": 8},
    }
    from ats.config.schema import ATSConfig

    config = ATSConfig.model_validate(config_dict)
    try:
        _check_architecture_supported(config)
        raised = False
    except ConfigError:
        raised = True
    assert raised


def test_check_architecture_supported_allows_dense():
    from ats.config.schema import (
        ATSConfig,
        DataConfig,
        DataSource,
        TrainingConfig,
    )

    config = ATSConfig(
        model=ModelConfig(
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            intermediate_size=32,
            vocab_size=30,
        ),
        training=TrainingConfig(max_steps=10, learning_rate=1e-3, warmup_steps=1),
        data=DataConfig(sources=[DataSource(path="x.jsonl")], seq_length=8),
    )
    _check_architecture_supported(config)  # must not raise
