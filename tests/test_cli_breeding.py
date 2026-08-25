"""Tests for the ats.cli.train --init-weights flag's argument-level
behavior, and ats.cli.breed's argument parsing / perturb-spec resolution.

These deliberately stop short of running a real training job (that needs
DeepSpeed + real tokenized data; see tests/test_pbt.py and
tests/test_training.py for what's tested in isolation instead) and instead
test exactly what's reachable without one: argparse wiring, the
mutual-exclusivity guard (which fires before any config/model/DeepSpeed
work happens), and breed.py's own pure logic.
"""

from __future__ import annotations

import pytest

from ats.cli.breed import _resolve_perturb_specs
from ats.cli.breed import build_arg_parser as build_breed_parser
from ats.cli.train import build_arg_parser as build_train_parser
from ats.cli.train import main as train_main


def test_train_parser_accepts_init_weights_flag():
    parser = build_train_parser()
    args = parser.parse_args(
        ["--config", "configs/debug.yaml", "--init-weights", "/some/checkpoint"]
    )
    assert args.init_weights == "/some/checkpoint"
    assert args.resume is None


def test_train_parser_init_weights_defaults_to_none():
    parser = build_train_parser()
    args = parser.parse_args(["--config", "configs/debug.yaml"])
    assert args.init_weights is None


def test_train_main_rejects_resume_and_init_weights_together(tmp_path, caplog):
    # This must fail BEFORE ever touching --config (which doesn't even need
    # to exist for this test), since the mutual-exclusivity check in main()
    # runs first specifically so it never reaches DeepSpeed/model
    # construction.
    exit_code = train_main(
        [
            "--config",
            str(tmp_path / "does_not_exist.yaml"),
            "--resume",
            str(tmp_path / "ckpt"),
            "--init-weights",
            str(tmp_path / "weights.safetensors"),
        ]
    )
    assert exit_code == 1
    assert "mutually exclusive" in caplog.text


def test_train_main_with_only_init_weights_proceeds_past_the_guard(tmp_path, caplog):
    # No --resume given, so the mutual-exclusivity guard passes through;
    # the run should fail later (missing config file), not on the guard --
    # this pins down that the guard doesn't false-positive on a normal
    # --init-weights-only invocation.
    exit_code = train_main(
        [
            "--config",
            str(tmp_path / "does_not_exist.yaml"),
            "--init-weights",
            str(tmp_path / "weights.safetensors"),
        ]
    )
    assert exit_code == 1
    assert "mutually exclusive" not in caplog.text


# --------------------------------------------------------------------------
# ats.cli.breed
# --------------------------------------------------------------------------


def test_breed_parser_defaults():
    parser = build_breed_parser()
    args = parser.parse_args(["--config", "configs/debug.yaml"])
    assert args.population_size == 10
    assert args.generations == 10
    assert args.steps_per_generation == 100
    assert args.cull_fraction == 0.5
    assert args.no_perturb is False


def test_breed_parser_accepts_overrides():
    parser = build_breed_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/debug.yaml",
            "--population-size",
            "6",
            "--generations",
            "3",
            "--steps-per-generation",
            "50",
            "--cull-fraction",
            "0.3",
            "--output-dir",
            "/tmp/my_pbt_run",
            "--seed",
            "7",
        ]
    )
    assert args.population_size == 6
    assert args.generations == 3
    assert args.steps_per_generation == 50
    assert args.cull_fraction == pytest.approx(0.3)
    assert args.output_dir == "/tmp/my_pbt_run"
    assert args.seed == 7


def test_resolve_perturb_specs_defaults_to_three_specs():
    parser = build_breed_parser()
    args = parser.parse_args(["--config", "configs/debug.yaml"])
    specs = _resolve_perturb_specs(args)
    names = {s.name for s in specs}
    assert names == {"training.learning_rate", "training.weight_decay", "model.dropout"}


def test_resolve_perturb_specs_no_perturb_disables_everything():
    parser = build_breed_parser()
    args = parser.parse_args(["--config", "configs/debug.yaml", "--no-perturb"])
    assert _resolve_perturb_specs(args) == []


def test_resolve_perturb_specs_can_disable_individual_fields():
    parser = build_breed_parser()
    args = parser.parse_args(["--config", "configs/debug.yaml", "--no-perturb-dropout"])
    specs = _resolve_perturb_specs(args)
    names = {s.name for s in specs}
    assert names == {"training.learning_rate", "training.weight_decay"}
