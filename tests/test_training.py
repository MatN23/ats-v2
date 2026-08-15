"""Tests for ats.training: LR scheduler shape, AdaptiveController spike/plateau/
halt behavior, and a checkpoint save/corrupt/load roundtrip using plain torch
(no DeepSpeed dependency) to verify CheckpointManager's RNG + hash logic."""

from __future__ import annotations

import pytest
import torch

from ats.config.schema import (
    AdaptiveConfig,
    ATSConfig,
    CheckpointConfig,
    DataConfig,
    DataSource,
    ModelConfig,
    TrainingConfig,
)
from ats.training.adaptive_controller import AdaptiveController, TrainingMetrics
from ats.training.checkpoint import CheckpointManager
from ats.training.scheduler import WarmupCosineScheduler


def test_scheduler_warmup_is_linear():
    sched = WarmupCosineScheduler(
        base_lr=1.0, warmup_steps=10, max_steps=100, min_lr_ratio=0.1
    )
    assert sched.get_lr(0) == pytest.approx(1.0 / 10)
    assert sched.get_lr(9) == pytest.approx(1.0)
    assert sched.get_lr(0) < sched.get_lr(5) < sched.get_lr(9)


def test_scheduler_decays_to_min_lr_at_end():
    sched = WarmupCosineScheduler(
        base_lr=1.0, warmup_steps=10, max_steps=100, min_lr_ratio=0.1
    )
    assert sched.get_lr(99) == pytest.approx(0.1, abs=0.02)
    assert sched.get_lr(200) == pytest.approx(0.1)


def test_scheduler_rejects_warmup_greater_than_max_steps():
    with pytest.raises(ValueError):
        WarmupCosineScheduler(base_lr=1.0, warmup_steps=200, max_steps=100)


def _metrics(step, loss, grad_norm=1.0):
    return TrainingMetrics(
        step=step, loss=loss, grad_norm=grad_norm, learning_rate=1e-4
    )


def test_adaptive_controller_detects_loss_spike():
    config = AdaptiveConfig(spike_window=5, plateau_window=1000, history_size=2000)
    controller = AdaptiveController(config)
    action = None
    for step in range(10):
        action = controller.step(_metrics(step, loss=1.0))
    assert action is None  # stable loss, nothing fires

    # Now inject a spike: recent losses much higher than the prior window.
    # The controller correctly fires loss_spike_lr_cut on the FIRST step
    # where the spike becomes detectable, then its cooldown (50 steps)
    # correctly suppresses re-firing on every subsequent step in this loop
    # (by design -- see AdaptiveController._make_action's docstring on
    # preventing oscillation). So the action to check is whichever one
    # fired first, not just whatever the last iteration happened to return
    # (which is None here, since it's inside the cooldown window).
    action = None
    for step in range(10, 15):
        result = controller.step(_metrics(step, loss=10.0))
        if action is None:
            action = result
    assert action is not None
    assert action.type == "loss_spike_lr_cut"
    assert action.apply is True


def test_adaptive_controller_detects_plateau():
    config = AdaptiveConfig(
        plateau_window=10, plateau_rel_std=0.01, spike_window=1000, history_size=2000
    )
    controller = AdaptiveController(config)
    action = None
    for step in range(10):
        action = controller.step(_metrics(step, loss=1.0 + (step % 2) * 1e-6))
    assert action is not None
    assert action.type == "plateau_lr_boost"


def test_adaptive_controller_halts_after_three_emergency_cuts():
    config = AdaptiveConfig(
        grad_norm_threshold=5.0,
        spike_window=1000,
        plateau_window=1000,
        history_size=2000,
    )
    controller = AdaptiveController(config)
    halted = False
    step = 0
    for _ in range(4):
        action = controller.step(_metrics(step, loss=1.0, grad_norm=100.0))
        if action is not None and action.type == "training_halt":
            halted = True
            break
        step += 101  # exceed the 100-step cooldown so each call actually fires
    assert halted


def test_adaptive_controller_respects_cooldown():
    config = AdaptiveConfig(
        grad_norm_threshold=5.0,
        spike_window=1000,
        plateau_window=1000,
        history_size=2000,
    )
    controller = AdaptiveController(config)
    first = controller.step(_metrics(0, loss=1.0, grad_norm=100.0))
    second = controller.step(_metrics(1, loss=1.0, grad_norm=100.0))
    assert first is not None
    assert second is None  # within cooldown window


def test_adaptive_controller_disabled_returns_none():
    config = AdaptiveConfig(enabled=False)
    controller = AdaptiveController(config)
    assert controller.step(_metrics(0, loss=1.0, grad_norm=999.0)) is None


class _TinyModelEngine:
    """Minimal stand-in for a DeepSpeed model engine's checkpoint interface,
    used to test CheckpointManager's client_state / RNG-state / hash logic
    without requiring deepspeed to be installed."""

    def __init__(self, model: torch.nn.Module):
        self.module = model

    def save_checkpoint(self, save_dir, tag, client_state, save_latest=True):
        import json
        import os

        ckpt_dir = os.path.join(save_dir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.module.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        with open(os.path.join(ckpt_dir, "client_state.json"), "w") as f:
            json.dump(
                {
                    "global_step": client_state["global_step"],
                    "epoch": client_state["epoch"],
                    "config_hash": client_state["config_hash"],
                },
                f,
            )
        torch.save(client_state["rng_state"], os.path.join(ckpt_dir, "rng_state.pt"))

    def load_checkpoint(self, load_dir, tag):
        import json
        import os

        ckpt_dir = os.path.join(load_dir, tag)
        # weights_only=False: this checkpoint's client_state (specifically
        # the numpy RNG state tuple CheckpointManager stores) isn't
        # loadable under PyTorch 2.6+'s weights_only=True default (numpy's
        # pickled reconstruction globals aren't on torch's default safe
        # list). Real DeepSpeed's own TorchCheckpointEngine.load already
        # passes weights_only=False explicitly for the same reason -- this
        # fake engine needs to match that to be a faithful stand-in.
        self.module.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "model.pt"), weights_only=False)
        )
        with open(os.path.join(ckpt_dir, "client_state.json")) as f:
            client_state = json.load(f)
        client_state["rng_state"] = torch.load(
            os.path.join(ckpt_dir, "rng_state.pt"), weights_only=False
        )
        return self.module, client_state


def _make_config(tmp_path):
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
        checkpoint=CheckpointConfig(output_dir=str(tmp_path / "ckpts")),
    )


def test_checkpoint_save_corrupt_load_roundtrip(tmp_path):
    config = _make_config(tmp_path)
    model = torch.nn.Linear(8, 8)
    engine = _TinyModelEngine(model)
    manager = CheckpointManager(config)

    original_weight = model.weight.detach().clone()
    ckpt_dir = manager.save(engine, global_step=5, epoch=0)

    # Corrupt the in-memory weights.
    with torch.no_grad():
        model.weight.add_(999.0)
    assert not torch.allclose(model.weight, original_weight)

    client_state = manager.load(engine, str(ckpt_dir))

    assert torch.allclose(model.weight, original_weight)
    assert client_state["global_step"] == 5


def test_checkpoint_load_rejects_mismatched_config(tmp_path):
    config = _make_config(tmp_path)
    model = torch.nn.Linear(8, 8)
    engine = _TinyModelEngine(model)
    manager = CheckpointManager(config)
    ckpt_dir = manager.save(engine, global_step=1, epoch=0)

    other_model_config = ModelConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=32,
        vocab_size=20,
    )
    other_config = config.model_copy(update={"model": other_model_config})
    other_manager = CheckpointManager(other_config)

    from ats.config.schema import ConfigError

    with pytest.raises(ConfigError):
        other_manager.load(engine, str(ckpt_dir))


def test_checkpoint_save_writes_config_yaml(tmp_path):
    """Regression test: export.py's auto-discovery depends on
    CheckpointManager.save() actually writing config.yaml into the
    checkpoint's tag directory. Previously nothing ever wrote this file, so
    the auto-discovery path always failed even when export.py looked in the
    'right' place."""
    import yaml

    config = _make_config(tmp_path)
    model = torch.nn.Linear(8, 8)
    engine = _TinyModelEngine(model)
    manager = CheckpointManager(config)

    ckpt_dir = manager.save(engine, global_step=3, epoch=0)
    config_path = ckpt_dir / "config.yaml"
    assert config_path.exists()

    with open(config_path) as f:
        loaded = yaml.safe_load(f)
    assert loaded["model"]["hidden_size"] == config.model.hidden_size
    assert loaded["training"]["max_steps"] == config.training.max_steps


def test_checkpoint_save_writes_safetensors(tmp_path):
    """Regression/spec test: model weights must be saved as .safetensors
    (fast, pickle-free, HF-standard), not solely inside DeepSpeed's own
    pickle-based checkpoint format."""
    from ats.training.checkpoint import load_model_weights_safetensors

    config = _make_config(tmp_path)
    model = torch.nn.Linear(8, 8)
    engine = _TinyModelEngine(model)
    manager = CheckpointManager(config)

    ckpt_dir = manager.save(engine, global_step=7, epoch=0)
    safetensors_path = ckpt_dir / "model.safetensors"
    assert safetensors_path.exists()

    loaded_weights = load_model_weights_safetensors(str(ckpt_dir))
    assert torch.allclose(loaded_weights["weight"], model.weight.detach())
    assert torch.allclose(loaded_weights["bias"], model.bias.detach())


def test_checkpoint_save_only_rank_zero_writes_files(tmp_path, monkeypatch):
    """Regression test for a checkpoint I/O race condition: under
    distributed training (e.g. ZeRO-3, where every rank has the full
    desharded model), only rank 0 should write model.safetensors,
    config.yaml, and training_state.json -- otherwise every rank
    redundantly writes the same (potentially huge) file simultaneously."""
    monkeypatch.setenv("RANK", "1")  # simulate a non-zero rank

    config = _make_config(tmp_path)
    model = torch.nn.Linear(8, 8)
    engine = _TinyModelEngine(model)
    manager = CheckpointManager(config)

    ckpt_dir = manager.save(engine, global_step=1, epoch=0)

    # _TinyModelEngine.save_checkpoint (DeepSpeed's own path, left untouched)
    # still writes model.pt/client_state.json/rng_state.pt regardless of
    # rank in this test double -- but the ADDITIONAL rank-0-only artifacts
    # ats.training.checkpoint itself is responsible for must NOT appear.
    assert not (ckpt_dir / "model.safetensors").exists()
    assert not (ckpt_dir / "config.yaml").exists()
    assert not (ckpt_dir / "training_state.json").exists()


def test_checkpoint_save_rank_zero_writes_files(tmp_path, monkeypatch):
    """Control test: rank 0 (the default / common single-process case)
    must still write all three files, proving the rank guard doesn't just
    always skip writing."""
    monkeypatch.setenv("RANK", "0")

    config = _make_config(tmp_path)
    model = torch.nn.Linear(8, 8)
    engine = _TinyModelEngine(model)
    manager = CheckpointManager(config)

    ckpt_dir = manager.save(engine, global_step=1, epoch=0)

    assert (ckpt_dir / "model.safetensors").exists()
    assert (ckpt_dir / "config.yaml").exists()
    assert (ckpt_dir / "training_state.json").exists()
