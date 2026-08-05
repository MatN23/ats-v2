"""Checkpoint save/resume.

Delegates model + optimizer state to DeepSpeed's own save_checkpoint /
load_checkpoint (client_state is used for everything DeepSpeed doesn't own:
global_step, RNG states, and a hash of the resolved config so a resume with a
mismatched config fails loudly instead of silently producing garbage).
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from ats.config.schema import ATSConfig, ConfigError
from ats.utils.logging_utils import get_logger

logger = get_logger("ats.training.checkpoint")

_TRAINING_STATE_FILENAME = "training_state.json"
_SAFETENSORS_FILENAME = "model.safetensors"


class TrainingHaltError(RuntimeError):
    """Raised when the adaptive controller forces training to stop."""


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [t.tolist() for t in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(tuple(state["python"]) if isinstance(state["python"], list) else state["python"])
    np_state = state["numpy"]
    if isinstance(np_state, list):
        np_state = tuple(np_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(
            [torch.tensor(t, dtype=torch.uint8) for t in state["torch_cuda"]]
        )


def load_model_weights_safetensors(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Loads just the model weights from a checkpoint's model.safetensors
    file, with no DeepSpeed engine and no pickle execution risk. Useful for
    export/inspection tooling that only needs weights, not optimizer state
    or the ability to resume training."""
    path = Path(checkpoint_dir) / _SAFETENSORS_FILENAME
    if not path.exists():
        raise ConfigError(
            f"No {_SAFETENSORS_FILENAME} found at {path}. "
            f"Fix: point checkpoint_dir at a directory created by "
            f"CheckpointManager.save (e.g. checkpoints/run/step_5000)."
        )
    return safetensors_load_file(str(path))


class CheckpointManager:
    def __init__(self, config: ATSConfig) -> None:
        self.config = config
        self.output_dir = Path(config.checkpoint.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _tag(self, global_step: int) -> str:
        return f"step_{global_step}"

    def save(self, model_engine: Any, global_step: int, epoch: int) -> Path:
        tag = self._tag(global_step)
        ckpt_dir = self.output_dir / tag

        client_state = {
            "global_step": global_step,
            "epoch": epoch,
            "config_hash": self.config.config_hash(),
            "rng_state": _capture_rng_state(),
        }
        model_engine.save_checkpoint(
            str(self.output_dir), tag=tag,
            client_state=client_state, save_latest=True,
        )

        state_path = ckpt_dir / _TRAINING_STATE_FILENAME
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(
                {"global_step": global_step, "epoch": epoch, "config_hash": self.config.config_hash()},
                f, indent=2,
            )

        # Write a copy of the resolved config alongside the checkpoint so
        # export.py (and manual inspection) can find it without requiring
        # --config to be passed explicitly every time.
        config_path = ckpt_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.model_dump(), f, sort_keys=False)

        # Write a plain safetensors snapshot of the model weights alongside
        # DeepSpeed's own checkpoint. DeepSpeed's save_checkpoint() above
        # already handles full ZeRO-sharded optimizer/model resume (that's
        # DeepSpeed's own internal format, not something ats controls or
        # should reimplement) -- this safetensors copy is an ADDITIONAL,
        # de-sharded, pickle-free artifact for fast loading / HF export /
        # manual inspection, matching the .safetensors benefits (fast I/O,
        # no arbitrary-code-execution risk on load, HF-standard format).
        module = model_engine.module if hasattr(model_engine, "module") else model_engine
        state_dict = {k: v.detach().cpu().contiguous() for k, v in module.state_dict().items()}
        safetensors_save_file(state_dict, str(ckpt_dir / _SAFETENSORS_FILENAME))

        logger.info("Saved checkpoint at step %d to %s", global_step, ckpt_dir)
        self._prune_old_checkpoints()
        return ckpt_dir

    def load(self, model_engine: Any, checkpoint_dir: str) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise ConfigError(
                f"--resume path does not exist: {checkpoint_path}. "
                f"Fix: point --resume at a directory created by CheckpointManager.save "
                f"(e.g. checkpoints/run/step_5000)."
            )
        tag = checkpoint_path.name
        load_dir = checkpoint_path.parent

        _, client_state = model_engine.load_checkpoint(str(load_dir), tag=tag)
        if client_state is None:
            raise ConfigError(
                f"Checkpoint at {checkpoint_path} has no client_state (global_step, "
                f"RNG state, config_hash). It may not have been saved by "
                f"CheckpointManager.save. Fix: resume from a valid ats-v2 checkpoint."
            )

        saved_hash = client_state.get("config_hash")
        current_hash = self.config.config_hash()
        if saved_hash != current_hash:
            raise ConfigError(
                f"Config mismatch on resume: checkpoint was saved with config_hash "
                f"'{saved_hash}' but the current config hashes to '{current_hash}'. "
                f"Fix: resume with the exact same config file used for the original run, "
                f"or start a fresh run if the architecture change is intentional."
            )

        _restore_rng_state(client_state["rng_state"])
        logger.info("Resumed from %s at step %d", checkpoint_path, client_state["global_step"])
        return client_state

    def _prune_old_checkpoints(self) -> None:
        keep_n = self.config.training.keep_last_n_checkpoints
        step_dirs = sorted(
            (p for p in self.output_dir.glob("step_*") if p.is_dir()),
            key=lambda p: int(p.name.split("_")[1]),
        )
        excess = len(step_dirs) - keep_n
        for old_dir in step_dirs[:max(0, excess)]:
            logger.info("Pruning old checkpoint %s", old_dir)
            shutil.rmtree(old_dir, ignore_errors=False)
