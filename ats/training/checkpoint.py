"""Checkpoint save/resume.

Delegates model + optimizer state to DeepSpeed's own save_checkpoint /
load_checkpoint (client_state is used for everything DeepSpeed doesn't own:
global_step, RNG states, and a hash of the resolved config so a resume with a
mismatched config fails loudly instead of silently producing garbage).
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

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


def _current_rank() -> int:
    """The current process's global rank.

    BUG-121: this used to read only the RANK/LOCAL_RANK environment
    variables. Those are set by the torchrun/deepspeed launchers, but they
    are NOT the authoritative source once a process group exists -- a job
    that calls dist.init_process_group() with an explicit rank (or any
    launcher that does not export RANK) leaves every process reading "0".
    Every rank then believes it is rank 0 and they all write the same
    model.safetensors file to the same path simultaneously, which is
    exactly the corruption this function's callers guard against; save()'s
    own comment describes that hazard while relying on the weaker signal to
    detect it. torch.distributed is asked first when it is initialized, and
    the environment is used only as the pre-initialization fallback.
    """
    try:
        import torch.distributed as dist
    except ImportError:
        pass
    else:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _barrier_if_distributed() -> None:
    """Synchronizes all ranks if torch.distributed is initialized, so ranks
    other than 0 don't race ahead assuming files rank 0 just wrote already
    exist on disk. A no-op in single-process (non-distributed) runs."""
    try:
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class TrainingHaltError(RuntimeError):
    """Raised when the adaptive controller forces training to stop."""


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [t.tolist() for t in torch.cuda.get_rng_state_all()]
    return state


def _lists_to_tuples(value: Any) -> Any:
    """Recursively converts lists (and their nested lists) to tuples.
    random.getstate() returns nested tuples (version, big-tuple-of-ints,
    gauss_next); a JSON round-trip turns every level of nesting into a
    list, and random.setstate() requires the inner structure to be tuples
    too, not just the outermost container."""
    if isinstance(value, list):
        return tuple(_lists_to_tuples(v) for v in value)
    return value


def _restore_rng_state(state: dict[str, Any]) -> None:
    python_state = state["python"]
    if isinstance(python_state, list):
        python_state = _lists_to_tuples(python_state)
    random.setstate(python_state)

    np_state = state["numpy"]
    if isinstance(np_state, (list, tuple)):
        np_state = list(np_state)
        np_state[1] = np.array(np_state[1], dtype=np.uint32)
        np_state = tuple(np_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))

    cuda_states = state.get("torch_cuda")
    if not torch.cuda.is_available():
        if cuda_states:
            logger.warning(
                "Checkpoint carries CUDA RNG state for %d device(s) but no CUDA "
                "device is visible; CUDA RNG was not restored. The resumed run "
                "is not bitwise-identical to the original.",
                len(cuda_states),
            )
        return
    if not cuda_states:
        logger.warning(
            "Resuming on CUDA from a checkpoint that carries no CUDA RNG state; "
            "CUDA RNG starts fresh. The resumed run is not bitwise-identical to "
            "the original."
        )
        return
    device_count = torch.cuda.device_count()
    if len(cuda_states) != device_count:
        logger.warning(
            "Checkpoint carries CUDA RNG state for %d device(s) but %d are "
            "visible now; CUDA RNG was not restored. The resumed run is not "
            "bitwise-identical to the original. Fix: resume on the same number "
            "of GPUs the checkpoint was written on.",
            len(cuda_states),
            device_count,
        )
        return
    torch.cuda.set_rng_state_all(
        [torch.tensor(t, dtype=torch.uint8) for t in cuda_states]
    )


def load_model_weights_safetensors(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Loads just the model weights from a checkpoint's model.safetensors
    file, with no DeepSpeed engine and no pickle execution risk."""
    path = Path(checkpoint_dir) / _SAFETENSORS_FILENAME
    if not path.exists():
        raise ConfigError(
            f"No {_SAFETENSORS_FILENAME} found at {path}. "
            f"Fix: point checkpoint_dir at a directory created by "
            f"CheckpointManager.save (e.g. checkpoints/run/step_5000)."
        )
    return safetensors_load_file(str(path))


def load_initial_weights(model: Any, path: str) -> None:
    """Loads model weights only -- no optimizer state, no global_step, no RNG
    state, and no config_hash match requirement."""
    p = Path(path)
    if p.is_dir():
        weights = load_model_weights_safetensors(str(p))
        source_desc = str(p / _SAFETENSORS_FILENAME)
    elif p.suffix == ".safetensors":
        if not p.exists():
            raise ConfigError(
                f"--init-weights path does not exist: {p}. "
                f"Fix: point it at a checkpoint directory or a .safetensors file "
                f"produced by CheckpointManager.save."
            )
        weights = safetensors_load_file(str(p))
        source_desc = str(p)
    else:
        raise ConfigError(
            f"--init-weights path {p} is neither a directory containing "
            f"{_SAFETENSORS_FILENAME} nor a .safetensors file. "
            f"Fix: point it at a checkpoint directory (e.g. checkpoints/run/step_5000) "
            f"or a *.safetensors file."
        )

    try:
        missing, unexpected = model.load_state_dict(weights, strict=False)
    except RuntimeError as exc:
        raise ConfigError(
            f"--init-weights source {source_desc} does not match the destination "
            f"model's architecture (shape mismatch): {exc}. Fix: --init-weights is "
            f"only valid between members with identical model architecture "
            f"(hidden_size, num_layers, num_heads, use_moe/use_mla/etc. must all "
            f"match); only training-level hyperparameters (learning rate, weight "
            f"decay, dropout, ...) may differ between source and destination."
        ) from exc

    if missing:
        # BUG FIX: tied parameters (e.g. lm_head.weight sharing storage
        # with embed_tokens.weight when model.tie_word_embeddings=True) are
        # legitimately absent from a saved checkpoint -- see
        # CheckpointManager.save's dedup-by-storage-identity fix. The model
        # ties them together at construction time (before this function
        # ever runs), so loading the OTHER half of the tie already updates
        # the shared tensor. A "missing" key is not actually missing if the
        # model's current parameter for that name shares storage with a
        # parameter that WAS loaded.
        model_state = model.state_dict()
        loaded_ptrs = {
            model_state[k].data_ptr() for k in weights if k in model_state
        }
        missing = [
            name
            for name in missing
            if name not in model_state
            or model_state[name].data_ptr() not in loaded_ptrs
        ]

    if missing or unexpected:
        raise ConfigError(
            f"--init-weights source {source_desc} does not match the destination "
            f"model's architecture: {len(missing)} missing key(s), "
            f"{len(unexpected)} unexpected key(s). Fix: --init-weights is only valid "
            f"between members with identical model architecture (hidden_size, "
            f"num_layers, num_heads, use_moe/use_mla/etc. must all match); only "
            f"training-level hyperparameters (learning rate, weight decay, dropout, "
            f"...) may differ between source and destination."
        )
    logger.info("Loaded initial weights from %s", source_desc)


class CheckpointManager:
    def __init__(self, config: ATSConfig) -> None:
        self.config = config
        self.output_dir = Path(config.checkpoint.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _tag(self, global_step: int) -> str:
        return f"step_{global_step}"

    def save(
        self,
        model_engine: Any,
        global_step: int,
        epoch: int,
        extra_client_state: dict[str, Any] | None = None,
    ) -> Path:
        tag = self._tag(global_step)
        ckpt_dir = self.output_dir / tag
        rank = _current_rank()

        client_state = {
            "global_step": global_step,
            "epoch": epoch,
            "config_hash": self.config.config_hash(),
            "rng_state": _capture_rng_state(),
        }
        if extra_client_state:
            client_state.update(extra_client_state)
        model_engine.save_checkpoint(
            str(self.output_dir),
            tag=tag,
            client_state=client_state,
            save_latest=True,
        )

        # BUG FIX: plain module.state_dict() under ZeRO-3 does NOT gather
        # sharded/offloaded parameters -- it returns each rank's local
        # partition, which for a single-rank offloaded run comes back as
        # empty (shape [0]) placeholder tensors. save_16bit_model() is
        # DeepSpeed's purpose-built method that performs the actual
        # collective gather internally before returning a full state dict.
        # It must still be called on every rank (the gather is collective),
        # but only returns a non-empty dict on rank 0.
        tmp_dir = ckpt_dir / "_tmp_16bit"
        success = model_engine.save_16bit_model(str(tmp_dir), "model.pt")

        if rank == 0:
            if not success:
                raise RuntimeError(
                    "model_engine.save_16bit_model() failed to produce a "
                    "consolidated state dict on rank 0."
                )
            full_state_dict = torch.load(tmp_dir / "model.pt", map_location="cpu")

            # BUG FIX: tied weights (lm_head.weight sharing storage with
            # embed_tokens.weight when model.tie_word_embeddings=True) are
            # the same underlying tensor under two different keys.
            # safetensors_save_file() refuses to save two keys aliasing the
            # same storage. Deduplicate by storage identity, keeping only
            # the first key seen per unique tensor -- ATSTransformer.__init__
            # already re-ties lm_head.weight = embed_tokens.weight at
            # construction time (see ats/model/transformer.py), so loading
            # only the kept key is sufficient; nothing is lost.
            seen_data_ptrs: dict[int, str] = {}
            state_dict: dict[str, torch.Tensor] = {}
            for key, tensor in full_state_dict.items():
                ptr = tensor.data_ptr()
                if ptr in seen_data_ptrs:
                    continue
                seen_data_ptrs[ptr] = key
                state_dict[key] = tensor.detach().cpu().contiguous()
            del full_state_dict
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if rank == 0:
            state_path = ckpt_dir / _TRAINING_STATE_FILENAME
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "config_hash": self.config.config_hash(),
                    },
                    f,
                    indent=2,
                )

            config_path = ckpt_dir / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config.model_dump(), f, sort_keys=False)

            safetensors_save_file(state_dict, str(ckpt_dir / _SAFETENSORS_FILENAME))

            logger.info("Saved checkpoint at step %d to %s", global_step, ckpt_dir)
            self._prune_old_checkpoints()

        _barrier_if_distributed()
        return ckpt_dir

    def load(self, model_engine: Any, checkpoint_dir: str) -> dict[str, Any]:
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

        if "rng_state" not in client_state:
            raise ConfigError(
                f"Checkpoint at {checkpoint_path} has a client_state but no "
                f"'rng_state' entry, so RNG cannot be restored and the resumed "
                f"run would silently diverge from the original trajectory. "
                f"Fix: resume from a checkpoint written by this version of "
                f"CheckpointManager.save."
            )
        _restore_rng_state(client_state["rng_state"])
        logger.info(
            "Resumed from %s at step %d", checkpoint_path, client_state["global_step"]
        )
        return client_state

    def _prune_old_checkpoints(self) -> None:
        keep_n = self.config.training.keep_last_n_checkpoints
        step_dirs = sorted(
            (p for p in self.output_dir.glob("step_*") if p.is_dir()),
            key=lambda p: int(p.name.split("_")[1]),
        )
        excess = len(step_dirs) - keep_n
        for old_dir in step_dirs[: max(0, excess)]:
            logger.info("Pruning old checkpoint %s", old_dir)
            shutil.rmtree(old_dir, ignore_errors=False)