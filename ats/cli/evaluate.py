#!/usr/bin/env python
"""Entry point: python -m ats.cli.evaluate --checkpoint checkpoints/run/step_10000 \
    --tasks mmlu,hellaswag,arc_easy [--output-path ./eval_results] [--device cuda]

Standardized benchmark tasks (mmlu, hellaswag, arc_easy, ...) are delegated
to lm-evaluation-harness rather than reimplemented here -- this file does
NOT contain a hand-rolled benchmark suite; it exports the checkpoint to a
HuggingFace-compatible directory (reusing ats.export.huggingface, only if
not already exported) and shells out to `python -m lm_eval`.

Perplexity on your own held-out data (config.data.sources), which
lm-eval-harness doesn't compute for you, is available as a separate mode:
pass --config instead of --tasks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ats.config.loader import load_config
from ats.config.schema import ConfigError
from ats.export.huggingface import export_to_huggingface
from ats.model.transformer import ATSTransformer
from ats.parallelism.deepspeed_utils import initialize_engine
from ats.training.checkpoint import CheckpointManager
from ats.utils.logging_utils import get_logger, setup_logging

logger = get_logger("ats.cli.evaluate")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an ats-v2 checkpoint, via lm-evaluation-harness for "
        "standard benchmarks or directly for perplexity on your own data.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory, e.g. checkpoints/run/step_10000.",
    )
    parser.add_argument(
        "--tasks",
        default="mmlu,hellaswag,arc_easy",
        help="Comma-separated lm-eval-harness task names. Ignored if --config is given "
        "(perplexity mode).",
    )
    parser.add_argument(
        "--batch-size", default="auto", help="lm-eval-harness batch size, or 'auto'."
    )
    parser.add_argument(
        "--output-path",
        default="./eval_results",
        help="Where lm-eval-harness writes results.",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device for lm-eval-harness (cuda, cpu, ...)."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="If given, run in perplexity mode against config.data.sources instead of "
        "calling lm-eval-harness. Mutually exclusive in effect with --tasks.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Per-GPU micro batch size (perplexity mode only). Overrides "
        "training.micro_batch_size from the config if given.",
    )
    parser.add_argument(
        "--force-reexport",
        action="store_true",
        help="Re-export to HuggingFace format even if checkpoint/hf_exported/config.json already exists.",
    )
    return parser


def _resolve_config_path(checkpoint_dir: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    candidate = checkpoint_dir / "config.yaml"
    if candidate.exists():
        return str(candidate)
    raise ConfigError(
        f"No --config was given and no config.yaml was found at {candidate}. "
        f"Fix: pass --config configs/<size>.yaml explicitly."
    )


def _ensure_hf_export(checkpoint_dir: Path, config_path: str, force: bool) -> Path:
    """Exports the checkpoint to HuggingFace format if not already exported
    (or if --force-reexport is given), returning the export directory."""
    export_dir = checkpoint_dir / "hf_exported"
    if export_dir.exists() and (export_dir / "config.json").exists() and not force:
        logger.info("Reusing existing HuggingFace export at %s", export_dir)
        return export_dir

    config = load_config(config_path)
    if (
        config.model.use_mla
        or config.model.use_moe
        or config.model.use_mod
        or config.model.use_mamba
        or config.model.model_type == "diffusion"
    ):
        raise ConfigError(
            "lm-eval-harness evaluation requires a HuggingFace export, which is only "
            "supported for dense/SWA autoregressive models (see ats/export/huggingface.py). "
            "This checkpoint uses an architecture (MoE/MoD/MLA/Mamba/diffusion) that "
            "cannot be exported. Fix: use --config for perplexity-mode evaluation instead."
        )

    # Bug 4 fix: thread ep_size through from parallelism config (see
    # ats/cli/train.py for the full explanation).
    model = ATSTransformer(
        config.model, ep_size=max(1, config.parallelism.gpus * config.parallelism.nodes)
    )
    model_engine, _optimizer, _, _ = initialize_engine(
        model, config, micro_batch_size=1
    )
    checkpoint_manager = CheckpointManager(config)
    checkpoint_manager.load(model_engine, str(checkpoint_dir))

    module = model_engine.module if hasattr(model_engine, "module") else model_engine
    export_to_huggingface(module, config.model, str(export_dir))
    return export_dir


def _run_lm_eval(export_dir: Path, args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={export_dir},dtype=bfloat16",
        "--tasks",
        args.tasks,
        "--batch_size",
        args.batch_size,
        "--device",
        args.device,
        "--output_path",
        args.output_path,
    ]
    logger.info("Running: %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        logger.error(
            "lm_eval exited with code %d. Common causes: lm-eval isn't installed "
            "(pip install lm-eval>=0.4.0), or --device cuda was requested without a GPU.",
            result.returncode,
        )
    return result.returncode


def _run_perplexity_mode(
    config_path: str, checkpoint_dir: str, micro_batch_size_arg: int | None
) -> int:
    import torch

    from ats.data.dataloader import build_dataloader

    config = load_config(config_path)
    micro_batch_size = (
        micro_batch_size_arg
        if micro_batch_size_arg is not None
        else config.training.micro_batch_size
    )

    # Bug 4 fix: thread ep_size through from parallelism config (see
    # ats/cli/train.py for the full explanation).
    model = ATSTransformer(
        config.model, ep_size=max(1, config.parallelism.gpus * config.parallelism.nodes)
    )
    model_engine, _optimizer, _, _ = initialize_engine(model, config, micro_batch_size)

    checkpoint_manager = CheckpointManager(config)
    client_state = checkpoint_manager.load(model_engine, checkpoint_dir)
    logger.info(
        "Loaded checkpoint from step %d (epoch %d).",
        client_state["global_step"],
        client_state["epoch"],
    )
    model_engine.eval()

    eval_dataloader = build_dataloader(
        config.data,
        batch_size=micro_batch_size,
        rank=0,
        world_size=1,
        seed=config.training.seed,
    )
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            output = model_engine(
                batch["input_ids"], attention_mask=batch.get("attention_mask")
            )
            shift_logits = output.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            num_valid = (shift_labels != -100).sum().item()
            total_loss += float(loss.item())
            total_tokens += int(num_valid)

    if total_tokens == 0:
        logger.error("Eval dataloader produced zero valid tokens.")
        return 1
    perplexity = float(torch.exp(torch.tensor(total_loss / total_tokens)))
    logger.info("Perplexity: %.4f", perplexity)
    return 0


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        logger.error("--checkpoint path does not exist: %s", checkpoint_dir)
        return 1

    if args.config is not None:
        try:
            return _run_perplexity_mode(
                args.config, args.checkpoint, args.micro_batch_size
            )
        except ConfigError as exc:
            logger.error("Config error: %s", exc)
            return 1

    try:
        config_path = _resolve_config_path(checkpoint_dir, None)
        export_dir = _ensure_hf_export(checkpoint_dir, config_path, args.force_reexport)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 1

    return _run_lm_eval(export_dir, args)


if __name__ == "__main__":
    sys.exit(main())
