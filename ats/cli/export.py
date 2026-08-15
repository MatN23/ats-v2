#!/usr/bin/env python
"""Entry point: python export.py --checkpoint checkpoints/run/step_10000 \
    --output_dir ./exported --format huggingface [--config configs/7b.yaml]"""

from __future__ import annotations

import argparse
import sys

from ats.config.loader import load_config
from ats.config.schema import ConfigError
from ats.export.huggingface import export_to_huggingface
from ats.model.transformer import ATSTransformer
from ats.parallelism.deepspeed_utils import initialize_engine
from ats.training.checkpoint import CheckpointManager
from ats.utils.logging_utils import get_logger, setup_logging

logger = get_logger("ats.export_entrypoint")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an ats-v2 checkpoint to a downstream format.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory, e.g. checkpoints/run/step_10000.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to write the exported model to."
    )
    parser.add_argument(
        "--format",
        choices=["huggingface"],
        default="huggingface",
        help="Export target format. Only 'huggingface' is currently supported.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config file used for the original training run. If omitted, "
        "ats looks for a sibling 'config.yaml' next to the checkpoint.",
    )
    parser.add_argument(
        "--tokenizer_dir",
        default=None,
        help="Directory of tokenizer files to copy alongside the export.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Per-GPU micro batch size used only to initialize the DeepSpeed engine for "
        "loading the checkpoint. Overrides training.micro_batch_size if given.",
    )
    return parser.parse_args(argv)


def _resolve_config_path(args: argparse.Namespace) -> str:
    if args.config is not None:
        return args.config
    from pathlib import Path

    # CheckpointManager.save() writes config.yaml INSIDE the checkpoint's own
    # tag directory (e.g. checkpoints/run/step_10000/config.yaml), not in its
    # parent, so look there first.
    candidate = Path(args.checkpoint) / "config.yaml"
    if candidate.exists():
        return str(candidate)
    legacy_candidate = Path(args.checkpoint).parent / "config.yaml"
    if legacy_candidate.exists():
        return str(legacy_candidate)
    raise ConfigError(
        f"No --config was given and no config.yaml was found at {candidate} "
        f"(or {legacy_candidate}). "
        f"Fix: pass --config configs/<size>.yaml explicitly, matching the config the "
        f"checkpoint was trained with."
    )


def main(argv=None) -> int:
    setup_logging()
    args = parse_args(argv)

    try:
        config_path = _resolve_config_path(args)
        config = load_config(config_path)
    except ConfigError as exc:
        logger.error("Config error: %s", exc)
        return 1

    micro_batch_size = (
        args.micro_batch_size
        if args.micro_batch_size is not None
        else config.training.micro_batch_size
    )

    if config.model.use_mla:
        logger.error(
            "MLA models cannot be exported to LlamaForCausalLM format because MLA uses "
            "a different attention mechanism (compressed latent KV cache, decoupled "
            "RoPE) that Llama's architecture has no equivalent for. "
            "Fix: export a dense/SWA checkpoint, or disable use_mla for the checkpoint "
            "you want to export."
        )
        return 1

    # Bug 4 fix: thread ep_size through from parallelism config (see
    # ats/cli/train.py for the full explanation).
    model = ATSTransformer(
        config.model, ep_size=max(1, config.parallelism.gpus * config.parallelism.nodes)
    )
    model_engine, _optimizer, _, _ = initialize_engine(model, config, micro_batch_size)

    checkpoint_manager = CheckpointManager(config)
    try:
        checkpoint_manager.load(model_engine, args.checkpoint)
    except ConfigError as exc:
        logger.error("Failed to load checkpoint: %s", exc)
        return 1

    try:
        output_path = export_to_huggingface(
            model=model_engine.module
            if hasattr(model_engine, "module")
            else model_engine,
            model_config=config.model,
            output_dir=args.output_dir,
            tokenizer_dir=args.tokenizer_dir,
        )
    except ConfigError as exc:
        logger.error("Export failed: %s", exc)
        return 1

    print(f"Exported model to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
