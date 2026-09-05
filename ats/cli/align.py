#!/usr/bin/env python
"""Entry point: python -m ats.cli.align --config configs/125m.yaml \\
    --base-checkpoint checkpoints/run/step_10000 \\
    --preference-data data/preferences.jsonl [--dpo-beta 0.1] [--output-dir ./aligned]

DPO (Direct Preference Optimization, Rafailov et al. 2023) alignment on top
of an ats-train (optionally ats-finetune) checkpoint. See ats.training.dpo's
module docstring for the algorithm and why DPO specifically (not the more
general "RLHF"/PPO-against-a-reward-model) is what's implemented here.

--method rlhf is still not implemented: classic RLHF needs a trained reward
model and a generation/rollout loop, neither of which exists anywhere in
this codebase, and building both correctly is a substantially larger
undertaking than DPO -- this raises a clear, specific error rather than
either crashing confusingly or silently running DPO under the `rlhf` name.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ats.config.loader import load_config
from ats.config.schema import ATSConfig, ConfigError
from ats.data.preference_dataset import build_preference_dataloader
from ats.model.transformer import ATSTransformer
from ats.training.checkpoint import load_model_weights_safetensors
from ats.training.dpo import DPOTrainer
from ats.utils.logging_utils import get_logger, setup_logging

logger = get_logger("ats.align")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align an ats-v2 checkpoint via DPO.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        default=None,
        help="Base model checkpoint directory to align from, e.g. "
        "checkpoints/run/step_10000 (must contain model.safetensors, as "
        "written by CheckpointManager.save).",
    )
    parser.add_argument(
        "--base-checkpoint",
        dest="checkpoint",
        default=None,
        help="Alias for --checkpoint.",
    )
    parser.add_argument(
        "--preference-data",
        required=True,
        help='Path to a JSONL file of {"prompt": ..., "chosen": ..., "rejected": ...} '
        "lines. See ats.data.preference_dataset for the exact format.",
    )
    parser.add_argument(
        "--method",
        choices=["dpo", "rlhf"],
        default="dpo",
        help="Alignment method. 'rlhf' is not implemented -- see this module's "
        "docstring for why.",
    )
    parser.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO's beta hyperparameter: how strongly the policy is pulled away "
        "from the reference model. Higher = more conservative (stays closer to "
        "the reference). See ats.training.dpo.dpo_loss.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write checkpoints to. Defaults to checkpoint.output_dir "
        "from the config.",
    )
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def apply_cli_overrides(config: ATSConfig, args: argparse.Namespace) -> ATSConfig:
    training_fields = {
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "micro_batch_size": args.micro_batch_size,
    }
    training_updates = {k: v for k, v in training_fields.items() if v is not None}
    if training_updates:
        config = config.model_copy(
            update={"training": config.training.model_copy(update=training_updates)}
        )
        try:
            config = ATSConfig.model_validate(config.model_dump())
        except Exception as exc:
            raise ConfigError(
                f"CLI overrides produced an invalid config: {exc}"
            ) from exc
    return config


def _check_architecture_supported(config: ATSConfig) -> None:
    """DPOTrainer reads model(input_ids).logits directly (see
    ats.training.dpo.DPOTrainer._sequence_logps) -- this works for any
    ATSTransformer architecture (dense, SWA, MLA, Mamba, MoE, MoD all
    produce that same .logits shape), EXCEPT model_type="diffusion": that
    model is a completely different top-level class (DiffusionLM, not
    ATSTransformer -- see ats.training.trainer.DiffusionTrainer's
    construction) with a fundamentally different forward signature
    (requires embed_tokens/timesteps arguments) and no next-token logits to
    take a sequence log-probability of in the first place. This is a
    narrower restriction than ats-finetune's (which also excludes MLA/
    Mamba/MoE/MoD, because THOSE have no merged HuggingFace export path --
    DPO training itself doesn't need one, since it checkpoints natively via
    CheckpointManager rather than exporting)."""
    if config.model.model_type == "diffusion":
        raise ConfigError(
            "ats-align does not support model_type='diffusion': DiffusionLM's "
            "forward pass has no next-token logits to take a DPO sequence "
            "log-probability of (see ats.model.diffusion.DiffusionLM.forward's "
            "signature -- it predicts noise, not next-token probabilities). "
            "Fix: align an autoregressive checkpoint."
        )


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.method == "rlhf":
        logger.error(
            "ats-align --method rlhf is not implemented: classic RLHF needs a "
            "trained reward model and a generation/rollout loop, neither of which "
            "exists in this codebase. Fix: use --method dpo (the default), which "
            "needs neither -- it optimizes directly against preference pairs."
        )
        return 1

    if args.checkpoint is None:
        logger.error("Config error: --checkpoint is required.")
        return 1

    try:
        config = load_config(args.config)
        config = apply_cli_overrides(config, args)
        _check_architecture_supported(config)
    except ConfigError as exc:
        logger.error("Config error: %s", exc)
        return 1

    try:
        base_weights = load_model_weights_safetensors(args.checkpoint)
    except ConfigError as exc:
        logger.error("Failed to load base checkpoint: %s", exc)
        return 1

    ep_size = max(1, config.parallelism.gpus * config.parallelism.nodes)
    policy_model = ATSTransformer(config.model, ep_size=ep_size)
    reference_model = ATSTransformer(config.model, ep_size=ep_size)

    for name, model in (("policy", policy_model), ("reference", reference_model)):
        try:
            missing, unexpected = model.load_state_dict(base_weights, strict=False)
        except RuntimeError as exc:
            logger.error(
                "Base checkpoint at %s does not match the model architecture "
                "described by --config (parameter shape mismatch, loading %s "
                "model): %s. Fix: use the exact config the checkpoint was "
                "trained with.",
                args.checkpoint,
                name,
                exc,
            )
            return 1
        if missing or unexpected:
            logger.error(
                "Base checkpoint at %s does not match the model architecture "
                "described by --config (loading %s model): %d missing key(s), "
                "%d unexpected key(s) (e.g. missing=%s, unexpected=%s). "
                "Fix: use the exact config the checkpoint was trained with.",
                args.checkpoint,
                name,
                len(missing),
                len(unexpected),
                missing[:3],
                unexpected[:3],
            )
            return 1

    output_dir = Path(
        args.output_dir if args.output_dir is not None else config.checkpoint.output_dir
    )
    if args.output_dir is not None:
        config = config.model_copy(
            update={
                "checkpoint": config.checkpoint.model_copy(
                    update={"output_dir": str(output_dir)}
                )
            }
        )

    logger.info(
        "Aligning %s from %s via DPO (beta=%.3f) on preference data %s",
        config.model.name,
        args.checkpoint,
        args.dpo_beta,
        args.preference_data,
    )

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = ep_size
    try:
        train_dataloader = build_preference_dataloader(
            args.preference_data,
            tokenizer_name=config.data.tokenizer_name,
            seq_length=config.data.seq_length,
            batch_size=config.training.micro_batch_size,
            rank=rank,
            world_size=world_size,
            seed=config.training.seed,
            num_workers=config.data.num_workers,
        )
    except ConfigError as exc:
        logger.error("Preference data error: %s", exc)
        return 1

    try:
        trainer = DPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            config=config,
            micro_batch_size=config.training.micro_batch_size,
            beta=args.dpo_beta,
        )
    except ConfigError as exc:
        logger.error("DPOTrainer initialization failed: %s", exc)
        return 1

    data_iter = iter(train_dataloader)
    for _ in range(config.training.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        trainer.train_step(batch)

    if rank == 0:
        ckpt_path = trainer.save_checkpoint()
        logger.info(
            "DPO alignment complete at step %d. Checkpoint saved to %s.",
            trainer.global_step,
            ckpt_path,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
