#!/usr/bin/env python
"""Entry point: python -m ats.cli.finetune --config configs/7b.yaml \
    --checkpoint checkpoints/run/step_10000 [--lora-r 16] [--output-dir ./lora-out]

LoRA fine-tunes an ats-v2 base checkpoint using `peft`. Reuses the same
config system, dataloader, and Trainer as ats-train -- a fine-tuning run is
just a training run that (a) starts from pretrained weights instead of
random init, and (b) only the injected LoRA adapter parameters have
requires_grad=True, so ats.parallelism.deepspeed_utils.get_param_groups
naturally only optimizes those.

Only dense and SWA autoregressive models are supported in this revision,
matching the same architecture restriction as `ats-export` (MoE/MoD, MLA,
Mamba, and diffusion models have no path to the merged single-checkpoint
output this CLI produces at the end).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from ats.config.loader import load_config
from ats.config.schema import ATSConfig, ConfigError
from ats.data.dataloader import build_dataloader
from ats.export.huggingface import export_to_huggingface
from ats.model.transformer import ATSTransformer
from ats.training.checkpoint import load_model_weights_safetensors
from ats.training.trainer import Trainer
from ats.utils.logging_utils import get_logger, setup_logging

logger = get_logger("ats.finetune")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune an ats-v2 checkpoint with peft.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--checkpoint", dest="checkpoint", default=None,
        help="Base model checkpoint directory to fine-tune from, e.g. "
             "checkpoints/run/step_10000 (must contain model.safetensors, as "
             "written by CheckpointManager.save).",
    )
    parser.add_argument(
        "--base-checkpoint", dest="checkpoint", default=None,
        help="Deprecated alias for --checkpoint.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write the LoRA adapter and merged checkpoint to. "
             "Defaults to checkpoint.output_dir from the config.",
    )

    # --- PEFT overrides (mutate config.peft; see apply_cli_overrides) ---
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument(
        "--target-modules", type=str, default=None,
        help="Comma-separated module names to wrap with LoRA adapters, e.g. "
             "'q_proj,v_proj,o_proj'.",
    )

    parser.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="Per-GPU micro batch size. Overrides training.micro_batch_size.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    return parser


def apply_cli_overrides(config: ATSConfig, args: argparse.Namespace) -> ATSConfig:
    """Same model_copy-then-revalidate pattern as ats.cli.train.apply_cli_overrides."""
    peft_fields = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    peft_updates = {k: v for k, v in peft_fields.items() if v is not None}
    if args.target_modules is not None:
        peft_updates["target_modules"] = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    # Running ats-finetune at all implies LoRA is what the user wants, so
    # force peft.enabled=True regardless of what the YAML says -- there is
    # no other reason to invoke this entry point.
    peft_updates["enabled"] = True
    config = config.model_copy(update={"peft": config.peft.model_copy(update=peft_updates)})

    training_fields = {
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "micro_batch_size": args.micro_batch_size,
    }
    training_updates = {k: v for k, v in training_fields.items() if v is not None}
    if training_updates:
        config = config.model_copy(update={"training": config.training.model_copy(update=training_updates)})

    try:
        config = ATSConfig.model_validate(config.model_dump())
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"CLI overrides produced an invalid config: {exc}") from exc

    return config


def _check_architecture_supported(config: ATSConfig) -> None:
    """Fails fast, before any weights are loaded, for architectures with no
    path to the merged single-checkpoint HF export this CLI produces at the
    end. Mirrors ats.export.huggingface.export_to_huggingface's own checks
    so the error surfaces immediately instead of after a full training run."""
    model_config = config.model
    if model_config.model_type == "diffusion":
        raise ConfigError(
            "ats-finetune does not support model_type='diffusion' in this revision: "
            "there is no merged single-checkpoint export path for diffusion LMs. "
            "Fix: fine-tune an autoregressive checkpoint."
        )
    if model_config.use_mla:
        raise ConfigError(
            "ats-finetune does not support use_mla=True in this revision: MLA has no "
            "standard HuggingFace-compatible merged export path. "
            "Fix: fine-tune a dense or SWA checkpoint."
        )
    if model_config.use_mamba:
        raise ConfigError(
            "ats-finetune does not support use_mamba=True in this revision: Mamba/SSM "
            "blocks have no attention mechanism to merge into a Llama-compatible export. "
            "Fix: fine-tune a dense or SWA checkpoint."
        )
    if model_config.use_moe or model_config.use_mod:
        raise ConfigError(
            "ats-finetune does not support use_moe/use_mod in this revision: MoE and "
            "Mixture-of-Depths have no standard HuggingFace architecture to merge into. "
            "Fix: fine-tune a dense or SWA checkpoint."
        )


def _build_lora_model(model: ATSTransformer, config: ATSConfig):
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ConfigError(
            "ats-finetune requires the 'peft' package, which is not installed. "
            "Fix: pip install 'ats-v2[finetune]' (or `pip install peft` directly)."
        ) from exc

    peft_config = config.peft
    lora_config = LoraConfig(
        r=peft_config.lora_r,
        lora_alpha=peft_config.lora_alpha,
        target_modules=list(peft_config.target_modules),
        lora_dropout=peft_config.lora_dropout,
        bias="none",
        # No task_type: ATSTransformer is not a HuggingFace PreTrainedModel,
        # so we deliberately get the generic PeftModel (forward() is a
        # transparent passthrough to the base model) rather than a
        # task-specific subclass (e.g. PeftModelForCausalLM) that assumes
        # HF generate()/prepare_inputs_for_generation conventions our
        # model doesn't implement.
    )
    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA adapters injected on %s: %d/%d trainable params (%.3f%%)",
        peft_config.target_modules, trainable, total, 100 * trainable / total,
    )
    return peft_model


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

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

    output_dir = Path(args.output_dir if args.output_dir is not None else config.checkpoint.output_dir)
    lora_dir = output_dir / "lora_adapter"
    merged_dir = output_dir / "merged"

    logger.info(
        "Fine-tuning %s from %s with LoRA(r=%d, alpha=%d, target_modules=%s)",
        config.model.name, args.checkpoint,
        config.peft.lora_r, config.peft.lora_alpha, config.peft.target_modules,
    )

    # Bug 4 fix: thread ep_size through from parallelism config (see
    # ats/cli/train.py for the full explanation).
    model = ATSTransformer(
        config.model, ep_size=max(1, config.parallelism.gpus * config.parallelism.nodes)
    )
    try:
        base_weights = load_model_weights_safetensors(args.checkpoint)
    except ConfigError as exc:
        logger.error("Failed to load base checkpoint: %s", exc)
        return 1
    try:
        missing, unexpected = model.load_state_dict(base_weights, strict=False)
    except RuntimeError as exc:
        # load_state_dict(strict=False) tolerates missing/unexpected *keys*
        # but still raises RuntimeError on a shape mismatch for a key that
        # exists in both -- e.g. --config pointing at a different model
        # size than the checkpoint was trained with.
        logger.error(
            "Base checkpoint at %s does not match the model architecture described by "
            "--config (parameter shape mismatch): %s. "
            "Fix: use the exact config the checkpoint was trained with.",
            args.checkpoint, exc,
        )
        return 1
    if missing or unexpected:
        logger.error(
            "Base checkpoint at %s does not match the model architecture described by "
            "--config: %d missing key(s), %d unexpected key(s) (e.g. missing=%s, "
            "unexpected=%s). Fix: use the exact config the checkpoint was trained with.",
            args.checkpoint, len(missing), len(unexpected), missing[:3], unexpected[:3],
        )
        return 1

    try:
        peft_model = _build_lora_model(model, config)
    except ConfigError as exc:
        logger.error("Config error: %s", exc)
        return 1

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = config.parallelism.gpus * config.parallelism.nodes
    train_dataloader = build_dataloader(
        config.data, batch_size=config.training.micro_batch_size,
        rank=rank, world_size=world_size, seed=config.training.seed,
    )

    try:
        trainer = Trainer(
            model=peft_model, config=config, train_dataloader=train_dataloader,
            micro_batch_size=config.training.micro_batch_size,
        )
    except ConfigError as exc:
        logger.error("Trainer initialization failed: %s", exc)
        return 1

    trainer.train()
    logger.info("LoRA fine-tuning complete at step %d.", trainer.global_step)

    # state_dict()-based operations (inside save_pretrained/merge_and_unload)
    # must run on every rank under ZeRO -- the same collective-gather
    # requirement CheckpointManager.save documents -- but only rank 0 should
    # write files, or every rank would race to write the same output paths.
    trained_module = (
        trainer.model_engine.module if hasattr(trainer.model_engine, "module") else trainer.model_engine
    )
    if rank == 0:
        trained_module.save_pretrained(str(lora_dir))

    # peft's merge_and_unload() runs a tied-weights check that expects
    # model.config to be dict-like (model_config.get("tie_word_embeddings"),
    # matching HF PretrainedConfig), but ATSTransformer.config is our own
    # ModelConfig pydantic object, which has no .get(). Swap in a minimal
    # dict-like shim just for the call, then restore the real ModelConfig
    # immediately after -- export_to_huggingface below needs the real one.
    base_module = model  # same nn.Module instance peft_model wraps
    original_config = base_module.config
    base_module.config = {"tie_word_embeddings": config.model.tie_word_embeddings}
    try:
        merged_model = trained_module.merge_and_unload()
    finally:
        base_module.config = original_config

    if rank == 0:
        export_to_huggingface(
            model=merged_model, model_config=config.model, output_dir=str(merged_dir),
        )
        logger.info("Saved LoRA adapter to %s and merged checkpoint to %s", lora_dir, merged_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())