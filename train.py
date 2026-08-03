#!/usr/bin/env python
"""Entry point: python train.py --config configs/7b.yaml [--use-swa] [--use-mla]
    [--architecture all] [--resume DIR] [--max-train-steps N] ...

Every CLI flag below actually overrides the loaded ATSConfig before the model
is constructed; see apply_cli_overrides(). Precedence, highest to lowest:
  1. CLI arguments
  2. YAML config file values
  3. model.size preset defaults (ats.config.defaults)
  4. Pydantic field defaults

Example (the whole point of this file): all of these read the SAME config file
and differ only in which architecture the CLI enables.
  python train.py --config configs/7b.yaml                     # dense
  python train.py --config configs/7b.yaml --use-swa            # SWA
  python train.py --config configs/7b.yaml --use-mla            # MLA
  python train.py --config configs/7b.yaml --use-moe --use-mod  # MoE + MoD
  python train.py --config configs/7b.yaml --architecture all   # everything compatible
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ats.config.loader import load_config
from ats.config.schema import ATSConfig, ConfigError
from ats.data.dataloader import build_dataloader
from ats.model.transformer import ATSTransformer
from ats.training.checkpoint import TrainingHaltError
from ats.training.trainer import DiffusionTrainer, Trainer
from ats.utils.logging_utils import get_logger, setup_logging

logger = get_logger("ats.train")

_ARCHITECTURE_PRESETS = ("dense", "swa", "mla", "mamba", "moe", "mod", "mtp", "all")
_ALL_ARCH_FLAGS = ("use_swa", "use_mla", "use_mamba", "use_moe", "use_mod", "use_mtp")


def _bool_flag(parser: argparse.ArgumentParser, name: str, help_text: str = "") -> None:
    """Registers --name / --no-name as a single tri-state flag: True, False,
    or None if the user passed neither (so YAML/preset values aren't
    clobbered by an argparse default)."""
    parser.add_argument(
        f"--{name}", dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction, default=None, help=help_text,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an ats-v2 model from a single YAML config, with CLI overrides.",
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Checkpoint directory to resume from.")
    parser.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="Per-GPU micro batch size. Overrides training.micro_batch_size from the "
             "config if given; otherwise the resolved config value (YAML, or its "
             "Pydantic default of 1) is used.",
    )
    parser.add_argument("--max-train-steps", type=int, default=None,
                         help="Override training.max_steps for this run only (e.g. smoke tests).")

    # --- Architecture toggles ---
    _bool_flag(parser, "use-swa", "Enable Sliding Window Attention.")
    _bool_flag(parser, "use-mla", "Enable Multi-Head Latent Attention.")
    _bool_flag(parser, "use-mamba", "Replace every Nth block with a Mamba SSM block.")
    _bool_flag(parser, "use-moe", "Enable Mixture-of-Experts FFN routing.")
    _bool_flag(parser, "use-mod", "Enable Mixture-of-Depths token routing.")
    _bool_flag(parser, "use-mtp", "Enable Multi-Token Prediction.")

    parser.add_argument(
        "--architecture", choices=_ARCHITECTURE_PRESETS, default=None,
        help="Convenience preset that sets multiple architecture flags at once. "
             "An explicit --use-x/--no-use-x for the same flag overrides the preset.",
    )
    parser.add_argument("--model-type", choices=["autoregressive", "diffusion"], default=None)
    parser.add_argument("--quantization", choices=["none", "int8", "fp8"], default=None)

    # --- Numeric architecture overrides ---
    parser.add_argument("--swa-window-size", type=int, default=None)
    parser.add_argument("--swa-full-attention-interval", type=int, default=None)
    parser.add_argument("--mla-latent-dim", type=int, default=None)
    parser.add_argument("--mla-compression-ratio", type=float, default=None)
    parser.add_argument("--mamba-d-state", type=int, default=None)
    parser.add_argument("--mamba-d-conv", type=int, default=None)
    parser.add_argument("--mamba-expand", type=int, default=None)
    parser.add_argument("--mamba-every-n-layers", type=int, default=None)
    parser.add_argument("--moe-num-experts", type=int, default=None)
    parser.add_argument("--moe-top-k", type=int, default=None)
    parser.add_argument("--moe-capacity-factor", type=float, default=None)
    parser.add_argument("--mod-capacity-factor", type=float, default=None)
    parser.add_argument("--mtp-num-tokens", type=int, default=None)

    # --- Model size overrides ---
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    _bool_flag(parser, "tie-word-embeddings", "Tie the LM head to the embedding matrix.")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--rms-norm-eps", type=float, default=None)
    parser.add_argument("--rope-theta", type=float, default=None)
    _bool_flag(parser, "use-flash-attention", "Use flash-attn where available.")
    _bool_flag(parser, "gradient-checkpointing", "Enable activation checkpointing.")

    # --- Training overrides ---
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--seed", type=int, default=None)

    # --- Data overrides ---
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    _bool_flag(parser, "streaming", "Stream data sources instead of loading eagerly.")

    # --- Parallelism overrides ---
    parser.add_argument(
        "--parallelism-strategy",
        choices=["auto", "deepspeed_zero0", "deepspeed_zero1", "deepspeed_zero2",
                 "deepspeed_zero3", "deepspeed_moe", "fsdp"],
        default=None,
    )
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=None)

    # --- Other ---
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--project-name", type=str, default=None)
    _bool_flag(parser, "use-wandb", "Log metrics to Weights & Biases.")
    _bool_flag(parser, "use-tensorboard", "Log metrics to TensorBoard.")

    return parser


def _apply_architecture_preset(args: argparse.Namespace) -> None:
    """Mutates args in place: for any of the 6 architecture flags the user did
    NOT explicitly pass (still None at this point), fill it in from
    --architecture. Flags the user DID pass explicitly are left untouched, so
    an explicit --use-x/--no-use-x always wins over the preset."""
    if args.architecture is None:
        return
    preset = args.architecture
    preset_values = {flag: False for flag in _ALL_ARCH_FLAGS}
    if preset == "all":
        preset_values = {flag: True for flag in _ALL_ARCH_FLAGS}
    elif preset != "dense":
        preset_values[f"use_{preset}"] = True
    # preset == "dense": preset_values already all False.

    for flag, value in preset_values.items():
        if getattr(args, flag) is None:
            setattr(args, flag, value)


def apply_cli_overrides(config: ATSConfig, args: argparse.Namespace) -> ATSConfig:
    """Merges CLI arguments into `config` with strict precedence (CLI highest).
    Returns a NEW ATSConfig; `config` is never mutated in place. The merged
    result is re-validated through Pydantic (model_validate) at the end, so
    every field_validator/model_validator — including the
    use_mtp+model_type=diffusion incompatibility check — runs again against
    the final merged values, not just against the original YAML."""
    _apply_architecture_preset(args)

    model_updates = {}
    for flag in _ALL_ARCH_FLAGS:
        value = getattr(args, flag)
        if value is not None:
            model_updates[flag] = value

    if args.model_type is not None:
        model_updates["model_type"] = args.model_type
    if args.quantization is not None:
        model_updates["quantization"] = args.quantization

    numeric_model_fields = {
        "swa_window_size": args.swa_window_size,
        "swa_full_attention_interval": args.swa_full_attention_interval,
        "mla_latent_dim": args.mla_latent_dim,
        "mla_compression_ratio": args.mla_compression_ratio,
        "mamba_d_state": args.mamba_d_state,
        "mamba_d_conv": args.mamba_d_conv,
        "mamba_expand": args.mamba_expand,
        "mamba_every_n_layers": args.mamba_every_n_layers,
        "num_experts": args.moe_num_experts,
        "moe_top_k": args.moe_top_k,
        "moe_capacity_factor": args.moe_capacity_factor,
        "mod_capacity_factor": args.mod_capacity_factor,
        "mtp_num_tokens": args.mtp_num_tokens,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "intermediate_size": args.intermediate_size,
        "vocab_size": args.vocab_size,
        "max_seq_len": args.max_seq_len,
        "tie_word_embeddings": args.tie_word_embeddings,
        "dropout": args.dropout,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "use_flash_attention": args.use_flash_attention,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    for field, value in numeric_model_fields.items():
        if value is not None:
            model_updates[field] = value

    if model_updates:
        config = config.model_copy(update={"model": config.model.model_copy(update=model_updates)})

    training_fields = {
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup_steps": args.warmup_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "grad_accum_steps": args.grad_accum_steps,
        "micro_batch_size": args.micro_batch_size,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "keep_last_n_checkpoints": args.keep_last_n_checkpoints,
        "mixed_precision": args.mixed_precision,
        "seed": args.seed,
    }
    training_updates = {k: v for k, v in training_fields.items() if v is not None}
    if training_updates:
        config = config.model_copy(update={"training": config.training.model_copy(update=training_updates)})

    data_fields = {
        "seq_length": args.seq_length,
        "tokenizer_name": args.tokenizer_name,
        "streaming": args.streaming,
    }
    data_updates = {k: v for k, v in data_fields.items() if v is not None}
    if data_updates:
        config = config.model_copy(update={"data": config.data.model_copy(update=data_updates)})

    parallelism_fields = {
        "strategy": args.parallelism_strategy,
        "gpus": args.gpus,
        "nodes": args.nodes,
    }
    parallelism_updates = {k: v for k, v in parallelism_fields.items() if v is not None}
    if parallelism_updates:
        config = config.model_copy(
            update={"parallelism": config.parallelism.model_copy(update=parallelism_updates)}
        )

    checkpoint_updates = {k: v for k, v in {"output_dir": args.output_dir}.items() if v is not None}
    if checkpoint_updates:
        config = config.model_copy(
            update={"checkpoint": config.checkpoint.model_copy(update=checkpoint_updates)}
        )

    logging_fields = {
        "project_name": args.project_name,
        "use_wandb": args.use_wandb,
        "use_tensorboard": args.use_tensorboard,
    }
    logging_updates = {k: v for k, v in logging_fields.items() if v is not None}
    if logging_updates:
        config = config.model_copy(update={"logging": config.logging.model_copy(update=logging_updates)})

    # model_copy(update=...) bypasses field/model validators entirely, so we
    # force everything to be re-checked by round-tripping through
    # model_validate() on the fully merged dict. This is what actually
    # catches invalid combinations produced by the CLI (e.g. a numeric
    # override that breaks num_heads % num_kv_heads == 0, or
    # use_mtp + model_type=diffusion).
    try:
        config = ATSConfig.model_validate(config.model_dump())
    except Exception as exc:  # noqa: BLE001 -- re-raised as ConfigError immediately below
        raise ConfigError(f"CLI overrides produced an invalid config: {exc}") from exc

    return config


def main(argv: Optional[List[str]] = None) -> int:
    setup_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        config = apply_cli_overrides(config, args)
    except ConfigError as exc:
        logger.error("Config error: %s", exc)
        return 1

    logger.info(
        "Resolved config: model=%s hidden_size=%d num_layers=%d "
        "use_swa=%s use_mla=%s use_mamba=%s use_moe=%s use_mod=%s use_mtp=%s model_type=%s",
        config.model.name, config.model.hidden_size, config.model.num_layers,
        config.model.use_swa, config.model.use_mla, config.model.use_mamba,
        config.model.use_moe, config.model.use_mod, config.model.use_mtp, config.model.model_type,
    )

    model = ATSTransformer(config.model)

    train_dataloader = build_dataloader(
        config.data, batch_size=config.training.micro_batch_size,
        rank=0, world_size=config.parallelism.gpus * config.parallelism.nodes,
        seed=config.training.seed,
    )

    try:
        if config.model.model_type == "diffusion":
            trainer = DiffusionTrainer(
                model=model, config=config, train_dataloader=train_dataloader,
                micro_batch_size=config.training.micro_batch_size,
            )
        else:
            trainer = Trainer(
                model=model, config=config, train_dataloader=train_dataloader,
                micro_batch_size=config.training.micro_batch_size,
            )
    except ConfigError as exc:
        logger.error("Trainer initialization failed: %s", exc)
        return 1

    if args.resume is not None:
        try:
            trainer.resume(args.resume)
        except ConfigError as exc:
            logger.error("Resume failed: %s", exc)
            return 1

    try:
        trainer.train(max_steps=args.max_train_steps)
    except TrainingHaltError as exc:
        logger.error("Training halted by AdaptiveController: %s", exc)
        return 1

    logger.info("Training complete at step %d.", trainer.global_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
