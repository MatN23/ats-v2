"""Pydantic v2 models for the ATS configuration system.

A single YAML file is parsed into an ATSConfig instance. All validation,
auto-tuning (via model.size) and auto-parallelism resolution happen here or
in ats.config.defaults / ats.config.loader, never scattered through the
training code.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigError(ValueError):
    """Raised for invalid ATS configuration. Always carries an actionable message."""


class ModelConfig(BaseModel):
    """Architecture configuration.

    Either set `size` to one of the published presets (see
    ats.config.defaults.MODEL_SIZE_PRESETS) and leave the architecture fields
    unset, or specify every architecture field explicitly. Mixing partial
    overrides with a preset is allowed: explicit fields win, unset fields are
    filled from the preset.
    """

    name: str = "ats_transformer"
    size: Optional[str] = None

    vocab_size: int = 50304
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_seq_len: int = 4096
    tie_word_embeddings: bool = True

    use_moe: bool = False
    num_experts: int = 8
    moe_top_k: int = 2
    moe_capacity_factor: float = 1.25
    moe_load_balancing_weight: float = 0.01

    use_mod: bool = False
    mod_capacity_factor: float = 0.5

    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    use_flash_attention: bool = True
    # Selective activation checkpointing: torch.utils.checkpoint.checkpoint()
    # is applied to every layer_idx where `layer_idx % checkpoint_every_n_layers
    # == 0`. 1 checkpoints every layer (equivalent to the old
    # gradient_checkpointing=True); None or 0 disables checkpointing entirely
    # (equivalent to the old gradient_checkpointing=False, and the default).
    checkpoint_every_n_layers: Optional[int] = None

    use_swa: bool = False
    swa_window_size: int = 4096
    swa_full_attention_interval: int = 4

    use_mla: bool = False
    mla_latent_dim: Optional[int] = None
    mla_compression_ratio: float = 0.25

    use_mamba: bool = False
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_chunk_size: int = 32
    mamba_expand: int = 2
    mamba_every_n_layers: int = 4

    use_mtp: bool = False
    mtp_num_tokens: int = 2

    model_type: Literal["autoregressive", "diffusion"] = "autoregressive"
    diffusion_num_timesteps: int = 1000
    quantization: Literal["none", "int8", "fp8"] = "none"

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_gradient_checkpointing(cls, data: Any) -> Any:
        """Backward compatibility for the old `gradient_checkpointing: bool`
        field, replaced by `checkpoint_every_n_layers: Optional[int]`. Maps
        True -> 1 (checkpoint every layer, the old True behavior) and
        False -> None (disabled, the old False behavior/default). Only
        applied when the new field wasn't also given explicitly, so an
        explicit checkpoint_every_n_layers always wins."""
        if isinstance(data, dict) and "gradient_checkpointing" in data:
            data = dict(data)
            legacy_value = data.pop("gradient_checkpointing")
            if "checkpoint_every_n_layers" not in data:
                warnings.warn(
                    "model.gradient_checkpointing is deprecated; use "
                    "model.checkpoint_every_n_layers instead (1 = every layer, "
                    "matching gradient_checkpointing=True; omit/None to disable, "
                    "matching gradient_checkpointing=False). Mapping "
                    f"gradient_checkpointing={legacy_value!r} -> "
                    f"checkpoint_every_n_layers={1 if legacy_value else None!r}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                data["checkpoint_every_n_layers"] = 1 if legacy_value else None
        return data

    @field_validator("checkpoint_every_n_layers")
    @classmethod
    def _validate_checkpoint_every_n_layers(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ConfigError(
                f"model.checkpoint_every_n_layers must be a positive integer or None, "
                f"got {v}. Fix: use a positive integer (1 = every layer), or omit/null "
                f"to disable checkpointing."
            )
        return v

    @field_validator("vocab_size")
    @classmethod
    def _validate_vocab_size(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(
                f"model.vocab_size must be positive, got {v}. "
                f"Fix: set model.vocab_size to match your tokenizer's vocabulary size."
            )
        return v

    @field_validator("max_seq_len")
    @classmethod
    def _validate_max_seq_len(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(
                f"model.max_seq_len must be positive, got {v}. "
                f"Fix: set model.max_seq_len to a positive integer, e.g. 4096."
            )
        return v

    @field_validator("num_experts")
    @classmethod
    def _validate_num_experts(cls, v: int) -> int:
        if v < 1:
            raise ConfigError(
                f"model.num_experts must be >= 1, got {v}. "
                f"Fix: set model.num_experts to a positive integer, e.g. 8."
            )
        return v

    @field_validator("moe_capacity_factor")
    @classmethod
    def _validate_moe_capacity_factor(cls, v: float) -> float:
        if v <= 0:
            raise ConfigError(
                f"model.moe_capacity_factor must be positive, got {v}. "
                f"Fix: set model.moe_capacity_factor to a positive float, e.g. 1.25."
            )
        return v

    @field_validator("moe_load_balancing_weight")
    @classmethod
    def _validate_moe_load_balancing_weight(cls, v: float) -> float:
        if v < 0:
            raise ConfigError(
                f"model.moe_load_balancing_weight must be >= 0, got {v}. "
                f"A negative value would flip the sign of the load-balancing "
                f"gradient, actively encouraging expert collapse instead of "
                f"discouraging it. Fix: use a small non-negative float, e.g. 0.01."
            )
        return v

    @field_validator("mod_capacity_factor")
    @classmethod
    def _validate_mod_capacity_factor(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ConfigError(
                f"model.mod_capacity_factor must be in (0.0, 1.0], got {v}. "
                f"Fix: use a value like 0.5. (Previously this was only validated "
                f"inside MixtureOfDepths.__init__, failing late at model "
                f"construction instead of at config load time.)"
            )
        return v

    @field_validator("dropout")
    @classmethod
    def _validate_dropout(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ConfigError(
                f"model.dropout must be in [0.0, 1.0), got {v}. "
                f"Fix: set model.dropout to a value like 0.0 or 0.1."
            )
        return v

    @field_validator("moe_top_k")
    @classmethod
    def _validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ConfigError(
                f"model.moe_top_k must be >= 1, got {v}. "
                f"Fix: set model.moe_top_k to at least 1."
            )
        return v

    @field_validator("swa_window_size", "swa_full_attention_interval")
    @classmethod
    def _validate_swa_positive(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(
                f"model.swa_window_size and model.swa_full_attention_interval must be "
                f"positive, got {v}."
            )
        return v

    @field_validator("mla_compression_ratio")
    @classmethod
    def _validate_mla_ratio(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ConfigError(
                f"model.mla_compression_ratio must be in (0.0, 1.0), got {v}. "
                f"Fix: use a value like 0.25 (hidden_size // 4)."
            )
        return v

    @field_validator("mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_every_n_layers", "mamba_chunk_size")
    @classmethod
    def _validate_mamba_positive(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(
                f"model.mamba_d_state/d_conv/expand/every_n_layers/chunk_size must all be positive, got {v}."
            )
        return v

    @field_validator("mtp_num_tokens")
    @classmethod
    def _validate_mtp_num_tokens(cls, v: int) -> int:
        if v < 1:
            raise ConfigError(f"model.mtp_num_tokens must be >= 1, got {v}.")
        return v

    @field_validator("diffusion_num_timesteps")
    @classmethod
    def _validate_diffusion_timesteps(cls, v: int) -> int:
        if v < 2:
            raise ConfigError(f"model.diffusion_num_timesteps must be >= 2, got {v}.")
        return v

    @model_validator(mode="after")
    def _check_mtp_diffusion_incompatible(self) -> "ModelConfig":
        if self.use_mtp and self.model_type == "diffusion":
            raise ConfigError(
                "model.use_mtp=True is incompatible with model.model_type='diffusion': "
                "MTP predicts several future discrete tokens via cross-entropy, which has "
                "no meaning for a continuous denoising objective. "
                "Fix: disable use_mtp, or set model_type back to 'autoregressive'."
            )
        return self

    @model_validator(mode="after")
    def _check_heads(self) -> "ModelConfig":
        if self.num_heads is not None and self.num_kv_heads is not None:
            if self.num_heads % self.num_kv_heads != 0:
                raise ConfigError(
                    f"model.num_heads ({self.num_heads}) must be divisible by "
                    f"model.num_kv_heads ({self.num_kv_heads}) for grouped-query "
                    f"attention. Fix: choose num_kv_heads that evenly divides num_heads."
                )
        if self.hidden_size is not None and self.num_heads is not None:
            if self.hidden_size % self.num_heads != 0:
                raise ConfigError(
                    f"model.hidden_size ({self.hidden_size}) must be divisible by "
                    f"model.num_heads ({self.num_heads}). "
                    f"Fix: choose a hidden_size that is a multiple of num_heads."
                )
        return self

    @property
    def head_dim(self) -> int:
        if self.hidden_size is None or self.num_heads is None:
            raise ConfigError(
                "model.hidden_size and model.num_heads must be resolved before "
                "head_dim can be computed. Did you forget to call "
                "ats.config.defaults.apply_size_preset()?"
            )
        return self.hidden_size // self.num_heads

    @property
    def resolved_mla_latent_dim(self) -> int:
        if self.mla_latent_dim is not None:
            return self.mla_latent_dim
        if self.hidden_size is None:
            raise ConfigError(
                "model.hidden_size must be resolved before resolved_mla_latent_dim "
                "can be computed."
            )
        computed = max(8, int(self.hidden_size * self.mla_compression_ratio))
        return computed

    def is_resolved(self) -> bool:
        required = (self.hidden_size, self.num_layers, self.num_heads,
                    self.num_kv_heads, self.intermediate_size)
        return all(v is not None for v in required)


class TrainingConfig(BaseModel):
    max_steps: int
    learning_rate: float
    min_lr_ratio: float = 0.1
    warmup_steps: int
    grad_clip_norm: float = 1.0
    grad_accum_steps: int = 1
    micro_batch_size: int = 1
    weight_decay: float = 0.1
    eval_every: int = 1000
    save_every: int = 1000
    keep_last_n_checkpoints: int = 3
    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    seed: int = 42

    @field_validator("keep_last_n_checkpoints")
    @classmethod
    def _validate_keep_last_n_checkpoints(cls, v: int) -> int:
        if v < 1:
            raise ConfigError(
                f"training.keep_last_n_checkpoints must be >= 1, got {v}. "
                f"A value of 0 or less would cause CheckpointManager to delete "
                f"every checkpoint -- including the one just saved in the same "
                f"save() call. Fix: set training.keep_last_n_checkpoints to a "
                f"positive integer, e.g. 3."
            )
        return v

    @field_validator("weight_decay")
    @classmethod
    def _validate_weight_decay(cls, v: float) -> float:
        if v < 0:
            raise ConfigError(
                f"training.weight_decay must be >= 0, got {v}. "
                f"Fix: set training.weight_decay to a non-negative float, e.g. 0.1."
            )
        return v

    @field_validator("micro_batch_size")
    @classmethod
    def _validate_micro_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(
                f"training.micro_batch_size must be positive, got {v}. "
                f"Fix: set training.micro_batch_size to the number of sequences "
                f"processed per GPU per forward/backward pass, e.g. 1 or 4."
            )
        return v

    @field_validator("max_steps")
    @classmethod
    def _validate_max_steps(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(
                f"training.max_steps must be positive, got {v}. "
                f"Fix: set training.max_steps to the number of optimizer steps you want to run."
            )
        return v

    @field_validator("learning_rate")
    @classmethod
    def _validate_lr(cls, v: float) -> float:
        if v <= 0:
            raise ConfigError(
                f"training.learning_rate must be positive, got {v}. "
                f"Fix: set training.learning_rate to a small positive float, e.g. 3e-4."
            )
        return v

    @field_validator("warmup_steps")
    @classmethod
    def _validate_warmup(cls, v: int) -> int:
        if v < 0:
            raise ConfigError(
                f"training.warmup_steps must be >= 0, got {v}. "
                f"Fix: set training.warmup_steps to a non-negative integer."
            )
        return v

    @model_validator(mode="after")
    def _check_warmup_vs_max_steps(self) -> "TrainingConfig":
        if self.warmup_steps > self.max_steps:
            raise ConfigError(
                f"training.warmup_steps ({self.warmup_steps}) cannot exceed "
                f"training.max_steps ({self.max_steps}). "
                f"Fix: reduce warmup_steps or increase max_steps."
            )
        return self


class OptimizerConfig(BaseModel):
    """Optimizer selection, kept separate from TrainingConfig's hyperparameters
    (lr/warmup/weight_decay/etc.) since those apply regardless of which
    optimizer implementation runs underneath them."""

    bits: Literal[32, 8] = 32

    @field_validator("bits")
    @classmethod
    def _validate_bits(cls, v: int) -> int:
        if v not in (32, 8):
            raise ConfigError(
                f"optimizer.bits must be 32 or 8, got {v}. "
                f"Fix: set optimizer.bits to 32 (default, torch AdamW) or 8 "
                f"(bitsandbytes 8-bit Adam, requires the '8bit' extra)."
            )
        return v


class PeftConfig(BaseModel):
    """LoRA fine-tuning configuration, consumed by ats.cli.finetune. Ignored
    entirely by ats-train; `enabled` gates whether ats-finetune injects LoRA
    adapters at all."""

    enabled: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05

    @field_validator("lora_r", "lora_alpha")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(f"peft.lora_r and peft.lora_alpha must be positive, got {v}.")
        return v

    @field_validator("target_modules")
    @classmethod
    def _validate_target_modules(cls, v: List[str]) -> List[str]:
        if not v:
            raise ConfigError(
                "peft.target_modules must contain at least one module name, e.g. "
                "['q_proj', 'v_proj']."
            )
        return v

    @field_validator("lora_dropout")
    @classmethod
    def _validate_lora_dropout(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ConfigError(f"peft.lora_dropout must be in [0.0, 1.0), got {v}.")
        return v


class DataSource(BaseModel):
    path: str
    weight: float = 1.0

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if v <= 0:
            raise ConfigError(
                f"data source weight must be positive, got {v}. "
                f"Fix: set weight to a positive float, e.g. 1.0."
            )
        return v


class DataConfig(BaseModel):
    sources: List[DataSource]
    seq_length: int
    tokenizer_name: str = "tiktoken:cl100k_base"
    streaming: bool = True

    @field_validator("sources")
    @classmethod
    def _validate_sources(cls, v: List[DataSource]) -> List[DataSource]:
        if not v:
            raise ConfigError(
                "data.sources must contain at least one entry. "
                "Fix: add at least one {path, weight} entry under data.sources."
            )
        return v

    @field_validator("seq_length")
    @classmethod
    def _validate_seq_length(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(
                f"data.seq_length must be positive, got {v}. "
                f"Fix: set data.seq_length to a positive integer, e.g. 4096."
            )
        return v


class ParallelismConfig(BaseModel):
    strategy: Literal[
        "auto", "deepspeed_zero0", "deepspeed_zero1", "deepspeed_zero2",
        "deepspeed_zero3", "deepspeed_moe", "fsdp",
    ] = "auto"
    gpus: int = 1
    nodes: int = 1

    @field_validator("gpus", "nodes")
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v < 1:
            raise ConfigError(
                f"parallelism.gpus and parallelism.nodes must be >= 1, got {v}. "
                f"Fix: set parallelism.gpus/nodes to at least 1."
            )
        return v


class LoggingConfig(BaseModel):
    project_name: str = "ats-training"
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_every: int = 10

    @field_validator("log_every")
    @classmethod
    def _validate_log_every(cls, v: int) -> int:
        if v <= 0:
            raise ConfigError(
                f"logging.log_every must be positive, got {v}. "
                f"Fix: set logging.log_every to a positive integer, e.g. 10."
            )
        return v


class CheckpointConfig(BaseModel):
    output_dir: str = "./checkpoints"
    save_optimizer: bool = True
    push_to_hub: Optional[str] = None

    @field_validator("push_to_hub")
    @classmethod
    def _validate_push_to_hub(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and "/" not in v:
            raise ConfigError(
                f"checkpoint.push_to_hub must look like 'org/model-name', got '{v}'. "
                f"Fix: use the format 'your-org/your-model-name'."
            )
        return v


class AdaptiveConfig(BaseModel):
    enabled: bool = True
    history_size: int = 1000
    grad_norm_threshold: float = 100.0
    spike_window: int = 20
    spike_ratio: float = 1.5
    plateau_window: int = 100
    plateau_rel_std: float = 0.001
    expert_collapse_threshold: float = 0.01
    min_lr: float = 1e-6

    # --- Plateau-detection stagnation check ---
    # Low relative std over plateau_window alone doesn't distinguish "stuck"
    # from "healthily converging" or "already converged" -- both look flat.
    # A plateau is only treated as a real stagnation signal (eligible for an
    # LR boost) when relative std is below plateau_rel_std AND the loss has
    # improved less than plateau_min_improvement (relative) between the
    # first and second half of the window.
    plateau_min_improvement: float = 0.01

    # Hard cap on back-to-back plateau_lr_boost actions with no intervening
    # emergency/spike cut to reset the count. Without this, a model that has
    # genuinely converged (which also shows near-zero improvement, so it can
    # still pass the stagnation check above) would otherwise get boosted
    # forever, every time the cooldown clears.
    max_consecutive_plateau_boosts: int = 3

    # --- LR multiplier bounds, applied on top of the schedule ---
    # AdaptiveController actions (emergency cuts, spike cuts, plateau
    # boosts) adjust a persistent multiplier applied as
    # effective_lr = scheduler.get_lr(step) * multiplier, rather than
    # overwriting the LR outright. This keeps repeated boosts from
    # compounding the effective LR arbitrarily high, and keeps repeated cuts
    # from collapsing it arbitrarily low.
    max_lr_multiplier: float = 2.0
    min_lr_multiplier: float = 0.05

    # Per-step decay applied to the multiplier's distance from 1.0
    # (new_multiplier = 1.0 + (old_multiplier - 1.0) * lr_multiplier_decay),
    # so a past boost/cut fades back toward the scheduled LR gradually
    # instead of persisting indefinitely.
    lr_multiplier_decay: float = 0.995

    @model_validator(mode="after")
    def _check_windows(self) -> "AdaptiveConfig":
        if self.history_size < 2 * self.spike_window:
            raise ConfigError(
                f"adaptive.history_size ({self.history_size}) must be at least "
                f"2 * adaptive.spike_window ({2 * self.spike_window}) so that spike "
                f"detection has both a recent and an older window available. "
                f"Fix: increase adaptive.history_size or decrease spike_window."
            )
        if self.history_size < self.plateau_window:
            raise ConfigError(
                f"adaptive.history_size ({self.history_size}) must be at least "
                f"adaptive.plateau_window ({self.plateau_window}). "
                f"Fix: increase adaptive.history_size or decrease plateau_window."
            )
        return self

    @field_validator("plateau_min_improvement")
    @classmethod
    def _validate_plateau_min_improvement(cls, v: float) -> float:
        if v < 0:
            raise ConfigError(
                f"adaptive.plateau_min_improvement must be >= 0, got {v}. "
                f"Fix: use a small non-negative float, e.g. 0.01 (1% relative "
                f"improvement required across the plateau window to NOT count "
                f"as stagnant)."
            )
        return v

    @field_validator("max_consecutive_plateau_boosts")
    @classmethod
    def _validate_max_consecutive_plateau_boosts(cls, v: int) -> int:
        if v < 1:
            raise ConfigError(
                f"adaptive.max_consecutive_plateau_boosts must be >= 1, got {v}. "
                f"Fix: use a positive integer, e.g. 3."
            )
        return v

    @field_validator("max_lr_multiplier")
    @classmethod
    def _validate_max_lr_multiplier(cls, v: float) -> float:
        if v <= 1.0:
            raise ConfigError(
                f"adaptive.max_lr_multiplier must be > 1.0, got {v}. "
                f"A value <= 1.0 would prevent plateau_lr_boost from ever having "
                f"any effect. Fix: use a value like 2.0."
            )
        return v

    @field_validator("min_lr_multiplier")
    @classmethod
    def _validate_min_lr_multiplier(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ConfigError(
                f"adaptive.min_lr_multiplier must be in (0.0, 1.0), got {v}. "
                f"Fix: use a value like 0.05."
            )
        return v

    @field_validator("lr_multiplier_decay")
    @classmethod
    def _validate_lr_multiplier_decay(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ConfigError(
                f"adaptive.lr_multiplier_decay must be in [0.0, 1.0), got {v}. "
                f"A value of 1.0 would mean the multiplier never decays back "
                f"toward the schedule. Fix: use a value like 0.995."
            )
        return v

    @model_validator(mode="after")
    def _check_multiplier_bounds(self) -> "AdaptiveConfig":
        if self.min_lr_multiplier >= self.max_lr_multiplier:
            raise ConfigError(
                f"adaptive.min_lr_multiplier ({self.min_lr_multiplier}) must be less "
                f"than adaptive.max_lr_multiplier ({self.max_lr_multiplier}). "
                f"Fix: lower min_lr_multiplier or raise max_lr_multiplier."
            )
        return self


class ATSConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    parallelism: ParallelismConfig = Field(default_factory=ParallelismConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    peft: PeftConfig = Field(default_factory=PeftConfig)

    @model_validator(mode="after")
    def _check_moe_mod_consistency(self) -> "ATSConfig":
        if self.model.use_moe and self.model.num_experts < self.model.moe_top_k:
            raise ConfigError(
                f"model.num_experts ({self.model.num_experts}) must be >= "
                f"model.moe_top_k ({self.model.moe_top_k}). "
                f"Fix: increase num_experts or decrease moe_top_k."
            )
        if self.model.use_moe and self.parallelism.strategy == "auto":
            pass  # resolved later by ats.parallelism.auto_parallel
        return self

    def config_hash(self) -> str:
        """Stable hash of the resolved config, used to validate checkpoint resumption."""
        import hashlib

        payload = self.model_dump_json(exclude={"logging"}).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]