"""Published-recipe defaults per model size.

These numbers come from the publicly documented architectures of
Llama-2/3, Mistral, and DeepSeek-family models. They are starting points,
not guarantees of matching any specific paper's exact hyperparameters.
"""

from __future__ import annotations

from typing import Any, Dict

from ats.config.schema import ConfigError, ModelConfig

# Architecture presets: hidden_size, num_layers, num_heads, num_kv_heads,
# intermediate_size chosen to roughly match published dense-model recipes.
MODEL_SIZE_PRESETS: Dict[str, Dict[str, int]] = {
    "125m": {
        "hidden_size": 768, "num_layers": 12, "num_heads": 12,
        "num_kv_heads": 12, "intermediate_size": 2048,
    },
    "1b": {
        "hidden_size": 2048, "num_layers": 22, "num_heads": 32,
        "num_kv_heads": 8, "intermediate_size": 5632,
    },
    "3b": {
        "hidden_size": 3072, "num_layers": 28, "num_heads": 24,
        "num_kv_heads": 8, "intermediate_size": 8192,
    },
    "7b": {
        "hidden_size": 4096, "num_layers": 32, "num_heads": 32,
        "num_kv_heads": 8, "intermediate_size": 11008,
    },
    "13b": {
        "hidden_size": 5120, "num_layers": 40, "num_heads": 40,
        "num_kv_heads": 8, "intermediate_size": 13824,
    },
    "14b": {
        "hidden_size": 5120, "num_layers": 40, "num_heads": 40,
        "num_kv_heads": 8, "intermediate_size": 13696,
    },
    "70b": {
        "hidden_size": 8192, "num_layers": 80, "num_heads": 64,
        "num_kv_heads": 8, "intermediate_size": 28672,
    },
}

# Training-recipe defaults (learning rate, batch/accum, warmup) keyed by size.
# These are only applied by apply_training_preset, which callers may skip if
# they want to fully control training.py fields themselves.
TRAINING_SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    "125m": {"learning_rate": 6.0e-4, "warmup_steps": 500, "grad_accum_steps": 1},
    "1b": {"learning_rate": 3.0e-4, "warmup_steps": 2000, "grad_accum_steps": 4},
    "3b": {"learning_rate": 2.5e-4, "warmup_steps": 2000, "grad_accum_steps": 8},
    "7b": {"learning_rate": 2.0e-4, "warmup_steps": 2000, "grad_accum_steps": 16},
    "13b": {"learning_rate": 1.5e-4, "warmup_steps": 2000, "grad_accum_steps": 32},
    "14b": {"learning_rate": 1.5e-4, "warmup_steps": 2000, "grad_accum_steps": 32},
    "70b": {"learning_rate": 1.0e-4, "warmup_steps": 3000, "grad_accum_steps": 64},
}

_ARCH_FIELDS = ("hidden_size", "num_layers", "num_heads", "num_kv_heads", "intermediate_size")


def apply_size_preset(model_config: ModelConfig) -> ModelConfig:
    """Fill unset architecture fields on `model_config` from `model_config.size`.

    Explicit fields set by the user always win over the preset. If
    `model_config.size` is None, every architecture field must already be
    set, or a ConfigError is raised naming exactly which fields are missing.
    """
    if model_config.size is None:
        missing = [f for f in _ARCH_FIELDS if getattr(model_config, f) is None]
        if missing:
            raise ConfigError(
                f"model.size was not set, so these architecture fields are required "
                f"but missing: {missing}. Fix: either set model.size to one of "
                f"{sorted(MODEL_SIZE_PRESETS)}, or explicitly set all of {list(_ARCH_FIELDS)}."
            )
        return model_config

    if model_config.size not in MODEL_SIZE_PRESETS:
        raise ConfigError(
            f"Unknown model.size '{model_config.size}'. "
            f"Fix: choose one of {sorted(MODEL_SIZE_PRESETS)}, or unset model.size "
            f"and specify {list(_ARCH_FIELDS)} explicitly."
        )

    preset = MODEL_SIZE_PRESETS[model_config.size]
    updates = {}
    for field in _ARCH_FIELDS:
        if getattr(model_config, field) is None:
            updates[field] = preset[field]

    if not updates:
        return model_config

    return model_config.model_copy(update=updates)


def get_training_preset(size: str) -> Dict[str, Any]:
    if size not in TRAINING_SIZE_PRESETS:
        raise ConfigError(
            f"No training preset for model size '{size}'. "
            f"Fix: choose one of {sorted(TRAINING_SIZE_PRESETS)} or set "
            f"training.learning_rate / warmup_steps / grad_accum_steps explicitly."
        )
    return dict(TRAINING_SIZE_PRESETS[size])
