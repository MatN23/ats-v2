"""Configuration system: Pydantic schema, size-based defaults, and YAML loading."""

from ats.config.schema import (
    ATSConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    DataSource,
    ParallelismConfig,
    LoggingConfig,
    CheckpointConfig,
    AdaptiveConfig,
)
from ats.config.defaults import MODEL_SIZE_PRESETS, apply_size_preset
from ats.config.loader import load_config

__all__ = [
    "ATSConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "DataSource",
    "ParallelismConfig",
    "LoggingConfig",
    "CheckpointConfig",
    "AdaptiveConfig",
    "MODEL_SIZE_PRESETS",
    "apply_size_preset",
    "load_config",
]
