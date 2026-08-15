"""Configuration system: Pydantic schema, size-based defaults, and YAML loading."""

from ats.config.defaults import MODEL_SIZE_PRESETS, apply_size_preset
from ats.config.loader import load_config
from ats.config.schema import (
    AdaptiveConfig,
    ATSConfig,
    CheckpointConfig,
    DataConfig,
    DataSource,
    LoggingConfig,
    ModelConfig,
    ParallelismConfig,
    TrainingConfig,
)

__all__ = [
    "MODEL_SIZE_PRESETS",
    "ATSConfig",
    "AdaptiveConfig",
    "CheckpointConfig",
    "DataConfig",
    "DataSource",
    "LoggingConfig",
    "ModelConfig",
    "ParallelismConfig",
    "TrainingConfig",
    "apply_size_preset",
    "load_config",
]
