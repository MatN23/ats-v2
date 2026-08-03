"""GPU-count/model-size aware auto-parallelism selection and DeepSpeed config generation."""

from ats.parallelism.auto_parallel import resolve_strategy
from ats.parallelism.deepspeed_utils import build_deepspeed_config, initialize_engine

__all__ = ["resolve_strategy", "build_deepspeed_config", "initialize_engine"]
