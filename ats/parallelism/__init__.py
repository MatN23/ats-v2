"""GPU-count/model-size aware auto-parallelism selection and DeepSpeed config generation."""

from ats.parallelism.auto_parallel import resolve_strategy
from ats.parallelism.deepspeed_utils import build_deepspeed_config, get_param_groups, initialize_engine

__all__ = ["resolve_strategy", "build_deepspeed_config", "initialize_engine", "get_param_groups"]
