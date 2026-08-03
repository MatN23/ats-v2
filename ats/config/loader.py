"""Load an ATSConfig from a YAML (or JSON) file, with ${ENV_VAR} substitution."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from pydantic import ValidationError

from ats.config.defaults import apply_size_preset
from ats.config.schema import ATSConfig, ConfigError

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-[^}]*)?\}")


def _substitute_env_vars(value: Any) -> Any:
    """Recursively replace ${VAR} / ${VAR:-default} in strings within a nested structure."""
    if isinstance(value, str):
        def _replace(match: "re.Match[str]") -> str:
            var_name = match.group(1)
            default = match.group(2)
            if var_name in os.environ:
                return os.environ[var_name]
            if default is not None:
                return default[2:]  # strip the leading ":-"
            raise ConfigError(
                f"Config references environment variable '${{{var_name}}}' which is not "
                f"set and has no default. Fix: export {var_name}=<value>, or use "
                f"'${{{var_name}:-<default>}}' syntax in the YAML."
            )
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    return value


def _read_raw(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}. "
            f"Fix: check the --config path, or run `ls configs/` to see available configs."
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML in {path}: {exc}") from exc
    elif path.suffix.lower() == ".json":
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Failed to parse JSON in {path}: {exc}") from exc
    else:
        raise ConfigError(
            f"Unsupported config file extension '{path.suffix}' for {path}. "
            f"Fix: use a .yaml, .yml, or .json file."
        )
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Top-level content of {path} must be a mapping (YAML dict / JSON object), "
            f"got {type(raw).__name__}."
        )
    return raw


def load_config(path: Union[str, Path]) -> ATSConfig:
    """Load, env-substitute, validate, and auto-tune a config file into an ATSConfig."""
    path = Path(path)
    raw = _read_raw(path)
    raw = _substitute_env_vars(raw)

    if "model" not in raw:
        raise ConfigError(
            f"Config {path} is missing the required top-level 'model' section. "
            f"Fix: add a 'model:' section with at least 'size' or the full architecture fields."
        )

    try:
        config = ATSConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config in {path}:\n{exc}") from exc

    resolved_model = apply_size_preset(config.model)
    if resolved_model is not config.model:
        config = config.model_copy(update={"model": resolved_model})

    return config
