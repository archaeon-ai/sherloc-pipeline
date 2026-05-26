"""
Runtime configuration access for services layer.

This module provides access to runtime configuration, delegating to the
module-level singleton in config.py.
"""

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from ..config import Config, get_config, reset_config


def get_runtime_config(config_file: Optional[Path] = None) -> Config:
    """Get runtime configuration, delegating to the module-level singleton.

    Args:
        config_file: Optional path to custom configuration file. If provided,
            bypasses the singleton and loads the specified file directly.

    Returns:
        Config object with all runtime parameters
    """
    if config_file is not None:
        from ..config import load_config
        return load_config(config_file)

    return get_config()


def reset_runtime_config() -> None:
    """Reset the runtime configuration cache.

    Delegates to the canonical reset_config() in the config module.
    """
    reset_config()


def _normalize_for_hash(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _normalize_for_hash(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize_for_hash(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _normalize_for_hash(vars(value))
    if isinstance(value, Path):
        return str(value)
    return value


def compute_config_hash(config: Config) -> str:
    """Compute a deterministic SHA-256 hash for a configuration object."""
    normalized = _normalize_for_hash(config)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
