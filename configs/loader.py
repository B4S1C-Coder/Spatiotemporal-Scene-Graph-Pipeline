"""
YAML configuration loading helpers.

These helpers provide repository-local config loading with optional shallow
section merges for runtime overrides used by tests and callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DETECTION_CONFIG_PATH = REPO_ROOT / "configs" / "detection.yaml"
BYTETRACK_CONFIG_PATH = REPO_ROOT / "configs" / "bytetrack.yaml"


def load_yaml_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load a YAML config file and merge optional overrides.

    Args:
        config_path: YAML file location.
        overrides: Optional shallow/nested override mapping.

    Returns:
        Merged configuration dictionary.
    """
    resolved_config_path = Path(config_path)
    config_data = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_data, dict):
        raise ValueError(f"Config file must contain a YAML object: {resolved_config_path}")

    if overrides:
        return _merge_config_dicts(config_data, overrides)
    return config_data


def _merge_config_dicts(
    base_config: dict[str, Any],
    override_config: dict[str, Any],
) -> dict[str, Any]:
    merged_config = dict(base_config)
    for key, override_value in override_config.items():
        base_value = merged_config.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged_config[key] = _merge_config_dicts(base_value, override_value)
        else:
            merged_config[key] = override_value
    return merged_config
