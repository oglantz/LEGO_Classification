"""Configuration file loader."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
	"""
	Load configuration from YAML file.
	"""
	path = Path(config_path)
	if not path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")
	with open(path, "r") as f:
		config = yaml.safe_load(f)
	return config or {}


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
	"""
	Get nested config value using dot notation.
	"""
	keys = key_path.split(".")
	value: Any = config
	for key in keys:
		if isinstance(value, dict) and key in value:
			value = value[key]
		else:
			return default
	return value


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Merge two config dictionaries, with override_config taking precedence.
	"""
	merged = base_config.copy()
	for key, value in override_config.items():
		if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
			merged[key] = merge_configs(merged[key], value)
		else:
			merged[key] = value
	return merged


