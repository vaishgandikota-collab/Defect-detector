"""
File and configuration management utilities.
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Union


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely loads and returns a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Saves serializable Python object to a JSON file safely.

    Args:
        data: Object to serialize.
        file_path: Output file path.
        indent: JSON indentation.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Loads data from a JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Any: Deserialized object.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensures a directory exists, creating all parents if necessary.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path
