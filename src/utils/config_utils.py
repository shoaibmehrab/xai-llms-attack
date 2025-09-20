"""
Configuration utility functions.
"""

from typing import Dict, Any
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

import json


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML configuration files")
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML configuration files")
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config: Original configuration
        updates: Updates to apply
        
    Returns:
        Updated configuration
    """
    config_copy = config.copy()
    
    def _update_nested(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = _update_nested(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return _update_nested(config_copy, updates)