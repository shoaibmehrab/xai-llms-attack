# Configuration

This directory contains configuration files for different experiments and models.

## Files

- `config.yaml` - Main configuration file
- `model_configs/` - Model-specific configurations
- `experiment_configs/` - Experiment-specific configurations

## Usage

Configuration files use YAML format and can be loaded using the utilities in `src/utils/config_utils.py`.

Example:
```python
from src.utils import load_config

config = load_config("docs/config.yaml")
```