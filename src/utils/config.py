from pathlib import Path
import yaml
import os


def load_config(config_path: str = None) -> dict:
    """Load configuration from file - config is required"""
    if not config_path:
        raise ValueError("Configuration file path is required")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            if not config:
                raise ValueError(f"Configuration file is empty: {config_path}")
            return config
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")