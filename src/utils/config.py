from pathlib import Path
import yaml


def _load_yaml(file_path: str) -> dict:
    """Load a single YAML file"""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        return config if config else {}


def load_config(config_dir: str = None) -> dict:
    """Load and merge all config files from the specified directory"""
    if not config_dir:
        raise ValueError("Configuration directory path is required")

    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    # Load and merge sub-configs
    sub_configs = [
        'llm.yaml',
        'literature_search.yaml',
        'knowledge_graph.yaml',
        'planning.yaml',
        'idea_generation.yaml',
        'idea_selection.yaml',
        'reviewer.yaml',
        'metadata.yaml'
    ]

    merged_config = {}
    for sub_file in sub_configs:
        sub_path = config_dir / sub_file
        if sub_path.exists():
            sub_config = _load_yaml(str(sub_path))
            merged_config.update(sub_config)
        else:
            raise FileNotFoundError(f"Required config file not found: {sub_path}")

    if not merged_config:
        raise ValueError(f"Configuration is empty after merging all sub-configs")

    return merged_config


def validate_config(config: dict) -> None:
    """Validate all required top-level keys exist, raise AssertionError if missing"""
    required_keys = [
        'llm',
        'token_cost',
        'semantic_scholar',
        'external_selector',
        'knowledge_graph',
        'graph_of_thought',
        'planning_module',
        'idea_generation',
        'internal_selector',
        'reviewer',
        'agent_name',
        'version',
        'logging',
        'ui',
        'proofreading',
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise AssertionError(f"Missing required config keys: {missing_keys}")
