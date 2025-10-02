from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List
import inspect

from core.config.constants import ConfigType, ConfigTypeMapping

from core.exceptions import InvalidConfigType, InvalidConfigSubtype

from core.utils.shared import CONFIGS_DIR


def validate_config_arguments(func):
    """Decorator to validate config_type and config_subtype arguments using signature inspection."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        try:
            bound_args = sig.bind_partial(*args, **kwargs)
        except TypeError as exc:
            # Convert missing-argument TypeError to ValueError to keep backward compatibility
            raise ValueError("Missing config_type or config_subtype for validation.") from exc

        bound_args.apply_defaults()

        # Extract config_type and config_subtype from bound arguments (may be None if absent)
        config_type = bound_args.arguments.get("config_type")
        config_subtype = bound_args.arguments.get("config_subtype")

        # Basic presence checks
        if config_type is None or config_subtype is None:
            raise ValueError("Missing config_type or config_subtype for validation.")

        # Type validation
        if not isinstance(config_type, ConfigType):
            raise InvalidConfigType(
                config_type,
                ConfigType.__members__.keys()
            )

        if not ConfigTypeMapping.is_valid_subtype(config_type, config_subtype):
            raise InvalidConfigSubtype(
                config_type,
                config_subtype,
                ConfigTypeMapping.get_subtype_enum(config_type).__members__.keys()
            )

        return func(*args, **kwargs)

    return wrapper





def with_configs(**config_args):
    """Decorator to inject configurations into function arguments.

    Args:
        **config_args: Mapping of parameter description to (config_name, config_type, config_subtype) tuple

    Example:
        @with_configs(
            model_config=("default", ConfigType.MODELS, ModelsConfigSubtype.LLM),
            dataset_config=("default", ConfigType.DATASET, DatasetConfigSubtype.TRAINING)
        )
        def train_model(model_config, dataset_config):
            # Both configs are automatically loaded and injected
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from core.config.manager import get_config_manager
            config_manager = get_config_manager()
            injected_kwargs = {}

            for param_name, (config_name, config_type, config_subtype) in config_args.items():
                config = config_manager.load_config(config_name, config_type, config_subtype)
                injected_kwargs[param_name] = config

            # Respect original kwargs and let user override if needed
            full_kwargs = {**injected_kwargs, **kwargs}

            # Check if the function supports the right args
            sig = inspect.signature(func)
            missing = [name for name in full_kwargs if name not in sig.parameters]
            if missing:
                raise TypeError(f"Injected unexpected config arguments: {missing}")

            return func(*args, **full_kwargs)
        return wrapper
    return decorator


def discover_config_files(base_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Discover available config files in the directory structure.

    Handles mixed structures:
    - base_path/config_type/*.yaml (flat structure -> 'default' subtype)
    - base_path/config_type/subtype/*.yaml (nested structure)
    - base_path/config_type with both *.yaml files AND subdirectories

    Returns:
        Dict mapping config_type -> subtype -> list of config files
    """
    discovered = {}
    if not base_path.exists():
        return discovered

    for type_dir in base_path.iterdir():
        if not type_dir.is_dir():
            continue
        config_type = type_dir.name
        discovered[config_type] = {}

        # Check for direct YAML files in the type directory
        yaml_files = list(type_dir.glob("*.yaml"))
        if yaml_files:
            # Add direct YAML files as 'default' subtype
            discovered[config_type]['default'] = [f.stem for f in yaml_files]

        # Check for subdirectories (subtypes)
        for item in type_dir.iterdir():
            if item.is_dir():
                subtype = item.name
                subtype_yaml_files = [f.stem for f in item.glob("*.yaml")]
                if subtype_yaml_files:
                    discovered[config_type][subtype] = subtype_yaml_files

        # Remove config_type if no configs were found
        if not discovered[config_type]:
            del discovered[config_type]

    return discovered
