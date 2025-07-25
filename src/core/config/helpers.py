from __future__ import annotations

from functools import wraps
from typing import Callable, TYPE_CHECKING
import inspect

from core.config.constants import ConfigType, ConfigTypeMapping
from core.exceptions import InvalidConfigType, InvalidConfigSubtype
if TYPE_CHECKING:
    from core.config.manager import ConfigManager

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

_config_manager_instance: ConfigManager | None = None

def get_config_manager() -> ConfigManager:
    """Get or create a singleton instance of ConfigManager."""
    from core.config.manager import ConfigManager
    global _config_manager_instance
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager(CONFIGS_DIR)
    return _config_manager_instance


def with_configs(**config_args):
    """Decorator to inject configurations into function arguments.

    Args:
        **config_args: Mapping of parameter name to (config_name, config_type, config_subtype) tuple

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
            config_manager = get_config_manager()
            injected_kwargs = {}

            for param_name, (config_name, config_type, config_subtype) in config_args.items():
                config = config_manager.load_config(config_type, config_subtype, config_name)
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