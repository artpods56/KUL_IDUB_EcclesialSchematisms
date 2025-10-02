
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, cast, overload

from core.exceptions import ConfigNotRegisteredError
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from core.config.registry import (
    ConfigSubTypes,
    ConfigType,
    get_config_schema,
    get_default_config,
    list_registered_configs,
    validate_config_with_schema
)
from core.config.helpers import validate_config_arguments, discover_config_files
from core.config.constants import ConfigTypeMapping
from core.utils.shared import CONFIGS_DIR

from structlog import get_logger
logger = get_logger(__name__)

def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf config to regular dict."""
    return cast(Dict[str, Any], OmegaConf.to_container(config, resolve=True))

class ConfigManager:
    """Centralized configuration management with automatic registry."""

    def __init__(self, configs_dir: Path):
        self.configs_dir = configs_dir
        self._configs = {}
        self._available_configs = None

    def _get_output_file_path(self, config_type: ConfigType, config_subtype: ConfigSubTypes, config_name: str) -> Path:
        """Get the output file path for a configuration file."""
        output_dir = self.configs_dir / config_type.value
        if config_subtype.value != "default":
            output_dir = output_dir / config_subtype.value
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{config_name}.yaml"

    def _get_cache_key(self, config_type: ConfigType, config_subtype: ConfigSubTypes, config_name: str) -> str:
        """Generate a consistent cache key for a configuration."""
        return f"{config_type.value}.{config_subtype.value}.{config_name}"

    @property
    def available_configs(self) -> Dict[str, Dict[str, list]]:
        """Get available configs by discovering files and checking registry."""
        if self._available_configs is None:
            self._available_configs = discover_config_files(self.configs_dir)
        return self._available_configs

    @property
    def registered_configs(self) -> Dict[str, list]:
        """Get all registered config schemas."""
        return list_registered_configs()

    @validate_config_arguments
    def validate_config(self, config_type: ConfigType, config_subtype: ConfigSubTypes, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against appropriate Pydantic model."""
        model_class = get_config_schema(config_type, config_subtype)
        validated_config = validate_config_with_schema(config_dict, model_class)
        return validated_config.model_dump()


    @validate_config_arguments
    def load_config(self, config_name: str, config_type: ConfigType, config_subtype: ConfigSubTypes) -> DictConfig:
        """Load configuration from file and validate if applicable."""

        subtype_enum = ConfigTypeMapping.get_subtype_enum(config_type)

        config_dir = self.configs_dir / Path(
            str(config_type.value)
        )
        # don't nest if its default subtype
        if config_subtype.value != "default":
            config_dir = config_dir / Path(
                str(config_subtype.value)
            )

        # Check if config file exists
        config_file = config_dir / f"{config_name}.yaml"
        if not config_file.exists():
            logger.error("Config file not found", config_file=str(config_file))
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
            config = compose(config_name=config_name)
            config_dict = config_to_dict(config)

            validated_dict = self.validate_config(config_type, config_subtype, config_dict)
            validated_config: DictConfig = OmegaConf.create(validated_dict)

            cache_key = self._get_cache_key(config_type, config_subtype, config_name)
            self._configs[cache_key] = validated_config
            return validated_config

    @validate_config_arguments
    def _generate_default_config(self, config_type: ConfigType, config_subtype: ConfigSubTypes) -> DictConfig:
        """Generate default configuration for a given config type and subtype."""
        config_dict = get_default_config(config_type, config_subtype)
        config = OmegaConf.create(config_dict)
        return config

    def _save_config(self, config_name: str, config_type: ConfigType, config_subtype: ConfigSubTypes, config: DictConfig):
        """Save configuration to a YAML file."""
        output_file = self._get_output_file_path(config_type, config_subtype, config_name)
        with output_file.open("w") as f:
            OmegaConf.save(config=config, f=f)

        logger.info("Saved config to file", config_name=config_name, config_type=config_type, 
                   config_subtype=config_subtype, path=str(output_file))

    def generate_default_configs(self, overwrite: bool = False, save: bool = True):
        """Generate default configurations for all registered schemas."""
        from core.config.registry import CONFIG_REGISTRY
        for (config_type, config_subtype), model_class in CONFIG_REGISTRY.items():
            logger.info("Generating default config.", config_type=config_type.value, config_subtype=config_subtype)
            default_config = self._generate_default_config(config_type, config_subtype)

            if save:
                output_file = self._get_output_file_path(config_type, config_subtype, "default")
                if output_file.exists() and not overwrite:
                    raise ValueError(f"Config already exists for {config_type}.{config_subtype}. Use overwrite=True.")

                self._save_config(
                    config_name="default",
                    config_type=config_type,
                    config_subtype=config_subtype,
                    config=default_config
                )
        self.refresh_available_configs()

    @validate_config_arguments
    def generate_default_config(self, config_type: ConfigType, config_subtype: ConfigSubTypes,
                               config_name: str = "default", save: bool = True) -> DictConfig:
        """Generate and optionally save a default configuration for specific type/subtype."""
        default_config = self._generate_default_config(config_type, config_subtype)
        if save:
            self._save_config(config_name, config_type, config_subtype, default_config)
        return default_config

    @validate_config_arguments
    def get_config(self, config_type: ConfigType, config_subtype: ConfigSubTypes, config_name: str) -> Optional[DictConfig]:
        """Get cached configuration."""
        cache_key = self._get_cache_key(config_type, config_subtype, config_name)
        return self._configs.get(cache_key)

    def list_available_configs(self) -> Dict[str, Dict[str, List[str]]]:
        """List all available configuration files."""
        return self.available_configs

    def refresh_available_configs(self):
        """Refresh the cache of available configuration files."""
        self._available_configs = None


def get_config_manager() -> ConfigManager:
    """Get or create a singleton instance of ConfigManager."""
    config_manager_instance = ConfigManager(CONFIGS_DIR)
    return config_manager_instance
