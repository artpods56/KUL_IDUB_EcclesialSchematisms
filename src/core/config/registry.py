"""Config registry for automatic model discovery and validation."""
from typing import Dict, List, Optional, Tuple, Type, Any, Union, TypeVar
from pathlib import Path
from core.exceptions import ConfigNotRegisteredError
from pydantic import BaseModel, ValidationError

from core.config.constants import ConfigType, ConfigSubTypes
from core.config.helpers import validate_config_arguments

from structlog import get_logger
logger = get_logger(__name__)

S = TypeVar("S", bound=BaseModel)  # S: Represents a specific Pydantic config schema subclass

# Global registry for config schemas
CONFIG_REGISTRY: Dict[Tuple[ConfigType, ConfigSubTypes], Type[BaseModel]] = {}

def register_config(config_type: ConfigType, config_subtype: ConfigSubTypes):
    """Decorator to register a config schema.
    
    Args:
        config_type: Enumerate of the main config category (e.g., 'ConfigType.MODELS', 'ConfigType.DATASET')
        config_subtype: The config subtype enumerate (e.g., 'ModelsConfigSubtype.DEFAULT', 'DatasetConfigSubtype.DEFAULT')
    """
    def decorator(cls: Type[S]) -> Type[S]:
        key = (config_type, config_subtype)
        if key in CONFIG_REGISTRY:
            logger.warning(f"Config schema already registered for {key}, overwriting")
        CONFIG_REGISTRY[key] = cls
        return cls
    return decorator

@validate_config_arguments
def get_config_schema(config_type: ConfigType, config_subtype: ConfigSubTypes) -> Type[BaseModel]:
    config_schema = CONFIG_REGISTRY.get((config_type, config_subtype), None)

    if not config_schema:
        raise ConfigNotRegisteredError(
            config_type,
            config_subtype,
            CONFIG_REGISTRY
        )

    return config_schema

def list_registered_configs() -> Dict[str, List[str]]:
    """List all registered config types and subtypes."""
    result: Dict[str, List[str]] = {}
    for (config_type, config_subtype), schema in CONFIG_REGISTRY.items():
        if config_type.value not in result:
            result[config_type.value] = []
        result[config_type.value].append(config_subtype.value)
    return result

def validate_config_with_schema(config_data: Dict[str, Any], schema_class: Type[S]) -> S:
    """Validate config data against a Pydantic schema.
    
    Args:
        config_data: Configuration data as a dictionary
        schema_class: Pydantic model class to validate against
        
    Returns:
        Validated Pydantic model instance
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        return schema_class(**config_data)
    except ValidationError as e:
        logger.error(
            "config_validation_failed",
            schema_class=schema_class.__name__,
            error=str(e),
            config_data=config_data
        )
        raise

@validate_config_arguments
def get_default_config(config_type: ConfigType, config_subtype: ConfigSubTypes) -> Optional[Dict[str, Any]]:
    """Generate default configuration for a given type and subtype."""
    schema_class = get_config_schema(config_type, config_subtype)
    if not schema_class:
        raise ConfigNotRegisteredError(
            config_type,
            config_subtype,
            CONFIG_REGISTRY
        )
    default_instance = schema_class()
    return default_instance.model_dump()


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
        logger.warning(f"Config directory does not exist: {base_path}")
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
