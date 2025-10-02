from pydantic import BaseModel
from typing import Dict, Type, TypeVar, Union
from enum import Enum

from core.exceptions import InvalidConfigType, InvalidConfigSubtype


class ConfigType(Enum):
    """Config types."""

    MODELS = "models"
    DATASET = "dataset"
    WANDB = "wandb"
    TESTS = "tests"


class DatasetConfigSubtype(Enum):
    """Dataset config subtype."""

    DEFAULT = "default"
    EVALUATION = "evaluation"
    TRAINING = "training"
    GENERATION = "generation"


class ModelsConfigSubtype(Enum):
    """Models config subtype."""

    DEFAULT = "default"
    LLM = "llm"
    LMV3 = "lmv3"
    OCR = "ocr"


class WandbConfigSubtype(Enum):
    """Wandb config subtype."""

    DEFAULT = "default"


class TestsConfigSubtype(Enum):
    __test__ = False
    """Tests config subtype."""
    DEFAULT = "default"


ConfigSubTypes = Union[
    DatasetConfigSubtype, ModelsConfigSubtype, WandbConfigSubtype, TestsConfigSubtype
]


class ConfigTypeMapping:
    """Manages configuration type mappings and their relationships."""

    _mappings: Dict[ConfigType, Type[ConfigSubTypes]] = {
        ConfigType.MODELS: ModelsConfigSubtype,
        ConfigType.DATASET: DatasetConfigSubtype,
        ConfigType.WANDB: WandbConfigSubtype,
        ConfigType.TESTS: TestsConfigSubtype,
    }

    @classmethod
    def get_subtype_mapping(cls) -> Dict[ConfigType, Type[ConfigSubTypes]]:
        """Returns the mapping of config types to their subtypes.

        Returns:
            Dict mapping ConfigType to their respective subtype Enums
        """
        return cls._mappings

    @classmethod
    def get_subtype_enum(cls, config_type: ConfigType) -> Type[ConfigSubTypes]:
        """Get the subtype Enum class for a given config type.

        Args:
            config_type: The configuration type to get subtypes for

        Returns:
            The Enum class containing valid subtypes for the given config type

        Raises:
            ValueError: If no subtype enum is registered for the config type
        """
        subtype_enum = cls._mappings.get(config_type)
        if not subtype_enum:
            raise InvalidConfigType(config_type, cls._mappings.keys())
        return subtype_enum

    @classmethod
    def is_valid_subtype(
        cls, config_type: ConfigType, config_subtype: ConfigSubTypes
    ) -> bool:
        """Check if a subtype is valid for a given config type.

        Args:
            config_type: The configuration type to check
            config_subtype: The subtype value to validate

        Returns:
            bool: True if the subtype is valid for the given config type
        """
        enum_class = cls.get_subtype_enum(config_type)
        try:
            return config_subtype in enum_class
        except InvalidConfigSubtype:
            raise InvalidConfigSubtype(
                config_type, config_subtype, enum_class.__members__.keys()
            )

    # @classmethod
    # def get_config_schema(cls, config_type: ConfigType, config_subtype: Type[ConfigSubTypes]) -> Type[ConfigurableModel]:
    #     """Get the registered config schema for a given type and subtype.
    #
    #     Args:
    #         config_type: The configuration type
    #         config_subtype: The configuration subtype
    #     Returns:
    #
    #     """
    #
    #     return
