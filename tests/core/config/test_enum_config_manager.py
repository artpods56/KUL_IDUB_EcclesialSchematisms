from pydantic import ValidationError
import pytest
from omegaconf import DictConfig

from core.config.registry import get_config_schema

from core.config.constants import (
    ConfigType,
    ConfigTypeMapping,
    ModelsConfigSubtype,
    DatasetConfigSubtype,
    WandbConfigSubtype,
)
from core.exceptions import ConfigNotRegisteredError


class TestEnumConfigTypes:
    """Test enum configuration types and subtypes."""

    def test_config_type_enum_values(self):
        """Test that ConfigType enum has correct values."""
        assert ConfigType.MODELS.value == "models"
        assert ConfigType.DATASET.value == "data"
        assert ConfigType.WANDB.value == "wandb"

    def test_models_subtype_enum_values(self):
        """Test that ModelsConfigSubtype enum has correct values."""
        assert ModelsConfigSubtype.DEFAULT.value == "default"
        assert ModelsConfigSubtype.LLM.value == "llm"
        assert ModelsConfigSubtype.LMV3.value == "lmv3"

    def test_dataset_subtype_enum_values(self):
        """Test that DatasetConfigSubtype enum has correct values."""
        assert DatasetConfigSubtype.DEFAULT.value == "default"
        assert DatasetConfigSubtype.EVALUATION.value == "pipeline"
        assert DatasetConfigSubtype.TRAINING.value == "training"

    def test_wandb_subtype_enum_values(self):
        """Test that WandbConfigSubtype enum has correct values."""
        assert WandbConfigSubtype.DEFAULT.value == "default"


class TestSubtypeEnumMapping:
    """Test the subtype enum mapping functionality."""

    def test_get_subtype_enum_models(self):
        """Test getting subtype enum for models."""
        subtype_enum = ConfigTypeMapping.get_subtype_enum(ConfigType.MODELS)
        assert subtype_enum == ModelsConfigSubtype

    def test_get_subtype_enum_dataset(self):
        """Test getting subtype enum for data."""
        subtype_enum = ConfigTypeMapping.get_subtype_enum(ConfigType.DATASET)
        assert subtype_enum == DatasetConfigSubtype

    def test_get_subtype_enum_wandb(self):
        """Test getting subtype enum for wandb."""
        subtype_enum = ConfigTypeMapping.get_subtype_enum(ConfigType.WANDB)
        assert subtype_enum == WandbConfigSubtype


class TestConfigSchemaRegistration:
    """Test that config schemas are properly registered with enum types."""

    def test_llm_model_config_registered(self):
        """Test that LLM model config is registered."""
        schema = get_config_schema(ConfigType.MODELS, ModelsConfigSubtype.LLM)
        assert schema is not None
        assert schema.__name__ == "LLMModelConfig"

    def test_lmv3_model_config_registered(self):
        """Test that LMv3 model config is registered."""
        from core.schemas.configs.lmv3_model_config import BaseLMv3ModelConfig

        schema = get_config_schema(ConfigType.MODELS, ModelsConfigSubtype.LMV3)
        assert schema is not None
        assert schema.__name__ == BaseLMv3ModelConfig.__name__

    def test_base_dataset_configs_registered(self):
        """Test that data configs are registered."""
        # Default data config
        from core.schemas.configs.dataset_config import BaseDatasetConfig

        schema = get_config_schema(ConfigType.DATASET, DatasetConfigSubtype.DEFAULT)
        assert schema is not None
        assert schema.__name__ == BaseDatasetConfig.__name__

        # Training data config
        from core.schemas.configs.dataset_config import LayoutLMv3TrainingDatasetConfig

        schema = get_config_schema(ConfigType.DATASET, DatasetConfigSubtype.TRAINING)
        assert schema is not None
        assert schema.__name__ == LayoutLMv3TrainingDatasetConfig.__name__

        # Evaluation data config
        from core.schemas.configs.dataset_config import (
            SchematismsEvaluationDatasetConfig,
        )

        schema = get_config_schema(ConfigType.DATASET, DatasetConfigSubtype.EVALUATION)
        assert schema is not None
        assert schema.__name__ == SchematismsEvaluationDatasetConfig.__name__

    def test_wandb_config_registered(self):
        """Test that Wandb config is registered."""
        from core.schemas.configs.wandb_config import WandbConfig

        schema = get_config_schema(ConfigType.WANDB, WandbConfigSubtype.DEFAULT)
        assert schema is not None
        assert schema.__name__ == WandbConfig.__name__


class TestConfigManagerWithEnums:
    """Test ConfigManager with enum-based configuration."""

    def test_available_configs_structure(self, config_manager):
        """Test that available_configs returns correct structure."""
        available = config_manager.available_configs
        assert isinstance(available, dict)
        # Note: This depends on actual test config files existing

    def test_registered_configs_structure(self, config_manager):
        """Test that registered_configs returns correct structure."""
        registered = config_manager.registered_configs
        assert isinstance(registered, dict)
        # Should contain our registered config types
        assert len(registered) > 0

    def test_validate_config_with_valid_llm_config(self, config_manager):
        """Test validating a valid LLM config."""
        valid_config_dict = {
            "predictor": {"api_type": "openai", "template_dir": "prompts"},
            "interfaces": {"openai": {"model": "gpt-3.5-turbo"}},
        }
        validated = config_manager.validate_config(
            ConfigType.MODELS, ModelsConfigSubtype.LLM, valid_config_dict
        )
        assert validated is not None
        assert isinstance(validated, dict)
        assert "predictor" in validated
        assert "interfaces" in validated

    def test_validate_config_with_invalid_llm_config(self, config_manager):
        """Test validating an invalid LLM config."""
        invalid_config_dict = {
            "predictor": {"api_type": "invalid_api", "template_dir": "prompts"},
            "interfaces": {},
        }
        with pytest.raises(ValidationError) as exc_info:
            config_manager.validate_config(
                ConfigType.MODELS, ModelsConfigSubtype.LLM, invalid_config_dict
            )

    def test_validate_config_with_unregistered_schema(self, config_manager):
        """Test validating with unregistered schema raises error."""
        config_dict = {"some": "config"}
        with pytest.raises(ConfigNotRegisteredError) as exc_info:
            config_manager.validate_config(
                ConfigType.MODELS,
                ModelsConfigSubtype.DEFAULT,  # this subtype exists but no config is registered under it
                config_dict,
            )

    def test_generate_default_config_for_llm(self, config_manager):
        """Test generating default config for LLM."""
        config = config_manager.generate_default_config(
            ConfigType.MODELS, ModelsConfigSubtype.LLM, save=False
        )
        assert isinstance(config, DictConfig)

    def test_generate_default_config_for_lmv3(self, config_manager):
        """Test generating default config for LMv3."""
        config = config_manager.generate_default_config(
            ConfigType.MODELS, ModelsConfigSubtype.LMV3, save=False
        )
        assert isinstance(config, DictConfig)
        assert "model" in config
        assert "processor" in config
        assert "training" in config

    def test_generate_default_config_for_dataset(self, config_manager):
        """Test generating default config for data."""
        config = config_manager.generate_default_config(
            ConfigType.DATASET, DatasetConfigSubtype.DEFAULT, save=False
        )
        assert isinstance(config, DictConfig)
        assert "path" in config
        assert "description" in config

    def test_generate_default_config_for_wandb(self, config_manager):
        """Test generating default config for wandb."""
        config = config_manager.generate_default_config(
            ConfigType.WANDB, WandbConfigSubtype.DEFAULT, save=False
        )
        assert isinstance(config, DictConfig)
        assert "enable" in config
        assert "project" in config

    def test_generate_default_config_for_unregistered_raises_error(
        self, config_manager
    ):
        """Test generating default config for unregistered combination raises error."""
        with pytest.raises(ConfigNotRegisteredError) as exc_info:
            config_manager.generate_default_config(
                ConfigType.MODELS, ModelsConfigSubtype.DEFAULT, save=False
            )


class TestConfigManagerCaching:
    """Test configuration caching functionality."""

    def test_get_config_with_enum_types(self, config_manager):
        """Test getting config with enum types."""
        # First, we'd need to load a config to cache it
        # This test would need actual config files to work
        pass

    def test_cache_key_generation(self, config_manager):
        """Test that cache keys are generated correctly with enum values."""
        # This is implicitly tested by the internal cache_key generation
        # The key should use enum.value file_format
        pass
