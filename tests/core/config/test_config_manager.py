from core.config.registry import ConfigType
from core.config.constants import ModelsConfigSubtype
from pathlib import Path


class TestConfigManager:
    """Test the ConfigManager class."""

    def test_config_manager_init(self, config_manager):
        """Test that the ConfigManager class is initialized correctly."""
        assert config_manager is not None

    def test_config_manager_load_config(self, config_manager):
        """Test that the ConfigManager class can load a config."""
        assert config_manager.load_config(ConfigType.MODELS, ModelsConfigSubtype.LLM, "default") is not None

    def test_config_manager_attributes(self, config_manager):
        """Test that the ConfigManager class has the correct attributes."""
        assert hasattr(config_manager, "configs_dir")
        
        assert isinstance(config_manager.configs_dir, Path)