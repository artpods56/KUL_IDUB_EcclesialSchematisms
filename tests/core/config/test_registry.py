import pytest
from pydantic import BaseModel, Field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from core.config.registry import (
    register_config,
    get_config_schema,
    list_registered_configs,
    validate_config_with_schema,
    get_default_config,
    discover_config_files,
    CONFIG_REGISTRY,
)
from core.config.constants import ConfigType, TestsConfigSubtype
from core.schemas.configs.tests_config import BaseTestsConfig

from structlog import get_logger
logger = get_logger(__name__)


@pytest.fixture
def tests_config_type():
    return ConfigType.TESTS.value

@pytest.fixture
def tests_config_subtype():
    return TestsConfigSubtype.DEFAULT.value

def test_register_config():
    """Test that configs are registered correctly."""
    schema = get_config_schema(ConfigType.TESTS, TestsConfigSubtype.DEFAULT)
    assert schema is not None
    assert schema == BaseTestsConfig


def test_list_registered_configs(tests_config_type, tests_config_subtype):
    """Test listing registered configs."""
    registered = list_registered_configs()
    assert tests_config_type in registered
    assert tests_config_subtype in registered[tests_config_type]


def test_validate_config_with_schema():
    """Test config validation with schema."""
    config_data = {"default_field": "sample_value"}
    validated =  validate_config_with_schema(config_data, BaseTestsConfig)

    assert validated.default_field == "sample_value"


def test_validate_config_with_schema_failure():
    """Test config validation failure."""
    config_data = {"default_field": 0}
    with pytest.raises(Exception):
        validate_config_with_schema(config_data, BaseTestsConfig)


def test_get_default_config():
    """Test getting default config."""
    default = get_default_config(ConfigType.TESTS, TestsConfigSubtype.DEFAULT)
    assert default is not None
    assert default["default_field"] == "default"

def test_discover_config_files():
    """Test discovering config files."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create nested structure
        config_dir = temp_path / "models" / "llm"
        config_dir.mkdir(parents=True)
        
        # Create a test config file
        config_file = config_dir / "test.yaml"
        config_file.write_text("test: config")
        
        # Test discovery
        discovered = discover_config_files(temp_path)
        assert "models" in discovered
        assert "llm" in discovered["models"]
        assert "test" in discovered["models"]["llm"]

def test_default_config_generation(config_manager, monkeypatch):
    """Test discovering config files in flat structure."""
    with TemporaryDirectory() as temp_dir:
        monkeypatch.setattr(config_manager, "configs_dir", Path(temp_dir))

        config_manager.generate_default_configs()
        available_configs = config_manager.list_available_configs()

        for config_type in ConfigType:
            assert config_type.value in available_configs


def test_get_config_schema_nonexistent():
    """Test getting non-existent config schema."""
    with pytest.raises(AttributeError) as exc_info:
        schema = get_config_schema(ConfigType.TESTS, TestsConfigSubtype.NONEXISTENT) # type: ignore

def test_get_default_config_nonexistent():
    """Test getting default config for non-existent schema."""

    with pytest.raises(AttributeError) as exc_info:
        default = get_default_config(ConfigType.TESTS, TestsConfigSubtype.NONEXISTENT) # type: ignore

def test_discover_mixed_config_structure():
    """Test discovering mixed config structures (both direct files and subdirectories)."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mixed structure:
        # models/
        #   ├── config.yaml        (default subtype)
        #   ├── global.yaml        (default subtype)
        #   ├── llm/
        #   │   └── openai.yaml
        #   └── lmv3/
        #       └── base.yaml
        
        models_dir = temp_path / "models"
        models_dir.mkdir()
        
        # Create direct YAML files
        (models_dir / "config.yaml").write_text("config: true")
        (models_dir / "global.yaml").write_text("global: true")
        
        # Create subdirectories
        llm_dir = models_dir / "llm"
        llm_dir.mkdir()
        (llm_dir / "openai.yaml").write_text("api_type: openai")
        
        lmv3_dir = models_dir / "lmv3"
        lmv3_dir.mkdir()
        (lmv3_dir / "base.yaml").write_text("model: base")
        
        # Test discovery
        discovered = discover_config_files(temp_path)
        
        # Verify mixed structure is handled correctly
        assert "models" in discovered
        assert "default" in discovered["models"]
        assert "llm" in discovered["models"]
        assert "lmv3" in discovered["models"]
        
        assert set(discovered["models"]["default"]) == {"config", "global"}
        assert discovered["models"]["llm"] == ["openai"]
        assert discovered["models"]["lmv3"] == ["base"]
