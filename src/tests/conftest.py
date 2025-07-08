import pytest
from dotenv import load_dotenv
from omegaconf import DictConfig

from core.config.config import ConfigManager
from core.utils.shared import CONFIGS_DIR


@pytest.fixture(scope="session", autouse=True)
def _load_dotenv_once_for_everybody() -> None:
    """Load .env file once per test session."""
    load_dotenv()

@pytest.fixture(scope="session")
def config_manager() -> ConfigManager:
    """One ConfigManager shared by the whole test session."""
    return ConfigManager(str(CONFIGS_DIR))

@pytest.fixture(scope="session")
def dataset_config(config_manager) -> DictConfig:
    """Loads the configuration for a dataset loader."""
    return config_manager.load_config(config_name="dataset_config", config_dir="tests")

@pytest.fixture(scope="session")
def llm_model_config(config_manager) -> DictConfig:
    """Loads the configuration for LLM model."""
    return config_manager.load_config(config_name="llm_config", config_dir="tests")


@pytest.fixture
def lmv3_model_config(config_manager) -> DictConfig:
    """Loads the configuration for LayoutLMv3 model."""
    return config_manager.load_config(config_name="lmv3_config", config_dir="tests")
