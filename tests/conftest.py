from PIL import Image
import pytest
import json

from dotenv import load_dotenv
from omegaconf import DictConfig

from core.config.manager import ConfigManager, ConfigType
from core.config.constants import DatasetConfigSubtype, ModelsConfigSubtype
from core.utils.shared import CONFIGS_DIR
from core.config.helpers import with_configs

import core.schemas.configs


@pytest.fixture(scope="session", autouse=True)
def _load_dotenv_once_for_everybody() -> None:
    """Load .env file once per test session."""
    load_dotenv()

@pytest.fixture(scope="session")
def config_manager() -> ConfigManager:
    """One ConfigManager shared by the whole test session."""
    return ConfigManager(CONFIGS_DIR)


@pytest.fixture(scope="session")
def dataset_config(config_manager) -> DictConfig:
    return config_manager.load_config(
        config_type=ConfigType.DATASET,
        config_subtype=DatasetConfigSubtype.EVALUATION,
        config_name="schematism_dataset_config",
    )
@pytest.fixture(scope="session")
def llm_model_config(config_manager) -> DictConfig:
    """Loads the configuration for LLM model."""
    return config_manager.load_config(
        config_type=ConfigType.MODELS,
        config_subtype=ModelsConfigSubtype.LLM,
        config_name="tests_llm_config",
    )

@pytest.fixture(scope="session")
def lmv3_model_config(config_manager) -> DictConfig:
    """Loads the configuration for LayoutLMv3 model."""
    return config_manager.load_config(
        config_type=ConfigType.MODELS,
        config_subtype=ModelsConfigSubtype.LMV3,
        config_name="lmv3_model_config",
    )

@pytest.fixture
def sample_structured_response():
    return json.dumps({
        "page_number": "56",
        "deanery": None,
        "entries": [
            {
                "parish": "sample",
                "dedication": "sample",
                "building_material": "sample"
            }
        ]
    })

@pytest.fixture
def sample_pil_image():
    pil_image = Image.open("/Volumes/T7/AI_Osrodek/tests/sample_data/0056.jpg")
    return pil_image


@pytest.fixture
def large_sample_image():
    """Large sample image for performance testing"""
    return PIL.Image.new(mode="RGB", size=(2000, 2000))


@pytest.fixture
def malformed_json_response():
    """Sample malformed JSON response"""
    return '{"page_number": "56", "entries": [{'
