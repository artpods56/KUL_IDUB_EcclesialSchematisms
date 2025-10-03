import json

import pytest
from PIL import Image
from dotenv import load_dotenv
from omegaconf import DictConfig

from core.config.constants import DatasetConfigSubtype, ModelsConfigSubtype
from core.config.manager import ConfigManager, ConfigType
from core.pipeline.pipeline import Pipeline
from core.schemas.data.pipeline import PipelineData
from core.schemas.data.schematism import SchematismPage
from core.utils.shared import CONFIGS_DIR


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
    return config_manager.load_config(config_name="schematism_dataset_config", config_type=ConfigType.DATASET,
                                      config_subtype=DatasetConfigSubtype.EVALUATION)
@pytest.fixture(scope="session")
def llm_model_config(config_manager) -> DictConfig:
    """Loads the configuration for LLM model."""
    return config_manager.load_config(config_name="tests_llm_config", config_type=ConfigType.MODELS,
                                      config_subtype=ModelsConfigSubtype.LLM)

@pytest.fixture(scope="session")
def lmv3_model_config(config_manager) -> DictConfig:
    """Loads the configuration for LayoutLMv3 model."""
    return config_manager.load_config(config_name="lmv3_model_config", config_type=ConfigType.MODELS,
                                      config_subtype=ModelsConfigSubtype.LMV3)


@pytest.fixture(scope="session")
def pipeline(llm_model_config, lmv3_model_config) -> Pipeline:
    pass

@pytest.fixture
def sample_structured_response() -> str:
    return json.dumps({
        "page_number": "56",
        "entries": [
            {
                "parish": "sample",
                "deanery": None,
                "dedication": "sample",
                "building_material": "sample"
            }
        ]
    })



@pytest.fixture
def sample_pil_image():
    pil_image = Image.open("tests/sample_data/0056.jpg")
    return pil_image


@pytest.fixture
def sample_page_data(sample_structured_response) -> SchematismPage:
    return SchematismPage(**json.loads(sample_structured_response))

@pytest.fixture
def sample_pipeline_data(sample_structured_response, sample_pil_image, sample_page_data):

    pipeline_items = {
        "image": sample_pil_image,
        "ground_truth": sample_page_data,
        "text": "sample text",
        "language": "unknown",
        "language_confidence": 0.0,
        "lmv3_prediction": sample_page_data,
        "llm_prediction": sample_page_data,
        "metadata": {}
    }

    return PipelineData(**pipeline_items)

@pytest.fixture
def large_sample_image():
    """Large sample image for performance testing"""
    return Image.new(mode="RGB", size=(2000, 2000))


@pytest.fixture
def malformed_json_response():
    """Sample malformed JSON response"""
    return '{"page_number": "56", "entries": [{'
