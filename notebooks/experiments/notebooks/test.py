from IPython.display import display

from core.utils.logging import setup_logging
setup_logging()
import structlog
logger = structlog.get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()
logger.info("Loaded environment variables.")
import os
import json
from tqdm import tqdm
from datasets import Dataset

from core.utils.shared import CONFIGS_DIR
from core.config.manager import ConfigManager

from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model

from core.data.utils import get_dataset
from core.data.stats import evaluate_json_response
from core.data.filters import filter_schematisms

from core.data.schematism_parser import SchematismParser
from core.data.translation_parser import Parser


from core.config.registry import ConfigType, ModelsConfigSubtype, DatasetConfigSubtype
config_manager = ConfigManager(CONFIGS_DIR)

config_manager.load_config(ConfigType.MODELS, ModelsConfigSubtype.DEFAULT, "default")