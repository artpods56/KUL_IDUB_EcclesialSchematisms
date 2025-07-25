import os
import json
from core.config.constants import ConfigType, DatasetConfigSubtype, ModelsConfigSubtype
from core.config.helpers import with_configs
from tqdm import tqdm
from omegaconf import DictConfig

import core.schemas.configs

from core.evaluation.runner import EvaluationRunner
from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model
from core.models.ocr.model import OcrModel

from core.data.utils import get_dataset
from core.data.filters import filter_schematisms

from core.data.translation_parser import Parser
from core.utils.shared import  TMP_DIR
from core.utils.logging import setup_logging
setup_logging()

# wandb integration
import wandb
from typing import Tuple

# default metric fields
_DEFAULT_FIELDS: Tuple[str, ...] = (
    "page_number",
    "deanery",
    "parish",
    "dedication",
    "building_material",
)

import structlog
logger = structlog.get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()
logger.info("Loaded environment variables.")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")


@with_configs(
    dataset_config=("schematism_dataset_config", ConfigType.DATASET, DatasetConfigSubtype.EVALUATION),
    llm_model_config=("llm_model_config", ConfigType.MODELS, ModelsConfigSubtype.LLM),
    lmv3_model_config=("lmv3_model_config", ConfigType.MODELS, ModelsConfigSubtype.LMV3)
)
def main(dataset_config: DictConfig, llm_model_config: DictConfig, lmv3_model_config: DictConfig):
    llm_model = LLMModel(llm_model_config)
    lmv3_model = LMv3Model(lmv3_model_config)
    ocr_model = OcrModel()
    evaluation_runner = EvaluationRunner(
        llm_model,
        lmv3_model,
        ocr_model,
        dataset_config,
        parser=Parser()
    )

    run = wandb.init(project="ai-osrodek", name="lmv3_llm_eval", mode="online", dir=TMP_DIR)

    for fld in _DEFAULT_FIELDS:
        for m in ("precision", "recall", "f1", "accuracy"):
            run.define_metric(f"{fld}/{m}", summary="mean")

    dataset = get_dataset(dataset_config, wrapper=False)

    schematisms_to_filter = dataset_config.full_schematisms + dataset_config.partial_schematisms

    dataset_subsets = {}
    logger.info("Preparing dataset subsets for schematisms...")
    for schematism in schematisms_to_filter:
        dataset_subset = dataset.filter(
            filter_schematisms(
                to_filter=schematism
            ),
            input_columns=["schematism_name"]) # type:ignore
        dataset_subsets[schematism] = dataset_subset

    logger.info("Starting evaluation...")

    for schematism, dataset_subset in tqdm(dataset_subsets.items(), desc="Evaluating schematisms"):
        evaluation_runner.run(dataset_subset, logger.bind(schematism=schematism), schematism, run)


    logger.info("Evaluation finished successfully.")

    run.finish()


if __name__ == "__main__":
    main()
