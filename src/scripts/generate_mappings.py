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
envs = load_dotenv()
if envs:
    logger.info("Loaded environment variables.")
else:
    logger.warning("No environment variables loaded.")

import warnings
from core.utils.mapping_utils import MappingSaver
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

    run = wandb.init(project="ai-osrodek", name="lmv3_llm_eval", mode="online", dir=TMP_DIR)
    mapping_saver = MappingSaver(batch_size=5)

    for fld in _DEFAULT_FIELDS:
        for m in ("precision", "recall", "f1", "accuracy"):
            run.define_metric(f"{fld}/{m}", summary="mean")

    dataset = get_dataset(dataset_config, wrapper=False)

    schematisms_to_filter = dataset_config.full_schematisms + dataset_config.partial_schematisms

    dataset_subsets = {}
    logger.info("Preparing dataset subsets for schematisms...")
    if len(schematisms_to_filter) != 0:
        logger.info(f"Evaluating on: {schematisms_to_filter} schematisms.")
        for schematism in schematisms_to_filter:
            dataset_subset = dataset.filter(
                filter_schematisms(
                    to_filter=schematism
                ),
                input_columns=["schematism_name"]) # type:ignore
            dataset_subsets[schematism] = dataset_subset
    else:
        unique_dataset_schematisms = dataset.unique("schematism_name")
        logger.info(f"Evaluating on: {len(unique_dataset_schematisms)} schematisms.")
        for schematism in unique_dataset_schematisms:
            dataset_subset = dataset.filter(
                filter_schematisms(
                    to_filter=schematism
                ),
                input_columns=["schematism_name"]) # type:ignore
            dataset_subsets[schematism] = dataset_subset

    logger.info("Starting evaluation...")

    for schematism, dataset_subset in dataset_subsets.items():
        logger.info(f"Processing schematism: {schematism} with {len(dataset_subset)} samples.")
        for sample in tqdm(dataset_subset, desc=f"Processing schematism: {schematism}"):
            filename = sample["filename"]
            schematism = sample["schematism_name"]  



            image = sample[dataset_config.image_column_name]  # type: ignore[index]
            results = sample[dataset_config.ground_truth_column_name]  # type: ignore[index]
            results = results.replace('"[brak_informacji]"', "null")
            results_json = json.loads(results)

            kwargs = {
                "metadata": {
                    "schematism": schematism,
                    "filename": filename,
                    "task": "mappings_creation",
                },
                "ground_truth": results_json,
            }

            if len(results_json["entries"]) == 0:
                continue

            lmv3_prediction = lmv3_model.predict(image, **kwargs)

            ocr_text = ocr_model.predict(image, text_only=True, **kwargs)

            kwargs.update(
                {
                "system_prompt": "system-mappings-creation.j2",
                "user_prompt": "user-mappings-creation.j2"
                }
            )

            llm_prediction = llm_model.predict(image=image, hints=lmv3_prediction, text=ocr_text, **kwargs)
            


            mapping_saver.update(schematism, filename, llm_prediction, results_json)



    mapping_saver.save(force=True)
    logger.info("Evaluation finished successfully.")


    run.finish()


if __name__ == "__main__":
    main()
