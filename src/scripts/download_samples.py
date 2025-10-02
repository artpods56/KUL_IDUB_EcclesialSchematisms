import os
from datetime import datetime

import warnings
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig
from sqlalchemy import create_engine

import core.schemas.configs  # type: ignore

from core.data.filters import filter_schematisms, filter_empty_samples
from core.config.constants import ConfigType, DatasetConfigSubtype, ModelsConfigSubtype
from core.config.helpers import with_configs
from core.models.base import ModelConfigMap
from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model
from core.models.ocr.model import OcrModel
from core.pipeline.pipeline import (
    IngestionPhase,
    Pipeline,
    SampleProcessingPhase,
    DatasetProcessingPhase,
)
from core.pipeline.steps.export import (
    SaveDataFrameStep,
    AppendDataFrameToSQLStep,
    DownloadSamplesStep,
)
from core.pipeline.steps.ingestion import (
    ImageFileIngestionStep,
    TextFileIngestionStep,
    HuggingFaceIngestionStep,
    FilterMapSpec,
)
from core.pipeline.steps.postprocessing import ReplaceValuesStep
from core.pipeline.steps.wrappers import (
    HuggingFaceToPipelineDataStep,
)
from core.utils.logging import setup_logging
from core.utils.shared import TMP_DIR

setup_logging()

import structlog

logger = structlog.get_logger(__name__)

envs = load_dotenv()
if not envs:
    logger.warning("No environment variables loaded.")

@with_configs(
    dataset_config=(
        "schematism_dataset_config",
        ConfigType.DATASET,
        DatasetConfigSubtype.EVALUATION,
    ),
    llm_model_config=("llm_model_config", ConfigType.MODELS, ModelsConfigSubtype.LLM),
    lmv3_model_config=(
        "lmv3_model_config",
        ConfigType.MODELS,
        ModelsConfigSubtype.LMV3,
    ),
    ocr_model_config=("ocr_model_config", ConfigType.MODELS, ModelsConfigSubtype.OCR),
)
def main(
    dataset_config: DictConfig,
    llm_model_config: DictConfig,
    lmv3_model_config: DictConfig,
    ocr_model_config: DictConfig,
):
    model_configs: ModelConfigMap = {
        LLMModel: llm_model_config,
        LMv3Model: lmv3_model_config,
        OcrModel: ocr_model_config,
    }

    pipeline = Pipeline(model_configs=model_configs, batched=True, batch_size=9)
    schematisms_to_filter = list(
        dataset_config.full_schematisms + dataset_config.partial_schematisms
    )
    languages = list(dataset_config.get("languages", []))
    column_map = dict(dataset_config.get("column_map", {}))

    ingestion_phase = IngestionPhase(
        name="ingestion",
        steps=[
            HuggingFaceToPipelineDataStep(
                wrapped_step=HuggingFaceIngestionStep(
                    dataset_config=dataset_config,
                    filter_map_specs=[
                        FilterMapSpec(operations=[
                            filter_schematisms(schematisms_to_filter),
                            ], input_columns=["schematism_name"], operation_type="filter",
                        ),
                        FilterMapSpec(operations=[
                            filter_empty_samples,
                        ], input_columns=["results"], operation_type="filter",)
                    ],
                ),
                column_map=column_map,
            ),
        ],
        description="Dataset ingestion from local files.",
    )

    export_phase = DatasetProcessingPhase(
        name="export",
        steps=[
            ReplaceValuesStep(
                source="ground_truth",
                old_value="[brak_informacji]",
                new_value=None,
            ),
            DownloadSamplesStep(directory=TMP_DIR / "input", shuffle=False),
        ],
        description="Exporting the data to file and database.",
        depends_on=ingestion_phase,
    )
    pipeline.add_phases(
        [
            ingestion_phase,
            export_phase,
        ]
    )

    pipeline.run()


if __name__ == "__main__":
    main()
