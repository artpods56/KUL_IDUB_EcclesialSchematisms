import os
import warnings
from datetime import datetime

import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig
from sqlalchemy import create_engine

import core.schemas.configs  # type: ignore
from core.config.constants import ConfigType, DatasetConfigSubtype, ModelsConfigSubtype
from core.config.helpers import with_configs
from core.data.filters import filter_schematisms, filter_empty_samples
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
from core.pipeline.steps.base import IngestionProcessingStep, DatasetProcessingStep
from core.pipeline.steps.evaluation import SampleEvaluationStep
from core.pipeline.steps.export import (
    SaveDataFrameStep,
)
from core.pipeline.steps.ingestion import (
    HuggingFaceIngestionStep,
    FilterMapSpec,
    PdfFileIngestionStep,
)
from core.pipeline.steps.logging import WandbLoggingStep
from core.pipeline.steps.postprocessing import (
    DeaneryFillingStep,
    EntriesParsingStep,
)
from core.pipeline.steps.prediction import (
    OCRStep,
    LMv3PredictionStep,
    LLMPredictionStep,
)
from core.pipeline.steps.wrappers import (
    HuggingFaceToPipelineDataStep,
    DataFrameSchemaMappingStep,
    PipelineDataToPandasDataFrameStep,
)
from core.utils.logging import setup_logging
from core.utils.shared import TMP_DIR, DATA_DIR

setup_logging()

import structlog

logger = structlog.get_logger(__name__)

envs = load_dotenv()
if not envs:
    logger.warning("No environment variables loaded.")

PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_NAME = os.getenv("PG_NAME")

sql_engine = create_engine(
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_NAME}"
)

wandb_run = wandb.init(
    project="ai-osrodek",
    name=f"inference_run_{datetime.now().isoformat()}",
    mode="online",
    dir=TMP_DIR,
)


warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.modeling_utils"
)


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

    pdf_ingestion_steps: list[IngestionProcessingStep] = []

    for pdf_path in (DATA_DIR / "pdfs").iterdir():
        pdf_ingestion_steps.append(
            PdfFileIngestionStep(file_path=pdf_path, modes={"image", "text"})
        )

    ingestion_phase = IngestionPhase(
        name="ingestion",
        steps=[
            *pdf_ingestion_steps,  # unpack the list of pdf ingestion steps
            # PdfFileIngestionStep(
            #     file_path=DATA_DIR / "pdfs" / "Warszawska_1846.pdf",
            #     modes={"image", "text"},
            # ),
            # ImageFileIngestionStep(
            #     data_directory=TMP_DIR / "input", file_extensions=[".jpg"]
            # ),
            # TextFileIngestionStep(
            #     data_directory=TMP_DIR / "input", file_extensions=[".txt"]
            # ),
        ],
        description="Dataset ingestion from local files.",
    )

    prediction_phase = SampleProcessingPhase(
        name="prediction",
        steps=[
            OCRStep(pipeline.get_model(OcrModel), force_ocr=False),
            # LanguageDetectionStep(languages),
            LMv3PredictionStep(pipeline.get_model(LMv3Model)),
            LLMPredictionStep(
                pipeline.get_model(LLMModel),
                system_prompt="system.j2",
                user_prompt="user.j2",
            ),
        ],
        description="Main prediction steps.",
        depends_on=ingestion_phase,
    )

    processing_phase = DatasetProcessingPhase(
        name="processing",
        steps=[
            DeaneryFillingStep(
                sources=["source_ground_truth", "ground_truth", "llm_prediction"]
            ),
            EntriesParsingStep(),
        ],
        description="Filling deanery names between sample entries and parsing them.",
        depends_on=prediction_phase,
    )

    logging_phase = DatasetProcessingPhase(
        name="logging",
        steps=[
            WandbLoggingStep(wandb_run=wandb_run, group_by_metadata_key="schematism")
        ],
        description="Aggregation and logging of the data.",
        depends_on=processing_phase,
    )

    experimental_export = DatasetProcessingPhase(
        name="experimental_export",
        steps=[
            PipelineDataToPandasDataFrameStep(
                wrapped_step=SaveDataFrameStep(
                    file_format="excel",
                    overwrite=True,
                    file_name="predictions",
                    group_by_metadata_key="schematism",
                ),
                source="llm_prediction",
            ),
            PipelineDataToPandasDataFrameStep(
                wrapped_step=SaveDataFrameStep(
                    file_format="excel",
                    overwrite=True,
                    file_name="parsed_predictions",
                    group_by_metadata_key="schematism",
                ),
                source="parsed_prediction",
            ),
        ],
        description="Export the data to Excel.",
        depends_on=logging_phase,
    )

    # export_phase = DatasetProcessingPhase(
    #     name="export",
    #     steps=[
    #         ToPandasDataFrameStep(source="parsed"),
    #         SaveDataFrameStep(
    #             file_path=TMP_DIR / "saved.xlsx", overwrite=True, file_format="excel"
    #         ),
    #         DataFrameSchemaMappingStep(
    #             mapping={
    #                 "parish": "parafia",
    #                 "deanery": "dekanat",
    #                 "dedication": "wezwanie",
    #                 "building_material": "material",
    #                 "page_number": "strona_p",
    #             },
    #             strict=True,
    #         ),
    #         SaveDataFrameStep(
    #             file_path=TMP_DIR / "saved_mapped.csv", overwrite=True, file_format="csv"
    #         ),
    # AppendDataFrameToSQLStep(
    #     table_name="dane_hasla",
    #     connection=create_engine(
    #         f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_NAME}"
    #     ),
    #     if_exists="append",
    #     use_columns=["parafia", "dekanat", "wezwanie", "material", "strona_p"],
    # ),
    # ],
    #     description="Exporting the data to file and database.",
    #     depends_on=experimental_export,
    # )
    pipeline.add_phases(
        [
            prediction_phase,
            ingestion_phase,
            processing_phase,
            experimental_export,
            logging_phase,
        ]
    )

    pipeline.run()


if __name__ == "__main__":
    main()
