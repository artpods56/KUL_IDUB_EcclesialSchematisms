import dagster as dg

from notarius.orchestration.jobs.ingestion import (
    ingestion_assets,
    ingestion_assets_configs,
)
from notarius.orchestration.jobs.prediction import prediction_assets
from notarius.orchestration.jobs.postprocessing import postprocessing_assets
from notarius.orchestration.jobs.exporting import exporting_assets

from notarius.orchestration.configs.ingestion_config import (
    RAW_HF_DATASET_OP_CONFIG,
    GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
    GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
)
from notarius.orchestration.configs.prediction_config import (
    PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LLM_OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
)
from notarius.orchestration.configs.postprocessing_config import (
    PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
)
from notarius.orchestration.configs.transformation_config import (
    EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
)
from notarius.orchestration.configs.exporting_config import (
    EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
    EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
    EVAL__WANDB_EXPORT_DATAFRAME__PANDAS,
)

# Combine all assets - Dagster will handle ordering based on dependencies
full_pipeline_assets = [
    *ingestion_assets,
    *prediction_assets,
    *postprocessing_assets,
    *exporting_assets,
]

# Merge all configs
full_pipeline_config = {
    "ops": {
        # Ingestion configs
        **RAW_HF_DATASET_OP_CONFIG,
        # **GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
        **GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
        # Prediction configs
        **PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
        **PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
        **PRED__LLM_OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
        # Postprocessing configs
        **PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
        **EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
        **GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
        **GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
        # Exporting configs
        **EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
        **EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
        **EVAL__WANDB_EXPORT_DATAFRAME__PANDAS,
    }
}

full_pipeline_job = dg.define_asset_job(
    name="full_pipeline",
    description="Complete end-to-end pipeline: ingestion → prediction → postprocessing → exporting.",
    selection=dg.AssetSelection.assets(*full_pipeline_assets),
    config=full_pipeline_config,
)
