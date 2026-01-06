import dagster as dg

from notarius.orchestration.jobs.exporting import exporting_assets
from notarius.orchestration.jobs.ingestion import (
    ingestion_assets,
)
from notarius.orchestration.jobs.postprocessing import postprocessing_assets
from notarius.orchestration.jobs.prediction import (
    models_config_assets,
    models_assets,
    prediction_assets,
)
from notarius.orchestration.configs.exporting_config import (
    EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
    EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
    EVAL__WANDB_EXPORT_DATAFRAME__PANDAS,
)
from notarius.orchestration.configs.postprocessing_config import (
    PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
)
from notarius.orchestration.configs.prediction_config import (
    PRED__OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
    PRED__LLM_OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
)

from notarius.orchestration.configs.transformation_config import (
    FILTERED__HF__DATASET_OP_CONFIG,
    EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
)

from notarius.orchestration.configs.ingestion_config import (
    RAW_HF_DATASET_OP_CONFIG,
    GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
    GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
)

# Combine all assets - Dagster will automatically resolve dependencies and execute in order
all_pipeline_assets = [
    # Step 1: Configs
    *models_config_assets,
    # Step 2: Ingestion
    *ingestion_assets,
    # Step 3: Models (loaded once, used throughout)
    *models_assets,
    # Step 4: Prediction
    *prediction_assets,
    # Step 5: Postprocessing
    *postprocessing_assets,
    # Step 6: Export
    *exporting_assets,
]

# Complete pipeline job - materializes all assets from start to finish
complete_pipeline_job = dg.define_asset_job(
    name="complete_pipeline",
    description="Full end-to-end pipeline: ingestion -> prediction -> postprocessing -> export",
    selection=dg.AssetSelection.assets(*all_pipeline_assets),
    config={
        "ops": {
            **RAW_HF_DATASET_OP_CONFIG,
            # Ingestion configs
            **FILTERED__HF__DATASET_OP_CONFIG,
            **GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
            **GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
            # Prediction configs
            **PRED__LMV3_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LLM_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__LLM_OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
            # Postprocessing configs
            **PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
            **GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
            **GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
            **EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
            # Export configs
            **EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
            **EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
            **EVAL__WANDB_EXPORT_DATAFRAME__PANDAS,
        }
    },
)
