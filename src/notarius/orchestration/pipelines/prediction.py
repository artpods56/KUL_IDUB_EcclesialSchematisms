import dagster as dg

from notarius.orchestration.assets.transform.transform import (
    pred__dataset__pandas,
    pred__merged_ocr_lmv3_dataset__pydantic,
)
from notarius.orchestration.assets.load.export import pred__excel_export_dataframe__pandas
from notarius.orchestration.jobs.ingestion import (
    raw__pdf__dataset,
    raw__hf__dataset,
    preprocessed__hf__dataset,
    base__dataset__pydantic,
    ingestion_assets_configs,
)
from notarius.orchestration.configs.ingestion_config import (
    RAW_HF_DATASET_OP_CONFIG,
)
from notarius.orchestration.jobs.prediction import (
    prediction_assets,
    prediction_assets_configs,
)
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
prediction_pipeline_assets = [
    raw__pdf__dataset,
    raw__hf__dataset,
    preprocessed__hf__dataset,
    base__dataset__pydantic,
    *prediction_assets,
    pred__merged_ocr_lmv3_dataset__pydantic,
    pred__dataset__pandas,
    pred__excel_export_dataframe__pandas,
]

# Merge all configs
prediction_pipeline_config = {
    "ops": {
        # Ingestion configs
        **RAW_HF_DATASET_OP_CONFIG,
        # Prediction configs
        **prediction_assets_configs,
    }
}

prediction_pipeline_job = dg.define_asset_job(
    name="prediction_pipeline",
    description="Complete end-to-end prediction pipeline: ingestion → prediction → postprocessing → exporting.",
    selection=dg.AssetSelection.assets(*prediction_pipeline_assets),
    config=prediction_pipeline_config,
)
