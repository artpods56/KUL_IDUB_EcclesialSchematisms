import dagster as dg

from notarius_dagster.assets.extract.ingest import (
    raw__hf__dataset,
    raw__pdf__dataset,
)
from notarius_dagster.assets.transform.preprocess import preprocessed__hf__dataset
from notarius_dagster.assets.transform.transform import (
    gt__source_dataset__pydantic,
    gt__parsed_dataset__pydantic,
    base__dataset__pydantic,
)
from notarius_dagster.configs.ingestion_config import (
    RAW_HF_DATASET_OP_CONFIG,
    GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
    GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
)

ingestion_assets = [
    raw__pdf__dataset,
    raw__hf__dataset,
    preprocessed__hf__dataset,
    base__dataset__pydantic,
    gt__source_dataset__pydantic,
    gt__parsed_dataset__pydantic,
]

ingestion_assets_configs = {
    **RAW_HF_DATASET_OP_CONFIG,
    **GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG,
    **GT_PARSED_DATASET_PYDANTIC_OP_CONFIG,
}

ingestion_job = dg.define_asset_job(
    name="ingestion_pipeline",
    description="Ingest data from PDF or HuggingFace dataset.",
    selection=dg.AssetSelection.assets(*ingestion_assets),
    config={"ops": ingestion_assets_configs},
)
