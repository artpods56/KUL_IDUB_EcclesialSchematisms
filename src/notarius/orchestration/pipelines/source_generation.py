"""Source generation job for creating Latin source dataset from Polish ground truth."""

import dagster as dg

from notarius.orchestration.assets.extract.ingest import raw__hf__dataset
from notarius.orchestration.assets.transform.transform import (
    gt__parsed_dataset__pydantic,
    base__dataset__pydantic,
)
from notarius.orchestration.assets.transform.predict import (
    pred__llm_ocr_enriched_dataset__pydantic,
)
from notarius.orchestration.assets.transform.source_generation import (
    source__generated_dataset__pydantic,
    source__exported_json,
)
from notarius.orchestration.configs.ingestion_config import RAW_HF_DATASET_OP_CONFIG
from notarius.orchestration.configs.prediction_config import (
    PRED__LLM_OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
)
from notarius.orchestration.constants import DataSource, AssetLayer
from notarius.orchestration.utils import AssetKeyHelper

# Assets for source generation pipeline (includes full dependency chain)
source_generation_assets = [
    source__generated_dataset__pydantic,
    source__exported_json,
]

# All assets needed for the complete source generation pipeline
_all_source_generation_assets = [
    # Ingestion
    raw__hf__dataset,
    gt__parsed_dataset__pydantic,
    base__dataset__pydantic,
    # OCR prediction
    pred__llm_ocr_enriched_dataset__pydantic,
    # pred__ocr_enriched_dataset__pydantic,
    # Source generation
    source__generated_dataset__pydantic,
    source__exported_json,
]

# Job definition with full pipeline
source_generation_job = dg.define_asset_job(
    name="source_generation_pipeline",
    selection=dg.AssetSelection.assets(*_all_source_generation_assets),
    description=(
        "Generate Latin source dataset from Polish ground truth. "
        "Includes ingestion (with configurable filtering), OCR, and source generation. "
        "Exports sample to JSON files for manual review before updating HuggingFace dataset."
    ),
    # Default config - can be overridden in launchpad
    config={
        "ops": {
            # Ingestion config
            **RAW_HF_DATASET_OP_CONFIG,
            # Filtering config - customize which schematisms to process
            AssetKeyHelper.build_prefixed_key(
                AssetLayer.INT, DataSource.HUGGINGFACE, "filtered", "hf", "dataset"
            ): {
                "config": OpConfig(
                    op_type="filter",
                    op_name="filter_schematisms",
                    input_columns=["schematism_name"],
                    kwargs={
                        "to_filter": [
                            # Default: process these schematisms for source generation
                            "wloclawek_1872",
                        ]
                    },
                ).model_dump()
            },
            # OCR config
            # AssetKeyHelper.build_prefixed_key(
            #     AssetLayer.STG,
            #     DataSource.HUGGINGFACE,
            #     "pred",
            #     "ocr_enriched_dataset",
            #     "pydantic",
            # ): {
            #     "config": OcrConfig(
            #         text_only=True,
            #         overwrite=False,
            #     ).model_dump()
            # },
            # Source generation config
            "fct__huggingface__source__generated_dataset__pydantic": {
                "config": {
                    "system_prompt": "tasks/source_generation/system.j2",
                    "user_prompt": "tasks/source_generation/user.j2",
                    "accumulate_context": True,
                    "enable_cache": True,
                }
            },
            # Export config
            "mrt__huggingface__source__exported_json": {
                "config": {
                    "group_by_schematism": True,
                    "pretty_print": True,
                }
            },
            **PRED__LLM_OCR_ENRICHED_DATASET__PYDANTIC__OP_CONFIG,
        }
    },
)
