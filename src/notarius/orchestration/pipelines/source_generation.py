import dagster as dg

from notarius.infrastructure.config.constants import ConfigType, DatasetConfigSubtype
from notarius.infrastructure.config.manager import config_manager
from notarius.orchestration.assets.extract.ingest import raw__hf__dataset
from notarius.orchestration.assets.transform.preprocess import preprocessed__hf__dataset
from notarius.orchestration.assets.transform.transform import (
    base__dataset__pydantic,
    gt__parsed_dataset__pydantic,
    pred__merged_ocr_source_dataset__pydantic,
    GroundTruthDatasetConfig,
)

from notarius.orchestration.assets.transform.predict import (
    pred__llm_ocr_enriched_dataset__pydantic,
    LLMOcrConfig,
)
from notarius.orchestration.assets.transform.source_generation import (
    ocr__exported_json,
    OcrExportConfig,
    source__generated_dataset__pydantic,
    source__exported_json,
    SourceExportConfig,
    SourceGenerationConfig,
)

_all_assets_with_configs: dict = {}

# --- data ingestion ---
_all_assets_with_configs.update(
    {
        raw__hf__dataset: {
            "config": config_manager.load_config_as_model(
                config_name="base_huggingface_config",
                config_type=ConfigType.DATASET,
                config_subtype=DatasetConfigSubtype.DEFAULT,
            ).model_dump()
        },
    }
)

# --- preprocessing ---
_all_assets_with_configs.update(
    {
        preprocessed__hf__dataset: None,
    }
)

# --- dataset split ---
_all_assets_with_configs.update(
    {
        base__dataset__pydantic: None,
        gt__parsed_dataset__pydantic: {
            "config": GroundTruthDatasetConfig(
                ground_truth_source="parsed"
            ).model_dump(),
        },
    }
)

# --- enhance with ocr using llm ---
_all_assets_with_configs.update(
    {
        pred__llm_ocr_enriched_dataset__pydantic: {
            "config": LLMOcrConfig(
                system_prompt="tasks/ocr/system.j2",
                user_prompt="tasks/ocr/user.j2",
                enable_cache=True,
            ).model_dump()
        },
    }
)

# --- export ocr results (backup) ---
_all_assets_with_configs.update(
    {
        ocr__exported_json: {
            "config": OcrExportConfig(
                group_by_schematism=True,
                pretty_print=True,
            ).model_dump()
        },
    }
)

# --- merge ocr with ground truth ---
_all_assets_with_configs.update(
    {
        pred__merged_ocr_source_dataset__pydantic: None,
    }
)

# --- generate source ground truth ---
_all_assets_with_configs.update(
    {
        source__generated_dataset__pydantic: {
            "config": SourceGenerationConfig(
                system_prompt="tasks/source_generation/system.j2",
                user_prompt="tasks/source_generation/user.j2",
                accumulate_context=True,
                enable_cache=True,
                include_next_page_ocr=True,
            ).model_dump()
        },
        source__exported_json: {
            "config": SourceExportConfig(
                group_by_schematism=True, pretty_print=True
            ).model_dump()
        },
    }
)

source_generation_assets = _all_assets_with_configs.keys()


source_generation_job = dg.define_asset_job(
    name="source_generation_pipeline",
    description=(
        "Generate Latin source dataset from Polish ground truth. "
        "Pipeline: HF ingestion → preprocessing → LLM OCR → source generation → JSON export. "
        "Exports results to JSON files for manual review before updating HuggingFace dataset."
    ),
    selection=dg.AssetSelection.assets(*_all_assets_with_configs.keys()),
    config=dg.RunConfig(
        ops={
            asset.key.to_python_identifier(): config
            for asset, config in _all_assets_with_configs.items()
            if config is not None
        },
    )
)
