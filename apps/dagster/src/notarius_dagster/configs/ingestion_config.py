from notarius.infrastructure.config.constants import (
    ConfigType,
    DatasetConfigSubtype,
)
from notarius.infrastructure.config.manager import config_manager
from notarius_dagster.assets.transform.transform import GroundTruthDatasetConfig
from notarius_dagster.constants import DataSource, AssetLayer
from notarius_dagster.utils import AssetKeyHelper

"""
Asset: [[ingest.py#raw__hf__dataset]]
Defined in: [[src/orchestration/assets/extract/ingest.py]]
Resolves to: stg__huggingface__raw__hf__dataset
"""
RAW_HF_DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.STG, DataSource.HUGGINGFACE, "raw", "hf", "dataset"
    ): {
        "config": config_manager.load_config_as_model(
            config_name="base_huggingface_config",
            config_type=ConfigType.DATASET,
            config_subtype=DatasetConfigSubtype.DEFAULT,
        ).model_dump()
    }
}

GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "source_dataset", "pydantic"
    ): {"config": GroundTruthDatasetConfig(ground_truth_source="source").model_dump()}
}


GT_PARSED_DATASET_PYDANTIC_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "parsed_dataset", "pydantic"
    ): {"config": GroundTruthDatasetConfig(ground_truth_source="parsed").model_dump()}
}
