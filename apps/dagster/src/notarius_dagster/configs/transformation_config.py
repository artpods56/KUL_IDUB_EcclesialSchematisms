from notarius_dagster.constants import DataSource, AssetLayer
from notarius_dagster.utils import AssetKeyHelper

"""
Asset: [[preprocess.py#preprocessed__hf__dataset]]
Defined in: [[src/orchestration/assets/transform/preprocess.py]]
Resolves to: int__huggingface__preprocessed__hf__dataset
"""
PREPROCESSED__HF__DATASET_OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.INT, DataSource.HUGGINGFACE, "preprocessed", "hf", "dataset"
    ): {"config": {}}
}

# Backwards compatibility alias
FILTERED__HF__DATASET_OP_CONFIG = PREPROCESSED__HF__DATASET_OP_CONFIG

#
# """
# Asset: [[transform.py#base__dataset__pydantic]]
# Defined in: [[src/orchestration/assets/transform/transform.py]]
# Resolves to: int__huggingface__base__dataset__pydantic
# """
# BASE__DATASET__PYDANTIC_OP_CONFIG = {
#     AssetKeyHelper.build_prefixed_key(
#         AssetLayer.INT, DataSource.HUGGINGFACE, "base", "dataset", "pydantic"
#     ): {"config": DatasetMappingConfig().model_dump()}
# }
#
#
# """
# Asset: [[transform.py#gt__source_dataset__pydantic]]
# Defined in: [[src/orchestration/assets/transform/transform.py]]
# Resolves to: int__huggingface__gt__source_dataset__pydantic
# """
# GT_SOURCE_DATASET_PYDANTIC_OP_CONFIG = {
#     AssetKeyHelper.build_prefixed_key(
#         AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "source_dataset", "pydantic"
#     ): {"config": DatasetMappingConfig(ground_truth_source="source").model_dump()}
# }
#
# """
# Asset: [[transform.py#gt__parsed_dataset__pydantic]]
# Defined in: [[src/orchestration/assets/transform/transform.py]]
# Resolves to: int__huggingface__gt__parsed_dataset__pydantic
# """
# GT_PARSED_DATASET_PYDANTIC_OP_CONFIG = {
#     AssetKeyHelper.build_prefixed_key(
#         AssetLayer.INT, DataSource.HUGGINGFACE, "gt", "parsed_dataset", "pydantic"
#     ): {"config": DatasetMappingConfig(ground_truth_source="parsed").model_dump()}
# }


"""
Asset: [[transform.py#eval__aligned_source_dataframe__pandas]]
Defined in: [[src/orchestration/assets/transform/transform.py]]
Resolves to: mrt__huggingface__eval__aligned_source_dataframe__pandas
"""
EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "eval",
        "aligned_source_dataframe",
        "pandas",
    ): {"config": {}}
}


"""
Asset: [[transform.py#eval__aligned_parsed_dataframe__pandas]]
Defined in: [[src/orchestration/assets/transform/transform.py]]
Resolves to: mrt__huggingface__eval__aligned_parsed_dataframe__pandas
"""
EVAL__ALIGNED_PARSED_DATAFRAME__PANDAS__OP_CONFIG = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "eval",
        "aligned_parsed_dataframe",
        "pandas",
    ): {"config": {}}
}
