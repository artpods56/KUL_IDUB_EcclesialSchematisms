import dagster as dg

from notarius_dagster.assets.transform.postprocess import (
    pred__parsed_dataset__pydantic,
    gt__aligned_parsed_dataset__pydantic,
    gt__aligned_source_dataset__pydantic,
)
from notarius_dagster.assets.transform.transform import (
    eval__aligned_source_dataframe__pandas,
    eval__aligned_parsed_dataframe__pandas,
    pred__merged_ocr_lmv3_dataset__pydantic,
    pred__merged_ocr_parsed_dataset__pydantic,
)
from notarius_dagster.configs.postprocessing_config import (
    PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
    GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
)
from notarius_dagster.configs.transformation_config import (
    EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
)

postprocessing_assets = [
    pred__merged_ocr_lmv3_dataset__pydantic,
    pred__merged_ocr_parsed_dataset__pydantic,
    pred__parsed_dataset__pydantic,
    gt__aligned_parsed_dataset__pydantic,
    gt__aligned_source_dataset__pydantic,
    eval__aligned_source_dataframe__pandas,
    eval__aligned_parsed_dataframe__pandas,
]
postprocessing_job = dg.define_asset_job(
    name="postprocessing_pipeline",
    selection=dg.AssetSelection.assets(*postprocessing_assets),
    config={
        "ops": {
            # asset refs
            **PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
            **EVAL__ALIGNED_SOURCE_DATAFRAME__PANDAS__OP_CONFIG,
            **GT__ALIGNED_SOURCE_DATASET__PYDANTIC__OP_CONFIG,
            **GT__ALIGNED_PARSED_DATASET__PYDANTIC__OP_CONFIG,
        }
    },
)
