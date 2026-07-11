import dagster as dg

from notarius_dagster.assets.load.export import (
    eval__excel_export_parsed_dataframe__pandas,
    eval__excel_export_source_dataframe__pandas,
    eval__wandb_export_dataframe__pandas,
)
from notarius_dagster.assets.transform.postprocess import (
    gt__aligned_source_dataset__pydantic,
    gt__aligned_parsed_dataset__pydantic,
)
from notarius_dagster.assets.transform.transform import (
    eval__aligned_source_dataframe__pandas,
    eval__aligned_parsed_dataframe__pandas,
)
from notarius_dagster.configs.exporting_config import (
    EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
    EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
    EVAL__WANDB_EXPORT_DATAFRAME__PANDAS,
)

exporting_assets = [
    gt__aligned_source_dataset__pydantic,
    gt__aligned_parsed_dataset__pydantic,
    eval__aligned_source_dataframe__pandas,
    eval__aligned_parsed_dataframe__pandas,
    eval__excel_export_parsed_dataframe__pandas,
    eval__excel_export_source_dataframe__pandas,
    eval__wandb_export_dataframe__pandas,
]
exporting_job = dg.define_asset_job(
    name="exporting_pipeline",
    selection=dg.AssetSelection.assets(*exporting_assets),
    config={
        "ops": {
            **EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
            **EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
            **EVAL__WANDB_EXPORT_DATAFRAME__PANDAS,
        }
    },
)
