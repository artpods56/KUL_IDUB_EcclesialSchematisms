import dagster as dg

from notarius_dagster.assets.transform.postprocess import (
    pred__parsed_dataset__pydantic,
)
from notarius_dagster.assets.transform.transform import (
    pred__parsed_dataframe__pandas,
    pred__source_dataframe__pandas,
)
from notarius_dagster.assets.load.export import (
    pred__excel_export_parsed_dataframe__pandas,
    pred__excel_export_source_dataframe__pandas,
)
from notarius_dagster.configs.postprocessing_config import (
    PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
)
from notarius_dagster.configs.exporting_config import (
    PRED__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
    PRED__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
)


prediction_export_assets = [
    # Parsing
    pred__parsed_dataset__pydantic,
    # DataFrames
    pred__parsed_dataframe__pandas,
    pred__source_dataframe__pandas,
    # Excel exports
    pred__excel_export_parsed_dataframe__pandas,
    pred__excel_export_source_dataframe__pandas,
]

prediction_export_job = dg.define_asset_job(
    name="prediction_export_pipeline",
    selection=dg.AssetSelection.assets(*prediction_export_assets),
    config={
        "ops": {
            **PRED__PARSED_DATASET__PYDANTIC__OP_CONFIG,
            **PRED__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS,
            **PRED__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS,
        }
    },
)
