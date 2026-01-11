from notarius.orchestration.assets.load.export import (
    WandBDataFrameExport,
    PredictionDataFrameExport,
)
from notarius.orchestration.constants import AssetLayer, DataSource
from notarius.orchestration.utils import AssetKeyHelper

"""
Asset: [[export.py#eval__excel_export_parsed_dataframe__pandas]]
Defined in: [[src/orchestration/assets/load/export.py]]
Resolves to: mrt__huggingface__eval__excel_export_parsed_dataframe__pandas
"""
EVAL__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "eval",
        "excel_export_parsed_dataframe",
        "pandas",
    ): {"config": {}}
}

"""
Asset: [[export.py#eval__excel_export_source_dataframe__pandas]]
Defined in: [[src/orchestration/assets/load/export.py]]
Resolves to: mrt__huggingface__eval__excel_export_source_dataframe__pandas
"""
EVAL__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "eval",
        "excel_export_source_dataframe",
        "pandas",
    ): {"config": {}}
}


"""
Asset: [[export.py#eval__wandb_export_dataframe__pandas]]
Defined in: [[src/orchestration/assets/load/export.py]]
Resolves to: mrt__huggingface__eval__wandb_export_dataframe__pandas
"""
EVAL__WANDB_EXPORT_DATAFRAME__PANDAS = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "eval",
        "wandb_export_dataframe",
        "pandas",
    ): {
        "config": WandBDataFrameExport(
            table_name="eval_aligned_dataframe", group_by_key="schematism_name"
        ).model_dump()
    }
}


"""
Asset: [[export.py#pred__excel_export_parsed_dataframe__pandas]]
Defined in: [[src/orchestration/assets/load/export.py]]
Resolves to: mrt__huggingface__pred__excel_export_parsed_dataframe__pandas
"""
PRED__EXCEL_EXPORT_PARSED_DATAFRAME__PANDAS = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "pred",
        "excel_export_parsed_dataframe",
        "pandas",
    ): {
        "config": PredictionDataFrameExport(
            file_name="parsed_predictions.xlsx"
        ).model_dump()
    }
}


"""
Asset: [[export.py#pred__excel_export_source_dataframe__pandas]]
Defined in: [[src/orchestration/assets/load/export.py]]
Resolves to: mrt__huggingface__pred__excel_export_source_dataframe__pandas
"""
PRED__EXCEL_EXPORT_SOURCE_DATAFRAME__PANDAS = {
    AssetKeyHelper.build_prefixed_key(
        AssetLayer.MRT,
        DataSource.HUGGINGFACE,
        "pred",
        "excel_export_source_dataframe",
        "pandas",
    ): {
        "config": PredictionDataFrameExport(
            file_name="source_predictions.xlsx"
        ).model_dump()
    }
}
