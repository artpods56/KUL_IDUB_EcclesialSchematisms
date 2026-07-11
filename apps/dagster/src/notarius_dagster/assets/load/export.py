import dataclasses
import re
from datetime import datetime
from pathlib import Path
from typing import Mapping

import dagster as dg
import pandas as pd
from PIL import Image
from dagster import AssetExecutionContext, AssetIn, MetadataValue


# Regex pattern for illegal Excel characters (control characters except tab, newline, carriage return)
ILLEGAL_EXCEL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """Remove illegal characters from string columns for Excel export."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: ILLEGAL_EXCEL_CHARS_RE.sub("", x) if isinstance(x, str) else x
            )
    return df


from notarius.application import ports
from notarius.application.services.scoring import (
    SimilarityEvaluationService,
    ClassificationEvaluationService,
    NormalizedLevenshteinDistanceScorer,
    ExactMatchScorer,
    SimilarityMetrics,
    ClassificationMetrics,
    ExactMatchStyling,
    GradientStyling,
    CellComparisonStyler,
)
from notarius.application.use_cases.export import (
    ExportDataFrameToWandB,
    WandBExportRequest,
    JsonExportUseCase,
    JsonExportRequest,
)
from notarius.application.use_cases.export.wandb_dataframe_export import (
    DataFrameExportConfig,
)
from notarius.infrastructure.persistence.storage import ImageRepository
from notarius_dagster.constants import (
    AssetLayer,
    DataSource,
    ResourceGroup,
    Kinds,
)
from notarius_dagster.resources.base import (
    ExcelWriterResource,
)
from notarius.schemas.data.pipeline import BaseDataset, BaseDataItem, PredictionDataItem
from notarius.shared.constants import OUTPUTS_DIR


DEFAULT_EVAL_COLUMNS: list[tuple[str, str]] = [
    ("deanery_a", "deanery_b"),
    ("parish_a", "parish_b"),
    ("dedication_a", "dedication_b"),
    ("building_material_a", "building_material_b"),
]


class ParsedDataFrameExportConfig(dg.Config):
    """Configuration for parsed dataset export with classification metrics."""

    file_name: str = "parsed_evaluation.xlsx"
    group_by_key: str = "schematism_name"
    include_index: bool = True
    include_header: bool = True
    scorer_type: str = "levenshtein"  # "levenshtein" or "exact_match"
    match_threshold: float = 1.0


class SourceDataFrameExportConfig(dg.Config):
    """Configuration for source dataset export with similarity metrics."""

    file_name: str = "source_evaluation.xlsx"
    group_by_key: str = "schematism_name"
    include_index: bool = True
    include_header: bool = True
    scorer_type: str = "levenshtein"


@dg.asset(
    name="eval__excel_export_parsed_dataframe__pandas",
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.EXCEL},
    ins={"dataframe": AssetIn(key="eval__aligned_parsed_dataframe__pandas")},
)
def eval__excel_export_parsed_dataframe__pandas(
    context: AssetExecutionContext,
    dataframe: pd.DataFrame,
    config: ParsedDataFrameExportConfig,
    excel_writer: ExcelWriterResource,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_file_name = f"{timestamp}_{config.file_name}"

    scorer = (
        ExactMatchScorer()
        if config.scorer_type == "exact_match"
        else NormalizedLevenshteinDistanceScorer()
    )

    service = ClassificationEvaluationService(scorer, threshold=config.match_threshold)
    styler = CellComparisonStyler(scorer=scorer, styling=ExactMatchStyling())

    with excel_writer.get_writer(full_file_name) as writer:
        for key, group in dataframe.groupby(config.group_by_key):
            # Sanitize data to remove illegal Excel characters
            group = sanitize_for_excel(group)

            aggregated_group_metrics = service.evaluate(group, DEFAULT_EVAL_COLUMNS)

            context.add_output_metadata(
                {
                    f"{str(key)}": dg.MetadataValue.json(
                        {
                            m.field: {
                                "f1": m.f1,
                                "accuracy": m.accuracy,
                                "precision": m.precision,
                                "recall": m.recall,
                            }
                            for m in aggregated_group_metrics.metrics
                        }
                    )
                }
            )

            metrics_sheet_name = f"{key}_Metrics"[:31]
            aggregated_group_metrics.to_dataframe().to_excel(
                writer, sheet_name=metrics_sheet_name, index=False
            )

            evaluation_sheet_name = f"{key}_Evaluation"[:31]
            styled_group = styler.style(group, columns_to_compare=DEFAULT_EVAL_COLUMNS)
            styled_group.to_excel(
                writer,
                sheet_name=evaluation_sheet_name,
                index=config.include_index,
                header=config.include_header,
            )

            context.log.info(
                f"Wrote sheet '{evaluation_sheet_name}' with {len(group)} rows"
            )

    return str(full_file_name)


@dg.asset(
    name="eval__excel_export_source_dataframe__pandas",
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.EXCEL},
    ins={"dataframe": AssetIn(key="eval__aligned_source_dataframe__pandas")},
)
def eval__excel_export_source_dataframe__pandas(
    context: AssetExecutionContext,
    dataframe: pd.DataFrame,
    config: SourceDataFrameExportConfig,
    excel_writer: ExcelWriterResource,
):
    """Export source dataset with similarity metrics (avg/min/max similarity)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_file_name = f"{timestamp}_{config.file_name}"

    scorer = (
        ExactMatchScorer()
        if config.scorer_type == "exact_match"
        else NormalizedLevenshteinDistanceScorer()
    )

    service = SimilarityEvaluationService(scorer)
    styler = CellComparisonStyler(scorer=scorer, styling=GradientStyling())

    with excel_writer.get_writer(full_file_name) as writer:
        for key, group in dataframe.groupby(config.group_by_key):
            # Sanitize data to remove illegal Excel characters
            group = sanitize_for_excel(group)

            aggregated_group_metrics = service.evaluate(group, DEFAULT_EVAL_COLUMNS)

            context.add_output_metadata(
                {
                    f"{str(key)}": dg.MetadataValue.json(
                        {
                            m.field: {
                                "average_similarity": m.average_similarity,
                                "min_similarity": m.min_similarity,
                                "max_similarity": m.max_similarity,
                            }
                            for m in aggregated_group_metrics.metrics
                        }
                    )
                }
            )

            metrics_sheet_name = f"{key}_Metrics"[:31]
            aggregated_group_metrics.to_dataframe().to_excel(
                writer, sheet_name=metrics_sheet_name, index=False
            )

            evaluation_sheet_name = f"{key}_Evaluation"[:31]
            styled_group = styler.style(group, columns_to_compare=DEFAULT_EVAL_COLUMNS)
            styled_group.to_excel(
                writer,
                sheet_name=evaluation_sheet_name,
                index=config.include_index,
                header=config.include_header,
            )

            context.log.info(
                f"Wrote sheet '{evaluation_sheet_name}' with {len(group)} rows"
            )

    return str(full_file_name)


class PredictionDataFrameExport(dg.Config):
    """Configuration for prediction-only DataFrame export (no ground truth comparison)."""

    file_name: str = "predictions.xlsx"
    group_by_key: str = "schematism_name"
    include_index: bool = False
    include_header: bool = True


def asset_factory__pred__excel_export_dataframe__pandas(
    asset_name: str, ins: Mapping[str, AssetIn]
):
    """Factory for prediction-only Excel export (no fuzzy comparison styling)."""

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.EXCEL},
        ins=ins,
    )
    def _asset__pred__excel_export_dataframe__pandas(
        context: AssetExecutionContext,
        dataframe: pd.DataFrame,
        config: PredictionDataFrameExport,
        excel_writer: ExcelWriterResource,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_file_name = f"{timestamp}_{config.file_name}"

        sheets_written = []

        with excel_writer.get_writer(full_file_name) as writer:
            for key, group in dataframe.groupby(config.group_by_key):
                # Sanitize data to remove illegal Excel characters
                group = sanitize_for_excel(group)

                sheet_name = str(key)[:31]
                group.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=config.include_index,
                    header=config.include_header,
                )
                sheets_written.append(sheet_name)
                context.log.info(f"Wrote sheet '{sheet_name}' with {len(group)} rows")

        context.add_output_metadata(
            {
                "file_name": MetadataValue.text(str(full_file_name)),
                "sheets_written": MetadataValue.json(sheets_written),
                "total_rows": MetadataValue.int(len(dataframe)),
            }
        )

        return str(full_file_name)

    return _asset__pred__excel_export_dataframe__pandas


pred__excel_export_parsed_dataframe__pandas = (
    asset_factory__pred__excel_export_dataframe__pandas(
        asset_name="pred__excel_export_parsed_dataframe__pandas",
        ins={"dataframe": AssetIn(key="pred__parsed_dataframe__pandas")},
    )
)

pred__excel_export_source_dataframe__pandas = (
    asset_factory__pred__excel_export_dataframe__pandas(
        asset_name="pred__excel_export_source_dataframe__pandas",
        ins={"dataframe": AssetIn(key="pred__source_dataframe__pandas")},
    )
)

pred__excel_export_dataframe__pandas = (
    asset_factory__pred__excel_export_dataframe__pandas(
        asset_name="pred__excel_export_dataframe__pandas",
        ins={"dataframe": AssetIn(key="pred__dataset__pandas")},
    )
)


class WandBRunResource(
    dg.ConfigurableResource  # pyright: ignore[reportMissingTypeArgument]
):
    run_name: str
    project_name: str
    mode: str = "online"

    def get_wandb_run(self):
        import wandb

        return wandb.init(
            project=self.project_name,
            name=self.run_name,
            mode=self.mode,  # pyright: ignore[reportArgumentType]
        )


class WandBDataFrameExport(dg.Config):
    parsed_table_name: str = "eval_parsed_dataframe"
    source_table_name: str = "eval_source_dataframe"
    group_by_key: str | None = None
    include_images: bool = True
    sample_id_column: str = "sample_id"


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.WANDB},
    ins={
        "parsed_dataframe": AssetIn(key="eval__aligned_parsed_dataframe__pandas"),
        "source_dataframe": AssetIn(key="eval__aligned_source_dataframe__pandas"),
        "pydantic_dataset": AssetIn(key="base__dataset__pydantic"),
    },
)
async def eval__wandb_export_dataframe__pandas(
    context: AssetExecutionContext,
    parsed_dataframe: pd.DataFrame,
    source_dataframe: pd.DataFrame,
    pydantic_dataset: BaseDataset[BaseDataItem],
    config: WandBDataFrameExport,
    wandb_run: WandBRunResource,
    images_repository: dg.ResourceParam[ImageRepository],
):
    """Export parsed and source dataframes to Weights & Biases as tables."""

    def _build_sample_id_to_image(
        dataset: BaseDataset[BaseDataItem],
    ) -> dict[str, Image.Image]:
        mapping = {}
        for item in dataset.items:
            if item.image_path and item.metadata:
                mapping[str(item.metadata.sample_id)] = images_repository.get(
                    Path(item.image_path)
                )
        return mapping

    run = wandb_run.get_wandb_run()

    # Build image mapping
    sample_id_to_image = _build_sample_id_to_image(pydantic_dataset)

    # Create use case
    use_case = ExportDataFrameToWandB(wandb_run=run)

    # Configure exports for both dataframes
    request = WandBExportRequest(
        exports=[
            DataFrameExportConfig(
                dataframe=parsed_dataframe,
                table_name=config.parsed_table_name,
                group_by_key=config.group_by_key,
                include_images=config.include_images,
                sample_id_column=config.sample_id_column,
            ),
            DataFrameExportConfig(
                dataframe=source_dataframe,
                table_name=config.source_table_name,
                group_by_key=config.group_by_key,
                include_images=config.include_images,
                sample_id_column=config.sample_id_column,
            ),
        ],
        sample_id_to_image=sample_id_to_image,
    )

    # Execute use case
    response = await use_case.execute(request)

    # Add metadata
    context.add_output_metadata(
        {
            "tables_logged": MetadataValue.json(response.tables_logged),
            "total_rows": MetadataValue.int(response.total_rows),
        }
    )

    context.log.info(
        f"Logged {len(response.tables_logged)} tables to W&B with {response.total_rows} total rows"
    )


class PredsSourceExportConfig(dg.Config):
    """Configuration for source dataset JSON export."""

    filename_prefix: str = "source_predictions"
    output_dir: str = str(OUTPUTS_DIR / "json_predictions")
    group_by_schematism: bool = True
    pretty_print: bool = True


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.MIXED],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.JSON},
    ins={
        "source_dataset": AssetIn(key="pred__llm_enriched_dataset__pydantic"),
    },
)
def pred__export_llm_enriched_dataset__json(
    context: AssetExecutionContext,
    source_dataset: BaseDataset[PredictionDataItem],
    config: PredsSourceExportConfig,
    file_storage: dg.ResourceParam[ports.FileStorage],
) -> dict[str, Path]:
    """Export generated source dataset to JSON files for manual review."""
    use_case = JsonExportUseCase(storage=file_storage)
    request = JsonExportRequest(
        dataset=source_dataset,
        output_dir=Path(config.output_dir),
        group_by_schematism=config.group_by_schematism,
        pretty_print=config.pretty_print,
        filename_prefix=config.filename_prefix,
    )
    response = use_case.execute(request)

    context.add_output_metadata(
        {
            "output_dir": MetadataValue.path(str(config.output_dir)),
            "files_created": MetadataValue.int(len(response.output_files)),
            "file_paths": MetadataValue.json(
                {k: str(v) for k, v in response.output_files.items()}
            ),
            "total_records": MetadataValue.int(len(source_dataset.items)),
        }
    )

    return response.output_files
