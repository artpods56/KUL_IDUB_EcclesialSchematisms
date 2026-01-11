"""
Configs for assets defined in this file lives in [[transformation_config.py]]
"""

from notarius.application.services.data.flattening import FlatteningService
from notarius.application.services.data.merging import MergingService

import random
from typing import Mapping, cast, Iterable

import pandas as pd
import dagster as dg
from dagster import (
    AssetExecutionContext,
    MetadataValue,
    AssetIn,
    TableSchema,
)
from datasets import Dataset
from pydantic import Field

from notarius.infrastructure.cache import get_image_hash
from notarius.infrastructure.persistence.storage import ImageRepository
from notarius.orchestration.constants import (
    DataSource,
    AssetLayer,
    ResourceGroup,
    Kinds,
)

import structlog

from notarius.schemas.data.dataset import (
    SchematismsDatasetItem,
)
from notarius.schemas.data.pipeline import (
    GroundTruthDataItem,
    BaseDataset,
    BaseMetaData,
    BaseDataItem,
    AlignedSchematismsDataItem,
    # Concrete subclasses for pickle compatibility
    BaseItemDataset,
    GroundTruthItemDataset,
    PredictionItemDataset,
    FlatSchematismEntryWithMetadata,
    FlatSchematismAlignedEntryWithMetadata,
    PredictionDataItem,
)
from notarius.domain.entities.schematism import SchematismPage

logger = structlog.get_logger(__name__)


def asset_factory__eval__aligned_dataframe_pandas(
    asset_name: str, ins: Mapping[str, AssetIn]
):
    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
        kinds={Kinds.PYTHON, Kinds.PANDAS},
        group_name=ResourceGroup.DATA,
        ins=ins,  # {"dataset": AssetIn("gt__aligned_dataset")},
    )
    def _asset__eval__aligned_dataframe__pandas(
        context: AssetExecutionContext,
        dataset: BaseDataset[AlignedSchematismsDataItem],
    ):
        rows = []

        rows: list[FlatSchematismAlignedEntryWithMetadata] = []

        for item in dataset.items:
            rows.extend(FlatteningService.flatten_aligned_pages(item))

        df = pd.DataFrame([row.model_dump() for row in rows])

        markdown_head = df.head(30).to_markdown()
        column_schema = TableSchema.from_name_type_dict(df.dtypes.astype(str).to_dict())

        return dg.MaterializeResult(
            value=df,
            metadata={
                "dagster/table_name": "table",
                "dagster/column_schema": column_schema,
                "dagster/row_count": len(df),
                "preview": MetadataValue.md(markdown_head or "unknown"),
            },
        )

    return _asset__eval__aligned_dataframe__pandas


eval__aligned_source_dataframe__pandas = asset_factory__eval__aligned_dataframe_pandas(
    asset_name="eval__aligned_source_dataframe__pandas",
    ins={"dataset": AssetIn("gt__aligned_source_dataset__pydantic")},
)
eval__aligned_parsed_dataframe__pandas = asset_factory__eval__aligned_dataframe_pandas(
    asset_name="eval__aligned_parsed_dataframe__pandas",
    ins={"dataset": AssetIn("gt__aligned_parsed_dataset__pydantic")},
)


def asset_factory__pred__dataframe_pandas(asset_name: str, ins: Mapping[str, AssetIn]):
    """Factory for creating prediction-only DataFrame assets (no ground truth alignment)."""

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
        kinds={Kinds.PYTHON, Kinds.PANDAS},
        group_name=ResourceGroup.DATA,
        ins=ins,
    )
    def _asset__pred__dataframe__pandas(
        context: AssetExecutionContext,
        dataset: BaseDataset[PredictionDataItem],
    ):

        rows: list[FlatSchematismEntryWithMetadata] = []

        for item in dataset.items:
            rows.extend(FlatteningService.flatten_prediction_data_item(item))

        df = pd.DataFrame([flat_entry.model_dump() for flat_entry in rows])

        column_schema = TableSchema.from_name_type_dict(df.dtypes.astype(str).to_dict())

        preview = df.head(30).to_markdown()

        return dg.MaterializeResult(
            value=df,
            metadata={
                "dagster/table_name": "predictions",
                "dagster/column_schema": column_schema,
                "dagster/row_count": len(df),
                "preview": MetadataValue.md(preview if preview else ""),
            },
        )

    return _asset__pred__dataframe__pandas


pred__parsed_dataframe__pandas = asset_factory__pred__dataframe_pandas(
    asset_name="pred__parsed_dataframe__pandas",
    ins={"dataset": AssetIn("pred__parsed_dataset__pydantic")},
)

pred__source_dataframe__pandas = asset_factory__pred__dataframe_pandas(
    asset_name="pred__source_dataframe__pandas",
    ins={"dataset": AssetIn("pred__llm_enriched_dataset__pydantic")},
)


class BaseDatasetConfig(dg.Config):
    pass


def asset_factory__base_dataset(
    asset_name: str,
    ins: Mapping[str, AssetIn],
):
    """Factory for creating base dataset assets (without ground truth)."""

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.PYDANTIC},
        ins=ins,
    )
    def _asset__base_dataset(
        context: AssetExecutionContext,
        hf_dataset: Dataset,
        config: BaseDatasetConfig,
        images_repository: dg.ResourceParam[ImageRepository],
        pdf_dataset: BaseItemDataset | None = None,
    ) -> BaseItemDataset:
        """Convert HuggingFace dataset to Pydantic BaseDataset with BaseDataItem.

        This asset creates a base dataset containing only image and metadata,
        without ground truth. It serves as the starting point for prediction
        pipelines (OCR and LMv3 enrichment).

        Args:
            context: Dagster execution context for logging and metadata
            hf_dataset: HuggingFace dataset to convert
            config: Configuration for the asset

        Returns:
            BaseDataset containing BaseDataItem instances
        """

        items: list[BaseDataItem] = []

        for i, sample in enumerate(cast(Iterable[SchematismsDatasetItem], hf_dataset)):
            # image_name = f"{sample['schematism_name']}_{sample['filename']}"

            image = sample["image"]

            image_hash = get_image_hash(image)

            if images_repository.exists(image_hash):
                image_path = images_repository.get_path(image_hash)
            else:
                image_path = images_repository.add(image, image_hash)

            metadata = BaseMetaData(
                sample_id=sample.get("sample_id", i),
                schematism_name=sample["schematism_name"],
                filename=sample["filename"],
            )
            items.append(BaseDataItem(image_path=str(image_path), metadata=metadata))

        if pdf_dataset:
            items.extend(pdf_dataset.items)

        combined_dataset = BaseItemDataset(items=items)

        context.add_output_metadata(
            {
                "all_items": MetadataValue.int(len(items)),
                "loaded_schematisms": MetadataValue.json(
                    {
                        schematism_name: len(dataset.items)
                        for schematism_name, dataset in combined_dataset.group_by_schematism()
                    }
                ),
                "random_sample": MetadataValue.json(random.choice(items).model_dump()),
            }
        )
        return combined_dataset

    return _asset__base_dataset


class GroundTruthDatasetConfig(dg.Config):
    ground_truth_source: str = Field(
        description="Source field name for ground truth data in the HuggingFace dataset"
    )


def asset_factory__ground_truth_dataset(
    asset_name: str,
    ins: Mapping[str, AssetIn],
):
    """Factory for creating ground truth dataset assets."""

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.PYDANTIC},
        ins=ins,
    )
    def _asset__ground_truth_dataset(
        context: AssetExecutionContext,
        hf_dataset: Dataset,
        config: GroundTruthDatasetConfig,
        images_repository: dg.ResourceParam[ImageRepository],
    ):
        """Convert HuggingFace dataset to Pydantic GroundTruthDataset.

        This asset creates a dataset containing image, metadata, and ground truth.
        It is used for evaluation and alignment pipelines.

        Args:
            context: Dagster execution context for logging and metadata
            hf_dataset: HuggingFace dataset to convert
            config: Configuration specifying the ground truth source field

        Returns:
            GroundTruthDataset containing GroundTruthDataItem instances
        """

        items: list[GroundTruthDataItem] = []

        for i, sample in enumerate(cast(Iterable[SchematismsDatasetItem], hf_dataset)):
            image = sample["image"]

            image_hash = get_image_hash(image)

            if images_repository.exists(image_hash):
                image_path = images_repository.get_path(image_hash)
            else:
                image_path = images_repository.add(image, image_hash)

            metadata = BaseMetaData(
                sample_id=sample.get("sample_id", i),
                schematism_name=sample["schematism_name"],
                filename=sample["filename"],
            )

            ground_truth: SchematismPage | None = sample.get(
                config.ground_truth_source, None
            )

            if ground_truth is None:
                raise ValueError(
                    f"Ground truth field '{config.ground_truth_source}' not found in sample: {sample}"
                )

            items.append(
                GroundTruthDataItem(
                    ground_truth=ground_truth,
                    image_path=str(image_path),
                    metadata=metadata,
                )
            )

        dataset = GroundTruthItemDataset(items=items)

        context.add_output_metadata(
            {
                "all_items": MetadataValue.int(len(items)),
                "loaded_schematisms": MetadataValue.json(
                    {
                        schematism_name: len(ds.items)
                        for schematism_name, ds in dataset.group_by_schematism()
                    }
                ),
                "random_sample": MetadataValue.json(random.choice(items).model_dump()),
            }
        )
        return dataset

    return _asset__ground_truth_dataset


base__dataset__pydantic = asset_factory__base_dataset(
    asset_name="base__dataset__pydantic",
    ins={
        "hf_dataset": AssetIn(key="preprocessed__hf__dataset"),
        "pdf_dataset": AssetIn(key="raw__pdf__dataset"),
    },
)

gt__source_dataset__pydantic = asset_factory__ground_truth_dataset(
    asset_name="gt__source_dataset__pydantic",
    ins={"hf_dataset": AssetIn(key="preprocessed__hf__dataset")},
)

gt__parsed_dataset__pydantic = asset_factory__ground_truth_dataset(
    asset_name="gt__parsed_dataset__pydantic",
    ins={"hf_dataset": AssetIn(key="preprocessed__hf__dataset")},
)


@dg.asset(
    name="pred__merged_ocr_lmv3_dataset__pydantic",
    key_prefix=[AssetLayer.INT, DataSource.PREDICTION],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "predictions": AssetIn(key="pred__lmv3_enriched_dataset__pydantic"),
        "ocr": AssetIn(key="pred__llm_ocr_enriched_dataset__pydantic"),
    },
)
def pred__merged_ocr_lmv3_dataset__pydantic(
    predictions: PredictionItemDataset,
    ocr: BaseItemDataset,
) -> PredictionItemDataset:
    """Merge LMv3 predictions with OCR text."""
    return MergingService().merge_predictions_with_ocr(predictions, ocr)


@dg.asset(
    key_prefix=[AssetLayer.INT, DataSource.PREDICTION],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "ground_truth": AssetIn(key="gt__parsed_dataset__pydantic"),
        "ocr": AssetIn(key="pred__llm_ocr_enriched_dataset__pydantic"),
    },
)
def pred__merged_ocr_parsed_dataset__pydantic(
    ground_truth: GroundTruthItemDataset,
    ocr: BaseItemDataset,
) -> PredictionItemDataset:
    """Merge ground truth (parsed Polish) with OCR text for source generation.

    Converts ground truth entries to prediction format to enable uniform
    downstream processing in source generation pipeline.
    """
    return MergingService().merge_ground_truth_with_ocr(ground_truth, ocr)


def asset_factory__pred__dataset__pandas(
    asset_name: str,
    ins: Mapping[str, AssetIn],
):
    """Factory for creating DataFrame assets from Pydantic prediction datasets."""

    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.PANDAS},
        ins=ins,
    )
    def _asset__pred__dataset__pandas(
        context: AssetExecutionContext,
        dataset: PredictionItemDataset,
    ) -> pd.DataFrame:
        """Convert Pydantic prediction dataset to pandas DataFrame using use case.

        Args:
            context: Dagster execution context for logging and metadata
            dataset: Prediction dataset to convert

        Returns:
            MaterializeResult with flattened DataFrame
        """

        rows: list[FlatSchematismEntryWithMetadata] = []

        for item in dataset.items:
            rows.extend(FlatteningService.flatten_prediction_data_item(item))

        df = pd.DataFrame([row.model_dump() for row in rows])

        column_schema = TableSchema.from_name_type_dict(df.dtypes.astype(str).to_dict())
        preview = df.head(30).to_markdown()

        context.add_output_metadata(
            metadata={
                "dagster/table_name": "predictions",
                "dagster/column_schema": column_schema,
                "dagster/row_count": len(rows),
                "total_items": MetadataValue.int(len(dataset.items)),
                "preview": MetadataValue.md(preview if preview else ""),
            },
        )

        return df

    return _asset__pred__dataset__pandas


pred__dataset__pandas = asset_factory__pred__dataset__pandas(
    asset_name="pred__dataset__pandas",
    ins={"dataset": AssetIn("pred__llm_enriched_dataset__pydantic")},
)
