"""Post-processing assets for data refinement and completion.

This module contains Dagster assets for post-processing operations
that perform cross-sample analysis and data completion tasks.
"""

import random
from typing import Literal

import dagster as dg
from dagster import AssetExecutionContext, MetadataValue, AssetIn

from notarius.domain.services.parser import Parser
from notarius.application.services.data.aligning import (
    Aligner,
    HungarianAligner,
    GreedyAligner,
)
from notarius_dagster.constants import (
    AssetLayer,
    ResourceGroup,
    DataSource,
    Kinds,
)
from notarius.schemas.data.pipeline import (
    BaseDataset,
    PredictionDataItem,
    GroundTruthDataItem,
    AlignedSchematismsDataItem,
    AlignedItemDataset,
)
from notarius.domain.entities.schematism import SchematismPage


class AlignmentConfig(dg.Config):
    """Configuration for entry alignment operation."""

    aligner_type: Literal["greedy", "hungarian"] = "greedy"
    threshold: float = 0.5
    position_weight: float = 0.0  # Only used for greedy aligner
    weights: dict[str, float] = {
        "deanery": 1.0,
        "parish": 2.0,
        "dedication": 1.5,
        "building_material": 0.5,
    }

    def get_aligner(self) -> Aligner:
        """Create the configured aligner instance."""
        if self.aligner_type == "hungarian":
            return HungarianAligner(weights=self.weights, threshold=self.threshold)
        return GreedyAligner(
            weights=self.weights,
            threshold=self.threshold,
            position_weight=self.position_weight,
        )


def asset_factory__gt_aligned_dataset__pydantic(
    asset_name: str, gt_dataset_asset: str, pred_dataset_asset: str
):
    @dg.asset(
        name=asset_name,
        key_prefix=[AssetLayer.FCT, DataSource.HUGGINGFACE],
        group_name=ResourceGroup.DATA,
        kinds={Kinds.PYTHON, Kinds.PYDANTIC},
        ins={
            "gt_dataset": AssetIn(key=gt_dataset_asset),
            "pred_dataset": AssetIn(key=pred_dataset_asset),
        },
    )
    def _asset__gt_aligned__dataset(
        context: AssetExecutionContext,
        gt_dataset: BaseDataset[GroundTruthDataItem],
        pred_dataset: BaseDataset[PredictionDataItem],
        config: AlignmentConfig,
    ) -> BaseDataset[AlignedSchematismsDataItem]:
        """Align ground truth entries with predictions using fuzzy matching.

        Matches ground truth and prediction datasets by sample_id, then aligns
        corresponding entries within each matched pair using the configured aligner.

        Args:
            context: Dagster execution context for logging and metadata
            gt_dataset: Ground truth dataset with SchematismPage entries
            pred_dataset: Predictions dataset with SchematismPage entries
            config: Configuration for alignment thresholds and weights

        Returns:
            Dataset with aligned ground truth and prediction entries
        """
        gt_by_id = {
            item.metadata.sample_id: item for item in gt_dataset.items if item.metadata
        }
        pred_by_id = {
            item.metadata.sample_id: item
            for item in pred_dataset.items
            if item.metadata
        }

        aligner = config.get_aligner()
        aligned_items: list[AlignedSchematismsDataItem] = []
        total_entries = 0

        common_ids = set(gt_by_id.keys()) & set(pred_by_id.keys())

        for sample_id in sorted(common_ids):
            gt_item = gt_by_id[sample_id]
            pred_item = pred_by_id[sample_id]

            if not gt_item.ground_truth or not pred_item.predictions:
                context.log.warning(
                    f"Missing data for sample {sample_id}, skipping alignment"
                )
                continue

            aligned_gt, aligned_pred = aligner.align_entries(
                gt_item.ground_truth.entries,
                pred_item.predictions.entries,
            )

            aligned_items.append(
                AlignedSchematismsDataItem(
                    image_path=gt_item.image_path,
                    text=pred_item.text,
                    metadata=gt_item.metadata,
                    aligned_schematism_pages=(
                        SchematismPage(
                            page_number=gt_item.ground_truth.page_number,
                            entries=list(aligned_gt),
                        ),
                        SchematismPage(
                            page_number=pred_item.predictions.page_number,
                            entries=list(aligned_pred),
                        ),
                    ),
                )
            )
            total_entries += len(aligned_gt)

        context.log.info(
            f"Aligned {len(aligned_items)} items with {total_entries} entries from {len(common_ids)} matching sample pairs",
        )

        if not aligned_items:
            context.log.warning("No items were successfully aligned")

        random_sample = (
            aligned_items[random.randint(0, len(aligned_items) - 1)]
            if aligned_items
            else None
        )

        context.add_output_metadata(
            {
                "gt_dataset_size": MetadataValue.int(len(gt_dataset.items)),
                "pred_dataset_size": MetadataValue.int(len(pred_dataset.items)),
                "common_samples": MetadataValue.int(len(common_ids)),
                "aligned_items": MetadataValue.int(len(aligned_items)),
                "total_aligned_entries": MetadataValue.int(total_entries),
                "aligner_type": MetadataValue.text(config.aligner_type),
                "alignment_threshold": MetadataValue.float(config.threshold),
                "random_sample": MetadataValue.json(
                    {
                        k: v
                        for k, v in random_sample.model_dump().items()
                        if k != "image"
                    }
                    if random_sample
                    else {}
                ),
            }
        )

        return AlignedItemDataset(items=aligned_items)

    return _asset__gt_aligned__dataset


gt__aligned_parsed_dataset__pydantic = asset_factory__gt_aligned_dataset__pydantic(
    asset_name="gt__aligned_parsed_dataset__pydantic",
    gt_dataset_asset="gt__parsed_dataset__pydantic",
    pred_dataset_asset="pred__parsed_dataset__pydantic",
)
gt__aligned_source_dataset__pydantic = asset_factory__gt_aligned_dataset__pydantic(
    asset_name="gt__aligned_source_dataset__pydantic",
    gt_dataset_asset="gt__source_dataset__pydantic",
    pred_dataset_asset="pred__llm_enriched_dataset__pydantic",
)


class ParsingConfig(dg.Config):
    """Configuration for parsing operation."""

    enable: bool = True


@dg.asset(
    key_prefix=[AssetLayer.FCT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "dataset": AssetIn(key="pred__llm_enriched_dataset__pydantic"),
    },
)
def pred__parsed_dataset__pydantic(
    context: AssetExecutionContext,
    dataset: BaseDataset[PredictionDataItem],
    parser: dg.ResourceParam[Parser],
    config: ParsingConfig,
) -> BaseDataset[PredictionDataItem]:
    """Parse and normalize LLM predictions using the translation parser.

    This asset processes each item's LLM predictions through the parser
    to normalize dedications, building materials, and deaneries using
    fuzzy matching against known mappings.

    Args:
        context: Dagster execution context for logging and metadata
        dataset: Dataset containing prediction items to process
        parser: Parser instance for translating/normalizing entries
        config: Configuration for parsing operation

    Returns:
        Updated dataset with parsed predictions
    """

    parsed_count = 0
    total_items = 0

    if not config.enable:
        return dataset

    for item in dataset.items:
        if item.predictions is not None:
            # Parse the predictions using the parser
            parsed_page = parser.parse_page(page_data=item.predictions)
            item.predictions = parsed_page
            parsed_count += 1
        else:
            context.log.warning("No predictions found in item. Skipping parsing.")

        total_items += 1

    context.log.info(f"Parsed {parsed_count} items out of {total_items} total items")

    random_sample = dataset.items[random.randint(0, len(dataset.items) - 1)]

    context.add_output_metadata(
        {
            "dataset_size": MetadataValue.int(len(dataset.items)),
            "items_parsed": MetadataValue.int(parsed_count),
            "parse_rate": MetadataValue.float(
                parsed_count / total_items if total_items > 0 else 0.0
            ),
            "random_sample": MetadataValue.json(
                {k: v for k, v in random_sample.model_dump().items() if k != "image"}
                if random_sample
                else {}
            ),
        }
    )

    return dataset
