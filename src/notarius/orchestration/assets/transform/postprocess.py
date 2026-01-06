"""Post-processing assets for data refinement and completion.

This module contains Dagster assets for post-processing operations
that perform cross-sample analysis and data completion tasks.
"""

import random
from typing import Optional, Literal

import dagster as dg
from dagster import AssetExecutionContext, MetadataValue, AssetIn

from notarius.domain.services.parser import Parser
from notarius.domain.services.aligner import JSONAligner, FlatHungarianAligner
from notarius.orchestration.constants import (
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
    # Concrete subclass for pickle compatibility
    AlignedItemDataset,
)
from notarius.domain.entities.schematism import SchematismEntry, SchematismPage


class DeaneryFillingConfig(dg.Config):
    """Configuration for deanery filling operation."""

    enabled: bool = True


@dg.asset(
    key_prefix=[AssetLayer.FCT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "dataset": AssetIn(key="pred__llm_enriched_dataset__pydantic"),
    },
)
def pred__deanery_filled_dataset__pydantic(
    context: AssetExecutionContext,
    dataset: BaseDataset[PredictionDataItem],
    config: DeaneryFillingConfig,
) -> BaseDataset[PredictionDataItem]:
    """Fill missing deanery values across dataset entries.

    This asset performs cross-sample deanery filling by propagating
    deanery values forward to entries that don't have them. It processes
    all entries sequentially and maintains the last seen deanery value.

    Args:
        context: Dagster execution context for logging and metadata
        dataset: Dataset containing prediction items to process
        config: Configuration for deanery filling operation

    Returns:
        Updated dataset with filled deanery values
    """

    if not config.enabled:
        return dataset

    all_entries = []

    # Collect all entries from predictions across all items
    for item in dataset.items:
        if item.predictions is not None:
            all_entries.extend(item.predictions.entries)
        else:
            context.log.warning(
                "No predictions found in item. Skipping for deanery filling."
            )

    # Process all entries sequentially, filling missing deaneries
    current_deanery: Optional[str] = None
    filled_count = 0

    for entry in all_entries:
        if entry.deanery and not current_deanery:
            # First deanery encountered
            current_deanery = entry.deanery
        elif not entry.deanery and current_deanery:
            # Fill missing deanery
            entry.deanery = current_deanery
            filled_count += 1
        elif entry.deanery != current_deanery:
            # New deanery encountered
            current_deanery = entry.deanery
        # else: deanery matches current, no action needed

    context.log.info(
        f"Filled {filled_count} deanery values across {len(all_entries)} entries "
        f"from {len(dataset.items)} items"
    )

    random_sample = dataset.items[random.randint(0, len(dataset.items) - 1)]

    context.add_output_metadata(
        {
            "dataset_size": MetadataValue.int(len(dataset.items)),
            "total_entries": MetadataValue.int(len(all_entries)),
            "deaneries_filled": MetadataValue.int(filled_count),
            "random_sample": MetadataValue.json(
                {k: v for k, v in random_sample.model_dump().items() if k != "image"}
                if random_sample
                else {}
            ),
        }
    )

    return dataset


class JSONAlignmentConfig(dg.Config):
    """Configuration for JSON alignment operation."""

    aligner_type: Literal["greedy", "hungarian"] = "greedy"
    threshold: float = 0.5
    weights: dict[str, float] = {
        "deanery": 1.0,
        "parish": 2.0,
        "dedication": 1.5,
        "building_material": 0.5,
    }


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
        config: JSONAlignmentConfig,
    ) -> BaseDataset[AlignedSchematismsDataItem]:
        """Align ground truth entries with parsed predictions using fuzzy matching.

        This asset matches ground truth and prediction datasets by sample_id,
        then uses the JSONAligner to align corresponding entries within each matched pair.
        The result preserves both the original ground truth and the aligned entries
        for downstream metric calculations.

        Args:
            context: Dagster execution context for logging and metadata
            gt_dataset: Ground truth dataset with SchematismPage entries
            pred_dataset: Parsed predictions dataset with SchematismPage entries
            config: Configuration for alignment thresholds and weights

        Returns:
            Dataset with aligned ground truth and prediction entries
        """
        gt_by_id = {
            item.metadata.sample_id: item for item in gt_dataset.items if item.metadata
        }
        parsed_by_id = {
            item.metadata.sample_id: item
            for item in pred_dataset.items
            if item.metadata
        }

        aligned_items = []
        aligned_count = 0
        total_entries = 0

        # Get all sample IDs that exist in both datasets
        common_sample_ids = set(gt_by_id.keys()) & set(parsed_by_id.keys())

        for sample_id in sorted(common_sample_ids):
            gt_item = gt_by_id[sample_id]
            parsed_item = parsed_by_id[sample_id]

            if not gt_item.ground_truth or not parsed_item.predictions:
                context.log.warning(
                    f"Missing data for sample {sample_id}, skipping alignment"
                )
                continue

            # Align the entries using the configured aligner
            gt_entries = [entry.model_dump() for entry in gt_item.ground_truth.entries]
            pred_entries = [
                entry.model_dump() for entry in parsed_item.predictions.entries
            ]

            if config.aligner_type == "hungarian":
                aligner = FlatHungarianAligner(
                    weights_mapping=config.weights, threshold=config.threshold
                )
                aligned_gt_entries, aligned_pred_entries = aligner.align_entries(
                    gt_entries, pred_entries
                )
            else:
                aligner = JSONAligner(
                    weights_mapping=config.weights, threshold=config.threshold
                )
                aligned_gt_entries, aligned_pred_entries = aligner.align_entries(
                    {"entries": gt_entries},
                    {"entries": pred_entries},
                )

            # Convert aligned entries back to SchematismEntry objects
            aligned_gt_page = SchematismPage(
                page_number=gt_item.ground_truth.page_number,
                entries=[SchematismEntry(**entry) for entry in aligned_gt_entries],
            )
            aligned_pred_page = SchematismPage(
                page_number=parsed_item.predictions.page_number,
                entries=[SchematismEntry(**entry) for entry in aligned_pred_entries],
            )

            # Create the aligned item
            aligned_item = AlignedSchematismsDataItem(
                image_path=gt_item.image_path,  # Use ground truth image_path as primary
                text=parsed_item.text,  # Use parsed text (likely has OCR)
                metadata=gt_item.metadata,  # Use ground truth metadata
                aligned_schematism_pages=(
                    aligned_gt_page,
                    aligned_pred_page,
                ),  # Aligned tuple
            )

            aligned_items.append(aligned_item)
            aligned_count += 1
            total_entries += len(aligned_gt_entries)

        context.log.info(
            f"Aligned {aligned_count} items with {total_entries} total aligned entries "
            f"from {len(common_sample_ids)} matching sample pairs"
        )

        if aligned_count == 0:
            context.log.warning("No items were successfully aligned")

        random_sample = (
            aligned_items[random.randint(0, len(aligned_items) - 1)]
            if aligned_items
            else None
        )

        context.add_output_metadata(
            {
                "gt_dataset_size": MetadataValue.int(len(gt_dataset.items)),
                "parsed_dataset_size": MetadataValue.int(len(pred_dataset.items)),
                "common_samples": MetadataValue.int(len(common_sample_ids)),
                "aligned_items": MetadataValue.int(aligned_count),
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

        # Use concrete subclass for pickle compatibility
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
