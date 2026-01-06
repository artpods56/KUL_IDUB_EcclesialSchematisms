"""Assets for generating source (Latin) dataset from parsed (Polish) ground truth."""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import cast

import dagster as dg
from dagster import AssetExecutionContext, AssetIn, MetadataValue

from notarius.application.services import (
    DatasetProcessor,
    ItemProcessor,
    get_context_strategy,
    ContextStrategySelection,
    SlidingWindowStrategy,
)
from notarius.application.services.message_builder import Jinja2MessageBuilder
from notarius.application.services.processors.item_processor import (
    StandardRequestHandler,
    PredictionsRefinementResponseHandler,
)
from notarius.application.use_cases.inference import (
    GenerateSourceDataset,
    GenerateSourceDatasetRequest,
    SOURCE_GENERATION_CONTEXT_PROVIDERS,
)
from notarius.domain.entities.schematism import SchematismPage
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.persistence.storage import ImageRepository
from notarius.orchestration.constants import (
    AssetLayer,
    DataSource,
    ResourceGroup,
    Kinds,
)
from notarius.orchestration.resources.base import LLMEngineResource
from notarius.schemas.data.pipeline import (
    BaseDataItem,
    BaseDataset,
    PredictionDataItem,
    PredictionItemDataset,
)
from notarius.shared.constants import OUTPUTS_DIR


class OcrExportConfig(dg.Config):
    """Configuration for OCR dataset JSON export."""

    output_dir: str = str(OUTPUTS_DIR / "llm_ocr")
    group_by_schematism: bool = True
    pretty_print: bool = True


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.JSON},
    ins={
        "ocr_dataset": AssetIn(key="pred__llm_ocr_enriched_dataset__pydantic"),
    },
)
def ocr__exported_json(
    context: AssetExecutionContext,
    ocr_dataset: BaseDataset[BaseDataItem],
    config: OcrExportConfig,
) -> dict[str, Path]:
    """Export LLM OCR-enriched dataset to JSON files for backup/review.

    Args:
        context: Dagster execution context
        ocr_dataset: Dataset with LLM OCR text extraction results
        config: Export configuration

    Returns:
        Dictionary mapping schematism names to output file paths
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files: dict[str, Path] = {}

    if config.group_by_schematism:
        by_schematism: dict[str, list[dict]] = {}

        for item in ocr_dataset.items:
            if not item.metadata:
                continue

            schematism_name = item.metadata.schematism_name or "unknown"
            if schematism_name not in by_schematism:
                by_schematism[schematism_name] = []

            record = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": schematism_name,
                "ocr_text": item.text,
            }
            by_schematism[schematism_name].append(record)

        for schematism_name, records in by_schematism.items():
            records.sort(key=lambda r: r.get("sample_id", ""))

            output_file = output_dir / f"{timestamp}_{schematism_name}_ocr.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schematism_name": schematism_name,
                        "generated_at": timestamp,
                        "total_records": len(records),
                        "records": records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2 if config.pretty_print else None,
                )

            output_files[schematism_name] = output_file
            context.log.info(
                f"Exported {len(records)} OCR records for '{schematism_name}' to {output_file}"
            )
    else:
        all_records = []
        for item in ocr_dataset.items:
            if not item.metadata:
                continue

            record = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": item.metadata.schematism_name,
                "ocr_text": item.text,
            }
            all_records.append(record)

        all_records.sort(
            key=lambda r: (r.get("schematism_name", ""), r.get("sample_id", ""))
        )

        output_file = output_dir / f"{timestamp}_all_ocr.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": timestamp,
                    "total_records": len(all_records),
                    "records": all_records,
                },
                f,
                ensure_ascii=False,
                indent=2 if config.pretty_print else None,
            )

        output_files["all"] = output_file
        context.log.info(f"Exported {len(all_records)} OCR records to {output_file}")

    context.add_output_metadata(
        {
            "output_dir": MetadataValue.path(str(output_dir)),
            "files_created": MetadataValue.int(len(output_files)),
            "file_paths": MetadataValue.json(
                {k: str(v) for k, v in output_files.items()}
            ),
            "total_records": MetadataValue.int(len(ocr_dataset.items)),
        }
    )

    return output_files


class SourceGenerationConfig(dg.Config):
    """Configuration for source generation."""

    task_name: str = "source_generation"
    context_strategy: str = "accumulating"
    enable_cache: bool = True
    group_by_schematism_name: bool = True


@dg.asset(
    key_prefix=[AssetLayer.FCT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "dataset": AssetIn(key="pred__merged_ocr_source_dataset__pydantic"),
    },
)
def source__generated_dataset__pydantic(
    context: AssetExecutionContext,
    dataset: PredictionItemDataset,
    config: SourceGenerationConfig,
    llm_engine_resource: LLMEngineResource,
    images_repository: dg.ResourceParam[ImageRepository],
) -> PredictionItemDataset:
    """Generate source (Latin) dataset from parsed (Polish) ground truth.

    This asset uses an LLM to find Latin source text on page images
    that corresponds to the parsed Polish ground truth entries.

    Uses DatasetProcessor with the configured ContextStrategy (default: accumulating).
    """
    llm_engine = llm_engine_resource.get_engine(
        cached=config.enable_cache, images_repository=images_repository
    )

    item_processor = ItemProcessor(
        llm_engine=llm_engine,
        request_handler=StandardRequestHandler(output_type=SchematismPage),
        response_handler=PredictionsRefinementResponseHandler[PredictionDataItem](),
    )

    message_builder = Jinja2MessageBuilder(
        prompt_renderer=Jinja2PromptRenderer(template_dir="prompts"),
        task_name=config.task_name,
    )

    context_strategy = SlidingWindowStrategy(
        message_builder=message_builder, window_size=5, strip_images=True
    )

    dataset_processor = DatasetProcessor(
        item_processor=item_processor,
        images_repository=images_repository,
        context_provider=SOURCE_GENERATION_CONTEXT_PROVIDERS,
        context_strategy=context_strategy,
    )

    use_case = GenerateSourceDataset(
        dataset_processor=dataset_processor,
    )

    request = GenerateSourceDatasetRequest(
        dataset=dataset,
        group_by_schematism_name=config.group_by_schematism_name,
    )

    response = use_case.execute(request)

    # Log a random sample for inspection
    if response.dataset.items:
        random_sample = random.choice(response.dataset.items)
        sample_preview = {
            k: v for k, v in random_sample.model_dump().items() if k != "image"
        }
    else:
        sample_preview = {}

    context.add_output_metadata(
        {
            "dataset_size": MetadataValue.int(len(response.dataset.items)),
            "random_sample": MetadataValue.json(sample_preview),
        }
    )

    return response.dataset


class SourceExportConfig(dg.Config):
    """Configuration for source dataset JSON export."""

    output_dir: str = str(OUTPUTS_DIR / "source_generation")
    group_by_schematism: bool = True
    pretty_print: bool = True


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.JSON},
    ins={
        "source_dataset": AssetIn(key="source__generated_dataset__pydantic"),
    },
)
def source__exported_json(
    context: AssetExecutionContext,
    source_dataset: BaseDataset[PredictionDataItem],
    config: SourceExportConfig,
) -> dict[str, Path]:
    """Export generated source dataset to JSON files for manual review.

    Args:
        context: Dagster execution context
        source_dataset: Dataset with generated Latin source entries
        config: Export configuration

    Returns:
        Dictionary mapping schematism names to output file paths
    """
    output_dir = Path(output_dir) if (output_dir := config.output_dir) else Path()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files: dict[str, Path] = {}

    if config.group_by_schematism:
        # Group items by schematism name
        by_schematism: dict[str, list[dict]] = {}

        for item in source_dataset.items:
            if not item.metadata:
                continue

            schematism_name = item.metadata.schematism_name or "unknown"
            if schematism_name not in by_schematism:
                by_schematism[schematism_name] = []

            # Build export record
            record = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": schematism_name,
            }

            if item.predictions:
                record["source"] = item.predictions.model_dump()

            by_schematism[schematism_name].append(record)

        # Write one JSON file per schematism
        for schematism_name, records in by_schematism.items():
            # Sort by sample_id for consistent ordering
            records.sort(key=lambda r: r.get("sample_id", ""))

            output_file = output_dir / f"{timestamp}_{schematism_name}_source.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schematism_name": schematism_name,
                        "generated_at": timestamp,
                        "total_records": len(records),
                        "records": records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2 if config.pretty_print else None,
                )

            output_files[schematism_name] = output_file
            context.log.info(
                f"Exported {len(records)} records for '{schematism_name}' to {output_file}"
            )
    else:
        # Single file with all records
        all_records = []
        for item in source_dataset.items:
            if not item.metadata:
                continue

            record = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": item.metadata.schematism_name,
            }

            if item.predictions:
                record["source"] = item.predictions.model_dump()

            all_records.append(record)

        # Sort by schematism_name, then sample_id
        all_records.sort(
            key=lambda r: (r.get("schematism_name", ""), r.get("sample_id", ""))
        )

        output_file = output_dir / f"{timestamp}_all_source.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": timestamp,
                    "total_records": len(all_records),
                    "records": all_records,
                },
                f,
                ensure_ascii=False,
                indent=2 if config.pretty_print else None,
            )

        output_files["all"] = output_file
        context.log.info(f"Exported {len(all_records)} records to {output_file}")

    context.add_output_metadata(
        {
            "output_dir": MetadataValue.path(str(output_dir)),
            "files_created": MetadataValue.int(len(output_files)),
            "file_paths": MetadataValue.json(
                {k: str(v) for k, v in output_files.items()}
            ),
            "total_records": MetadataValue.int(len(source_dataset.items)),
        }
    )

    return output_files


# class SourceGenerationConfig(dg.Config):
#     """Configuration for source generation."""
#
#     system_prompt: str = "tasks/source_generation/system.j2"
#     user_prompt: str = "tasks/source_generation/user.j2"
#     accumulate_context: bool = True
#     enable_cache: bool = True
#     include_next_page_ocr: bool = True
#
#
# @dg.asset(
#     key_prefix=[AssetLayer.FCT, DataSource.HUGGINGFACE],
#     group_name=ResourceGroup.DATA,
#     kinds={Kinds.PYTHON, Kinds.PYDANTIC},
#     ins={
#         "dataset": AssetIn(key="pred__merged_ocr_source_dataset__pydantic"),
#     },
# )
# def source__generated_dataset__pydantic(
#     context: AssetExecutionContext,
#     dataset: PredictionItemDataset,
#     config: SourceGenerationConfig,
#     llm_engine_resource: LLMEngineResource,
#     images_repository: dg.ResourceParam[ImageRepository],
# ) -> BaseDataset[PredictionDataItem]:
#     """Generate source (Latin) dataset from parsed (Polish) ground truth.
#
#     This asset uses an LLM to find Latin source text on page images
#     that corresponds to the parsed Polish ground truth entries.
#
#     Args:
#         context: Dagster execution context
#         dataset: Dataset with merged ground truth and OCR text
#         config: Source generation configuration
#         llm_engine_resource: LLM engine resource
#         images_repository: Image repository resource
#
#     Returns:
#         Dataset with generated Latin source entries
#     """
#     engine = llm_engine_resource.get_engine(cached=config.enable_cache)
#     use_case = GenerateSourceDataset(
#         llm_engine=engine,
#         image_repository=images_repository,
#     )
#
#     request = GenerateSourceDatasetRequest(
#         dataset=dataset,
#         system_prompt=config.system_prompt,
#         user_prompt=config.user_prompt,
#     )
#
#     response = use_case.execute(request)
#
#     # Log a random sample for inspection
#     if response.dataset.items:
#         random_sample = random.choice(response.dataset.items)
#         sample_preview = {
#             k: v for k, v in random_sample.model_dump().items() if k != "image"
#         }
#     else:
#         sample_preview = {}
#
#     context.add_output_metadata(
#         {
#             "dataset_size": MetadataValue.int(len(response.dataset.items)),
#             "execution_stats": MetadataValue.json(dict(response.execution_stats)),
#             "random_sample": MetadataValue.json(sample_preview),
#         }
#     )
#
#     context.log.info(
#         f"Generated source dataset with {len(response.dataset.items)} items"
#     )
#
#     return response.dataset
#


class SourceExportConfig(dg.Config):
    """Configuration for source dataset JSON export."""

    output_dir: str = str(OUTPUTS_DIR / "source_generation")
    group_by_schematism: bool = True
    pretty_print: bool = True


@dg.asset(
    key_prefix=[AssetLayer.MRT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.JSON},
    ins={
        "source_dataset": AssetIn(key="source__generated_dataset__pydantic"),
    },
)
def source__exported_json(
    context: AssetExecutionContext,
    source_dataset: BaseDataset[PredictionDataItem],
    config: SourceExportConfig,
) -> dict[str, Path]:
    """Export generated source dataset to JSON files for manual review.

    Args:
        context: Dagster execution context
        source_dataset: Dataset with generated Latin source entries
        config: Export configuration

    Returns:
        Dictionary mapping schematism names to output file paths
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files: dict[str, Path] = {}

    if config.group_by_schematism:
        # Group items by schematism name
        by_schematism: dict[str, list[dict]] = {}

        for item in source_dataset.items:
            if not item.metadata:
                continue

            schematism_name = item.metadata.schematism_name or "unknown"
            if schematism_name not in by_schematism:
                by_schematism[schematism_name] = []

            # Build export record
            record = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": schematism_name,
            }

            if item.predictions:
                record["source"] = item.predictions.model_dump()

            by_schematism[schematism_name].append(record)

        # Write one JSON file per schematism
        for schematism_name, records in by_schematism.items():
            # Sort by sample_id for consistent ordering
            records.sort(key=lambda r: r.get("sample_id", ""))

            output_file = output_dir / f"{timestamp}_{schematism_name}_source.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schematism_name": schematism_name,
                        "generated_at": timestamp,
                        "total_records": len(records),
                        "records": records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2 if config.pretty_print else None,
                )

            output_files[schematism_name] = output_file
            context.log.info(
                f"Exported {len(records)} records for '{schematism_name}' to {output_file}"
            )
    else:
        # Single file with all records
        all_records = []
        for item in source_dataset.items:
            if not item.metadata:
                continue

            record = {
                "sample_id": item.metadata.sample_id,
                "filename": item.metadata.filename,
                "schematism_name": item.metadata.schematism_name,
            }

            if item.predictions:
                record["source"] = item.predictions.model_dump()

            all_records.append(record)

        # Sort by schematism_name, then sample_id
        all_records.sort(
            key=lambda r: (r.get("schematism_name", ""), r.get("sample_id", ""))
        )

        output_file = output_dir / f"{timestamp}_all_source.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": timestamp,
                    "total_records": len(all_records),
                    "records": all_records,
                },
                f,
                ensure_ascii=False,
                indent=2 if config.pretty_print else None,
            )

        output_files["all"] = output_file
        context.log.info(f"Exported {len(all_records)} records to {output_file}")

    context.add_output_metadata(
        {
            "output_dir": MetadataValue.path(str(output_dir)),
            "files_created": MetadataValue.int(len(output_files)),
            "file_paths": MetadataValue.json(
                {k: str(v) for k, v in output_files.items()}
            ),
            "total_records": MetadataValue.int(len(source_dataset.items)),
        }
    )

    return output_files
