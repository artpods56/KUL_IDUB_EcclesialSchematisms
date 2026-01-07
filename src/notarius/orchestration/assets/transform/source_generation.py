"""Assets for generating source (Latin) dataset from parsed (Polish) ground truth."""

import random
from pathlib import Path

import dagster as dg
from dagster import AssetExecutionContext, AssetIn, MetadataValue

from notarius.application import ports
from notarius.application.services import (
    DatasetProcessor,
    ItemProcessor,
    SlidingWindowStrategy,
)
from notarius.application.services.message_builder import Jinja2MessageBuilder
from notarius.application.services.processors.item_processor import (
    StandardRequestHandler,
    PredictionsRefinementResponseHandler,
)
from notarius.application.use_cases.export.json_export import (
    JsonExportUseCase,
    JsonExportRequest,
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
    file_storage: dg.ResourceParam[ports.FileStorage],
) -> dict[str, Path]:
    """Export LLM OCR-enriched dataset to JSON files for backup/review.

    Args:
        context: Dagster execution context
        ocr_dataset: Dataset with LLM OCR text extraction results
        config: Export configuration
        file_storage: File storage resource

    Returns:
        Dictionary mapping schematism names to output file paths
    """
    use_case = JsonExportUseCase(storage=file_storage)
    request = JsonExportRequest(
        dataset=ocr_dataset,
        output_dir=Path(config.output_dir),
        group_by_schematism=config.group_by_schematism,
        pretty_print=config.pretty_print,
        filename_prefix="ocr",
    )
    response = use_case.execute(request)

    context.add_output_metadata(
        {
            "output_dir": MetadataValue.path(str(config.output_dir)),
            "files_created": MetadataValue.int(len(response.output_files)),
            "file_paths": MetadataValue.json(
                {k: str(v) for k, v in response.output_files.items()}
            ),
            "total_records": MetadataValue.int(len(ocr_dataset.items)),
        }
    )

    return response.output_files


class GenerateSourceGroundTruthDatasetConfig(dg.Config):
    """Configuration for source generation."""

    task_name: str = "source_generation"
    window_size: int = 5
    enable_cache: bool = True
    group_by_schematism_name: bool = True


@dg.asset(
    key_prefix=[AssetLayer.FCT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.PYDANTIC},
    ins={
        "dataset": AssetIn(key="pred__merged_ocr_parsed_dataset__pydantic"),
    },
)
def source__generated_dataset__pydantic(
    context: AssetExecutionContext,
    dataset: PredictionItemDataset,
    config: GenerateSourceGroundTruthDatasetConfig,
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
        message_builder=message_builder,
        window_size=config.window_size,
        strip_images=True,
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
    file_storage: dg.ResourceParam[ports.FileStorage],
) -> dict[str, Path]:
    """Export generated source dataset to JSON files for manual review.

    Args:
        context: Dagster execution context
        source_dataset: Dataset with generated Latin source entries
        config: Export configuration
        file_storage: File storage resource

    Returns:
        Dictionary mapping schematism names to output file paths
    """
    use_case = JsonExportUseCase(storage=file_storage)
    request = JsonExportRequest(
        dataset=source_dataset,
        output_dir=Path(config.output_dir),
        group_by_schematism=config.group_by_schematism,
        pretty_print=config.pretty_print,
        filename_prefix="source",
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
