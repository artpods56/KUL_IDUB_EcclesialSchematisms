from dagster import AssetExecutionContext, MetadataValue
from datasets import IterableDataset, Dataset
from datasets.fingerprint import generate_fingerprint

from notarius.application import ports
from notarius.application.use_cases.ingestion.ingest_documents_from_pdf import (
    IngestPDFUseCase,
    IngestPDFRequest,
)
from notarius.infrastructure.persistence import dataset_repository
from notarius.infrastructure.persistence.storage import ImageRepository

from notarius.schemas.configs.dataset_config import BaseDatasetConfig
from notarius.schemas.data.pipeline import BaseItemDataset
import dagster as dg

from notarius.orchestration.constants import (
    DataSource,
    AssetLayer,
    ResourceGroup,
    Kinds,
)


class PdfToDatasetConfig(dg.Config):
    file_paths: list[str] | None = None
    source_dir: str = "pdfs"
    glob_pattern: str = "*.pdf"


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.FILE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON},
)
def raw__pdf__dataset(
    context: dg.AssetExecutionContext,
    config: PdfToDatasetConfig,
    file_storage: dg.ResourceParam[ports.FileStorage],
    images_repository: dg.ResourceParam[ImageRepository],
) -> BaseItemDataset:
    request = IngestPDFRequest(
        source_dir=config.source_dir,
        pdf_paths=config.file_paths or [],
        glob_pattern=config.glob_pattern,
    )

    # Import the concrete PDFPlumberIngestor implementation
    from notarius.infrastructure.pdf import PDFPlumberIngestor

    pdf_ingestor = PDFPlumberIngestor(storage=file_storage)

    use_case = IngestPDFUseCase(
        pdf_ingestor=pdf_ingestor,
        image_repository=images_repository,
    )

    response = use_case.execute(request)

    context.add_asset_metadata(
        {path.name: MetadataValue.path(str(path)) for path in request.get_pdf_paths()}
    )

    context.add_output_metadata(
        {"all_items": MetadataValue.int(len(response.dataset.items))}
    )

    return response.dataset


class RawHuggingFaceDatasetConfig(dg.Config, BaseDatasetConfig): ...


@dg.asset(
    key_prefix=[AssetLayer.STG, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.HUGGINGFACE},
    io_manager_key="hf_dataset_io_manager",
)
def raw__hf__dataset(
    context: AssetExecutionContext, config: RawHuggingFaceDatasetConfig
) -> Dataset:
    dataset = dataset_repository.load_huggingface_dataset(
        config=config,
        streaming=config.streaming,
    )

    if isinstance(dataset, IterableDataset):
        raise NotImplementedError("The pipeline doesn't support streamed datasets.")

    dataset = dataset.add_column(
        name="sample_id",
        column=list(range(len(dataset))),
        new_fingerprint=generate_fingerprint(dataset),
    )

    return dataset
