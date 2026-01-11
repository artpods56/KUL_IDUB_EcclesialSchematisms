import io
from pathlib import Path
from typing import final, cast, override
from dataclasses import dataclass, field

from PIL import Image
from notarius.application.ports.outbound.pdf_ingestor import PDFIngestor
from notarius.application.ports.outbound.storage import AbstractFileRepository
from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    BaseUseCase,
)
from notarius.schemas.data.pipeline import (
    BaseDataItem,
    BaseItemDataset,
    BaseMetaData,
)


@dataclass
class IngestPDFRequest(BaseRequest):
    source_dir: str | None = None
    pdf_paths: list[str] = field(default_factory=list)
    glob_pattern: str = "*.pdf"

    def __post_init__(self) -> None:
        if not self.source_dir and not self.pdf_paths:
            raise ValueError("Either 'source_dir' or 'pdf_paths' must be provided")

    def get_pdf_paths(self) -> set[Path]:
        pdf_paths = set(map(Path, self.pdf_paths))
        if self.source_dir:
            pdf_paths.update(sorted(Path(self.source_dir).glob(self.glob_pattern)))
        return pdf_paths


@dataclass
class IngestPDFResponse(BaseResponse):
    dataset: BaseItemDataset


@final
class IngestPDFUseCase(BaseUseCase[IngestPDFRequest, IngestPDFResponse]):
    def __init__(
        self,
        pdf_ingestor: PDFIngestor,
        image_repository: AbstractFileRepository[Image.Image],
    ):
        self.pdf_ingestor = pdf_ingestor
        self.image_repository = image_repository

    def _ingest_pdf(self, pdf_path: Path) -> list[BaseDataItem]:
        """Ingest a single PDF and create BaseDataItems for each page.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of BaseDataItem objects, one per page
        """
        items: list[BaseDataItem] = []

        # Use injected pdf_ingestor to extract pages
        pages = self.pdf_ingestor.ingest(pdf_path)

        for i, (text, image) in enumerate(pages):
            pdf_image_filename = f"{pdf_path.stem}_{i}"

            # Check if image already exists to avoid duplicates
            if self.image_repository.exists(pdf_image_filename):
                image_path = self.image_repository.get_path(pdf_image_filename)
            else:
                image_path = self.image_repository.add(image, pdf_image_filename)

            items.append(
                BaseDataItem(
                    image_path=str(image_path),
                    text=text,
                    metadata=BaseMetaData(
                        sample_id=i,
                        filename=image_path.name,
                        schematism_name=pdf_path.name,
                    ),
                )
            )

        return items

    @override
    def execute(self, request: IngestPDFRequest) -> IngestPDFResponse:
        all_items: list[BaseDataItem] = []
        for pdf_path in request.get_pdf_paths():
            items = self._ingest_pdf(pdf_path)
            all_items.extend(items)

        return IngestPDFResponse(dataset=BaseItemDataset(items=all_items))
