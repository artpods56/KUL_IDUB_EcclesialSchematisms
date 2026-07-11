import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import fitz

from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import Source, SourceItem
from notarius_shared.object_store import LocalS3ObjectStore, create_local_s3_object_store


@dataclass(frozen=True, slots=True)
class IngestedPdfPage:
    source_item: SourceItem
    image_payload: bytes
    image_width: int
    image_height: int


@dataclass(frozen=True, slots=True)
class IngestedPdfSource:
    source: Source
    items: list[SourceItem]
    pages: list[IngestedPdfPage]
    document_uri: str


class PdfSourceIngestor:
    def __init__(self, object_store: LocalS3ObjectStore | None = None):
        self.object_store = object_store or create_local_s3_object_store()

    def ingest(
        self,
        project_id: UUID,
        filename: str,
        content: bytes,
        name: str | None = None,
        description: str | None = None,
    ) -> IngestedPdfSource:
        if not content:
            raise ValidationError("Uploaded PDF is empty")
        if not filename.lower().endswith(".pdf"):
            raise ValidationError("Uploaded file must be a PDF")

        source = Source(
            project_id=project_id,
            name=(name or Path(filename).stem or "Uploaded PDF").strip(),
            description=description,
        )
        safe_filename = _safe_filename(filename)
        object_prefix = f"projects/{project_id}/sources/{source.id}"
        document_uri = self.object_store.put_bytes(
            f"{object_prefix}/{safe_filename}",
            content,
        )

        try:
            document = fitz.open(stream=content, filetype="pdf")
            try:
                pages: list[IngestedPdfPage] = []
                for page_number, page in enumerate(document, start=1):
                    text = (page.get_text("text") or "").strip()
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False,
                    )
                    source_item = self._source_item_from_page(
                        source=source,
                        document_uri=document_uri,
                        object_prefix=object_prefix,
                        page_number=page_number,
                        text=text,
                    )
                    pages.append(
                        IngestedPdfPage(
                            source_item=source_item,
                            image_payload=pixmap.tobytes("png"),
                            image_width=pixmap.width,
                            image_height=pixmap.height,
                        )
                    )
            finally:
                document.close()
        except Exception as exc:
            raise ValidationError(f"Could not read uploaded PDF: {exc}") from exc

        if not pages:
            raise ValidationError("Uploaded PDF contains no pages")
        return IngestedPdfSource(
            source=source,
            items=[page.source_item for page in pages],
            pages=pages,
            document_uri=document_uri,
        )

    def _source_item_from_page(
        self,
        source: Source,
        document_uri: str,
        object_prefix: str,
        page_number: int,
        text: str,
    ) -> SourceItem:
        text_uri = self.object_store.put_bytes(
            f"{object_prefix}/pages/{page_number:04d}.txt",
            text.encode("utf-8"),
        )
        return SourceItem(
            source_id=source.id,
            order=page_number,
            text=None,
            metadata={
                "loader": "s3-text-page",
                "document_uri": document_uri,
                "text_object_uri": text_uri,
                "page_number": page_number,
                "text_preview": text[:240],
            },
        )


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).name).strip(".-")
    return cleaned or "document.pdf"
