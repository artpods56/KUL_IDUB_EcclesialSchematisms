"""PDFPlumber-based PDF ingestor implementation."""

import io
from pathlib import Path
from typing import cast, override

import pdfplumber
from PIL import Image

from notarius.application.ports.outbound.pdf_ingestor import PDFIngestor
from notarius.application.ports.outbound.storage import FileStorage


class PDFPlumberIngestor(PDFIngestor):
    """PDF ingestor using pdfplumber for text extraction and image rendering.

    This implementation uses pdfplumber to:
    - Extract text from each page
    - Render pages as PIL Images

    Example:
        storage = LocalFileStorage()
        ingestor = PDFPlumberIngestor(storage=storage)
        pages = ingestor.ingest(Path("document.pdf"))
        for text, image in pages:
            # Process text and image
            pass
    """

    def __init__(self, storage: FileStorage):
        """Initialize PDFPlumberIngestor with file storage.

        Args:
            storage: File storage for accessing PDF files
        """
        self.storage = storage

    @override
    def ingest(self, pdf_path: Path) -> list[tuple[str | None, Image.Image]]:
        """Ingest a PDF file and extract text and images from each page.

        Args:
            pdf_path: Path to the PDF file relative to storage root

        Returns:
            List of tuples containing (text, image) for each page

        Raises:
            FileNotFoundError: If PDF file doesn't exist in storage

        Example:
            pages = ingestor.ingest(Path("schematisms/doc1.pdf"))
            assert len(pages) > 0
            text, image = pages[0]
        """
        items: list[tuple[str | None, Image.Image]] = []

        with self.storage.load(pdf_path) as pdf_stream:
            with pdfplumber.open(cast(io.BytesIO, pdf_stream)) as pdf:
                for page in pdf.pages:
                    # Extract text from page
                    text = page.extract_text()

                    # Render page as image
                    image = page.to_image().original

                    items.append((text, image))

        return items
