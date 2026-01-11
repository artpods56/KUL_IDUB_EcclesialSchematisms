import abc
from pathlib import Path
from PIL import Image


class PDFIngestor(abc.ABC):
    """Port for ingesting documents from PDF files."""

    @abc.abstractmethod
    def ingest(self, pdf_path: Path) -> list[tuple[str | None, Image.Image]]:
        """Ingest a PDF file and return its pages as text and images.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of tuples, each containing extracted text and a PIL Image for a page.
        """
        raise NotImplementedError
