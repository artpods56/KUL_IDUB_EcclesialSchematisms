"""Fake storage implementations for testing.

This module provides in-memory storage fakes that replace file system operations.
"""

from pathlib import Path
from PIL import Image
from typing import Any, Iterator, Self, override
import io
from contextlib import contextmanager

from notarius.application.ports.outbound.storage import (
    FileStorage,
    AbstractFileRepository,
)
from notarius.domain.protocols import FileStreamProtocol


class FakeFileStorage(FileStorage):
    """Fake in-memory file storage for testing.

    This fake storage:
    - Implements the FileStorage interface
    - Stores file content in memory (dict of BytesIO)
    - Tracks all operations for assertions
    """

    def __init__(self, storage_root: Path = Path("/fake/storage")):
        self.storage_root = storage_root
        self.files: dict[Path, bytes] = {}
        self.save_calls: list[tuple[bytes, Path]] = []
        self.load_calls: list[Path] = []
        self.delete_calls: list[Path] = []
        self.exists_calls: list[Path] = []

    def save(self, stream: FileStreamProtocol, file_path: Path) -> Path:
        content = stream.read()
        self.save_calls.append((content, file_path))
        self.files[file_path] = content
        return self.storage_root / file_path

    @contextmanager
    def load(self, file_path: Path) -> Iterator[FileStreamProtocol]:
        self.load_calls.append(file_path)
        if file_path not in self.files:
            raise FileNotFoundError(f"File not found: {file_path}")

        stream = io.BytesIO(self.files[file_path])
        try:
            yield stream
        finally:
            stream.close()

    def delete(self, file_path: Path) -> None:
        self.delete_calls.append(file_path)
        if file_path in self.files:
            del self.files[file_path]

    def exists(self, file_path: Path) -> bool:
        self.exists_calls.append(file_path)
        return file_path in self.files


class FakeImageStorage(AbstractFileRepository[Image.Image]):
    """Fake in-memory image storage for testing without file I/O.

    This fake storage:
    - Implements the ImageRepository interface methods
    - Stores images in memory (dict)
    - Tracks all operations for assertions
    - No actual file system access
    """

    def __init__(self, storage: FileStorage | None = None):
        """Initialize fake image storage with in-memory dict."""
        self.storage = storage or FakeFileStorage()
        self.images: dict[str, Image.Image] = {}
        self.add_calls: list[tuple[Image.Image, str]] = []
        self.get_calls: list[Path] = []
        self.exists_calls: list[str] = []
        self.default_image: Image.Image | None = None

    def add(self, file: Image.Image, name: str) -> Path:
        """Add image to in-memory storage.

        Args:
            file: PIL Image to store
            name: Name/path for the image

        Returns:
            Path to the stored image
        """
        self.add_calls.append((file, name))

        # Ensure RGB mode (matching real implementation)
        image_to_store = file if file.mode == "RGB" else file.convert("RGB")

        # Store in memory
        self.images[name] = image_to_store

        return Path(name)

    def get(self, path: Path) -> Image.Image:
        """Get image from in-memory storage.

        Args:
            path: Path to the image

        Returns:
            PIL Image

        Raises:
            KeyError: If image not found
        """
        self.get_calls.append(path)

        # Try to get from storage
        path_str = str(path)
        if path_str in self.images:
            return self.images[path_str]

        # If not found and default image set, return default
        if self.default_image is not None:
            return self.default_image

        # Create a default white image if nothing configured
        return Image.new("RGB", (800, 600), color="white")

    def exists(self, name: str) -> bool:
        """Check if image exists in storage.

        Args:
            name: Name/path of the image

        Returns:
            True if image exists, False otherwise
        """
        self.exists_calls.append(name)
        return name in self.images

    def get_path(self, name: str) -> Path:
        """Get path for an image.

        Args:
            name: Name of the image

        Returns:
            Path to the image
        """
        return Path(name)

    def load_image(self, path: str) -> Image.Image:
        """Load image by path (alternative interface for compatibility).

        Args:
            path: Path to the image

        Returns:
            PIL Image
        """
        return self.get(Path(path))

    def configure_default_image(self, image: Image.Image) -> None:
        """Configure a default image to return when image not found.

        Args:
            image: The default PIL Image to return
        """
        self.default_image = image

    def reset(self) -> None:
        """Reset storage and call tracking."""
        self.images.clear()
        self.add_calls.clear()
        self.get_calls.clear()
        self.exists_calls.clear()
        self.default_image = None


from notarius.application.ports.outbound.pdf_ingestor import PDFIngestor


class FakePDFIngestor(PDFIngestor):
    """Fake PDF ingestor for testing without real PDF files."""

    def __init__(self, pages_content: list[str] | None = None):
        self.pages_content = pages_content or ["Page 1 content", "Page 2 content"]
        self.ingest_calls: list[Path] = []

    @override
    def ingest(self, pdf_path: Path) -> list[tuple[str | None, Image.Image]]:
        self.ingest_calls.append(pdf_path)
        results: list[tuple[str | None, Image.Image]] = []
        for i, content in enumerate(self.pages_content):
            image = Image.new("RGB", (800, 600), color="white")
            results.append((content, image))
        return results
