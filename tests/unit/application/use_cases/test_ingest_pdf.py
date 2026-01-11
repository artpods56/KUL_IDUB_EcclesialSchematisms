"""Tests for IngestPDFUseCase."""

import pytest
from pathlib import Path
from PIL import Image

from notarius.application.use_cases.ingestion.ingest_documents_from_pdf import (
    IngestPDFUseCase,
    IngestPDFRequest,
)
from tests.fakes.storage import FakeImageStorage, FakePDFIngestor
from notarius.schemas.data.pipeline import BaseItemDataset


@pytest.fixture
def fake_pdf_ingestor() -> FakePDFIngestor:
    """Create a fake PDF ingestor."""
    return FakePDFIngestor(pages_content=["Page 1", "Page 2"])


@pytest.fixture
def fake_image_repository() -> FakeImageStorage:
    """Create a fake image repository."""
    return FakeImageStorage()


class TestIngestPDFUseCase:
    """Test suite for IngestPDFUseCase."""

    def test_execute_with_pdf_paths(
        self,
        fake_pdf_ingestor: FakePDFIngestor,
        fake_image_repository: FakeImageStorage,
    ) -> None:
        """Test ingestion with explicit PDF paths."""
        use_case = IngestPDFUseCase(
            pdf_ingestor=fake_pdf_ingestor,
            image_repository=fake_image_repository,
        )

        request = IngestPDFRequest(pdf_paths=["test.pdf"])
        response = use_case.execute(request)

        assert isinstance(response.dataset, BaseItemDataset)
        assert len(response.dataset.items) == 2
        assert response.dataset.items[0].text == "Page 1"
        assert response.dataset.items[1].text == "Page 2"

        # Check that images were saved
        assert len(fake_image_repository.add_calls) == 2
        assert fake_image_repository.add_calls[0][1] == "test_0"
        assert fake_image_repository.add_calls[1][1] == "test_1"

    def test_execute_skips_existing_images(
        self,
        fake_pdf_ingestor: FakePDFIngestor,
        fake_image_repository: FakeImageStorage,
    ) -> None:
        """Test that existing images are not re-saved."""
        use_case = IngestPDFUseCase(
            pdf_ingestor=fake_pdf_ingestor,
            image_repository=fake_image_repository,
        )

        # Pre-populate storage
        mock_image = Image.new("RGB", (10, 10))
        fake_image_repository.add(mock_image, "test_0")

        request = IngestPDFRequest(pdf_paths=["test.pdf"])
        use_case.execute(request)

        # Should only have 1 more add call (for test_1)
        # Total calls = 1 (pre-populate) + 1 (during execute)
        assert len(fake_image_repository.add_calls) == 2
        assert any(call[1] == "test_1" for call in fake_image_repository.add_calls)

    def test_request_validation(self) -> None:
        """Test request validation logic."""
        with pytest.raises(
            ValueError, match="Either 'source_dir' or 'pdf_paths' must be provided"
        ):
            IngestPDFRequest()

    def test_get_pdf_paths_from_dir(self, tmp_path: Path) -> None:
        """Test gathering PDF paths from a directory."""
        pdf1 = tmp_path / "a.pdf"
        pdf2 = tmp_path / "b.pdf"
        pdf1.touch()
        pdf2.touch()
        (tmp_path / "c.txt").touch()

        request = IngestPDFRequest(source_dir=str(tmp_path))
        paths = request.get_pdf_paths()

        assert len(paths) == 2
        assert Path(pdf1) in paths
        assert Path(pdf2) in paths
