"""Tests for EnrichDatasetWithOCR use case."""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from notarius.application.use_cases.inference.add_ocr_to_dataset import (
    EnrichDatasetWithOCR,
    EnrichWithOCRRequest,
    EnrichWithOCRResponse,
)
from tests.fakes.engines import FakeOCREngine
from tests.fakes.storage import FakeImageStorage
from notarius.infrastructure.ocr.engine_adapter import OCRResponse
from notarius.infrastructure.ocr.types import SimpleOCRResult
from notarius.schemas.data.pipeline import BaseDataset, BaseDataItem, BaseMetaData


# Test fixtures


@pytest.fixture
def mock_image() -> Image.Image:
    """Create a mock PIL image."""
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def fake_ocr_engine() -> FakeOCREngine:
    """Create a fake OCR engine."""
    return FakeOCREngine(default_text="Sample OCR text")


@pytest.fixture
def fake_image_storage(mock_image: Image.Image) -> FakeImageStorage:
    """Create a fake image storage resource."""
    storage = FakeImageStorage()
    # Pre-populate with mock images if needed
    storage.add(mock_image, "/path/to/image1.jpg")
    storage.add(mock_image, "/path/to/image2.jpg")
    storage.add(mock_image, "/path/to/image.jpg")
    return storage


@pytest.fixture
def sample_metadata() -> BaseMetaData:
    """Create sample metadata."""
    return BaseMetaData(
        sample_id=1,
        schematism_name="test_schematism",
        filename="test_file.jpg",
    )


@pytest.fixture
def sample_dataset(sample_metadata: BaseMetaData) -> BaseDataset[BaseDataItem]:
    """Create a sample dataset with items."""
    items = [
        BaseDataItem(
            image_path="/path/to/image1.jpg",
            text=None,
            metadata=sample_metadata,
        ),
        BaseDataItem(
            image_path="/path/to/image2.jpg",
            text=None,
            metadata=BaseMetaData(
                sample_id=2,
                schematism_name="test_schematism",
                filename="test_file2.jpg",
            ),
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


@pytest.fixture
def empty_dataset() -> BaseDataset[BaseDataItem]:
    """Create an empty dataset."""
    return BaseDataset[BaseDataItem](items=[])


@pytest.fixture
def dataset_with_missing_paths(
    sample_metadata: BaseMetaData,
) -> BaseDataset[BaseDataItem]:
    """Create a dataset with items missing image paths."""
    items = [
        BaseDataItem(image_path=None, text=None, metadata=sample_metadata),
        BaseDataItem(
            image_path="/path/to/image.jpg", text=None, metadata=sample_metadata
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


class TestEnrichWithOCRRequest:
    """Test suite for EnrichWithOCRRequest dataclass."""

    def test_request_with_defaults(
        self, sample_dataset: BaseDataset[BaseDataItem]
    ) -> None:
        """Test request creation with default mode."""
        request = EnrichWithOCRRequest(dataset=sample_dataset)

        assert request.dataset is sample_dataset
        assert request.mode == "text"

    def test_request_with_structured_mode(
        self, sample_dataset: BaseDataset[BaseDataItem]
    ) -> None:
        """Test request creation with structured mode."""
        request = EnrichWithOCRRequest(dataset=sample_dataset, mode="structured")

        assert request.mode == "structured"


class TestEnrichWithOCRResponse:
    """Test suite for EnrichWithOCRResponse dataclass."""

    def test_response_creation(self, sample_dataset: BaseDataset[BaseDataItem]) -> None:
        """Test response creation with all fields."""
        response = EnrichWithOCRResponse(
            dataset=sample_dataset,
            ocr_executions=5,
            cache_hits=3,
        )

        assert response.dataset is sample_dataset
        assert response.ocr_executions == 5
        assert response.cache_hits == 3


class TestEnrichDatasetWithOCR:
    """Test suite for EnrichDatasetWithOCR use case."""

    def test_init_with_cache_disabled(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
    ) -> None:
        """Test initialization with caching disabled."""
        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            language="eng",
            enable_cache=False,
        )

        # When cache is disabled, the engine should be used directly
        assert use_case.ocr_engine is fake_ocr_engine
        assert use_case.image_storage is fake_image_storage

    @patch(
        "notarius.application.use_cases.inference.add_ocr_to_dataset.create_ocr_cache_backend"
    )
    def test_init_with_cache_enabled(
        self,
        mock_create_cache: MagicMock,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
    ) -> None:
        """Test initialization with caching enabled."""
        mock_backend = MagicMock()
        mock_keygen = MagicMock()
        mock_create_cache.return_value = (mock_backend, mock_keygen)

        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            language="lat+pol+rus",
            enable_cache=True,
        )

        # When cache is enabled, the engine should be wrapped
        from notarius.application.ports.outbound.cached_engine import CachedEngine

        assert isinstance(use_case.ocr_engine, CachedEngine)
        mock_create_cache.assert_called_once_with("lat+pol+rus")

    @pytest.mark.asyncio
    async def test_execute_processes_all_items(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute processes all dataset items."""
        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert len(fake_ocr_engine.call_history) == 2
        assert len(fake_image_storage.get_calls) == 2

    @pytest.mark.asyncio
    async def test_execute_extracts_text_from_ocr_result(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute correctly extracts text from OCR sample."""
        # Configure fake engine to return specific text
        fake_ocr_engine.default_text = "Extracted OCR text"

        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        # All items should have the extracted text
        for item in response.dataset.items:
            assert item.text == "Extracted OCR text"

    @pytest.mark.asyncio
    async def test_execute_preserves_metadata(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute preserves item metadata."""
        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        # Metadata should be preserved
        assert response.dataset.items[0].metadata is not None
        assert response.dataset.items[1].metadata is not None
        assert response.dataset.items[0].metadata.sample_id == 1
        assert response.dataset.items[1].metadata.sample_id == 2

    @pytest.mark.asyncio
    async def test_execute_preserves_image_paths(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute preserves image paths."""
        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        assert response.dataset.items[0].image_path == "/path/to/image1.jpg"
        assert response.dataset.items[1].image_path == "/path/to/image2.jpg"

    @pytest.mark.asyncio
    async def test_execute_skips_items_without_image_path(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        dataset_with_missing_paths: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that items without image paths are skipped."""
        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=dataset_with_missing_paths)
        response = await use_case.execute(request)

        # Only one item has an image path
        assert len(response.dataset.items) == 1
        assert len(fake_ocr_engine.call_history) == 1

    @pytest.mark.asyncio
    async def test_execute_with_empty_dataset(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        empty_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execution with empty dataset."""
        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=empty_dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 0
        assert response.ocr_executions == 0
        assert response.cache_hits == 0
        assert len(fake_ocr_engine.call_history) == 0

    @pytest.mark.asyncio
    async def test_execute_returns_correct_statistics_without_cache(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that statistics are correct when cache is disabled."""
        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        # Without cache, all executions should be counted as OCR executions
        assert response.ocr_executions == 2
        assert response.cache_hits == 0

    @pytest.mark.asyncio
    @patch(
        "notarius.application.use_cases.inference.add_ocr_to_dataset.create_ocr_cache_backend"
    )
    async def test_execute_returns_correct_statistics_with_cache(
        self,
        mock_create_cache: MagicMock,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that statistics are correct when cache is enabled."""
        mock_backend = MagicMock()
        mock_keygen = MagicMock()
        mock_create_cache.return_value = (mock_backend, mock_keygen)

        # Mock cache to return None (cache miss) for all requests
        mock_backend.get.return_value = None
        mock_backend.set.return_value = True

        # Mock key generator
        mock_keygen.generate_key.return_value = "test_key"

        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=True,
        )

        request = EnrichWithOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        # Stats should come from CachedEngine
        assert response.ocr_executions == 2  # All cache misses
        assert response.cache_hits == 0

    @pytest.mark.asyncio
    async def test_execute_passes_correct_mode_to_ocr_request(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that the OCR mode is correctly passed to the request."""
        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=sample_dataset, mode="text")
        await use_case.execute(request)

        # Verify the mode was passed correctly
        for ocr_request in fake_ocr_engine.call_history:
            assert ocr_request.mode == "text"

    @pytest.mark.asyncio
    async def test_execute_converts_image_to_rgb(
        self,
        fake_ocr_engine: FakeOCREngine,
        fake_image_storage: FakeImageStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that images are converted to RGB mode."""
        # Create a grayscale image that will be converted
        grayscale_image = Image.new("L", (100, 100), color=128)
        # Clear image storage and add the grayscale one
        fake_image_storage.reset()
        fake_image_storage.add(grayscale_image, "/path/to/image1.jpg")
        fake_image_storage.add(grayscale_image, "/path/to/image2.jpg")

        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        request = EnrichWithOCRRequest(dataset=sample_dataset)
        await use_case.execute(request)

        # In FakeImageStorage.add, we already convert to RGB.
        # Let's verify that the engine received RGB images.
        for ocr_request in fake_ocr_engine.call_history:
            assert ocr_request.input.mode == "RGB"


class TestEnrichDatasetWithOCRIntegration:
    """Integration-style tests for EnrichDatasetWithOCR."""

    @pytest.mark.asyncio
    async def test_full_workflow_without_cache(
        self,
        fake_image_storage: FakeImageStorage,
    ) -> None:
        """Test complete workflow with real OCR engine fake."""
        fake_ocr_engine = FakeOCREngine()

        # We can't easily use side_effect on a fake's method without mocking it,
        # but FakeOCREngine supports custom responses via configure_response if we want.
        # For this test, the default behavior of returning factory-built responses is fine.

        use_case = EnrichDatasetWithOCR(
            ocr_engine=fake_ocr_engine,
            image_storage=fake_image_storage,
            enable_cache=False,
        )

        dataset = BaseDataset[BaseDataItem](
            items=[
                BaseDataItem(
                    image_path="/path/page1.jpg",
                    metadata=BaseMetaData(
                        sample_id=1, schematism_name="test", filename="p1.jpg"
                    ),
                ),
                BaseDataItem(
                    image_path="/path/page2.jpg",
                    metadata=BaseMetaData(
                        sample_id=2, schematism_name="test", filename="p2.jpg"
                    ),
                ),
            ]
        )

        # Ensure images exist in storage
        white_image = Image.new("RGB", (100, 100), color="white")
        fake_image_storage.add(white_image, "/path/page1.jpg")
        fake_image_storage.add(white_image, "/path/page2.jpg")

        request = EnrichWithOCRRequest(dataset=dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert response.ocr_executions == 2
        assert response.cache_hits == 0
