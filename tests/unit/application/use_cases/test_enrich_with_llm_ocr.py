"""Tests for EnrichDatasetWithLLMOCR use case."""

import pytest
from collections.abc import Sequence
from typing import Any, override
from PIL import Image

from notarius.application.use_cases.inference.enrich_dataset_with_ocr_using_llm import (
    EnrichDatasetWithLLMOCR,
    EnrichWithLLMOCRRequest,
    EnrichWithLLMOCRResponse,
)
from notarius.application.services import DatasetProcessor
from tests.fakes.engines import FakeLLMEngine
from tests.fakes.storage import FakeImageStorage
from notarius.schemas.data.pipeline import BaseDataItem, BaseItemDataset, BaseMetaData


@pytest.fixture
def mock_image() -> Image.Image:
    """Create a mock PIL image."""
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def fake_llm_engine() -> FakeLLMEngine:
    """Create a fake LLM engine."""
    return FakeLLMEngine()


@pytest.fixture
def fake_image_storage(mock_image: Image.Image) -> FakeImageStorage:
    """Create a fake image storage resource."""
    storage = FakeImageStorage()
    storage.add(mock_image, "/path/to/image1.jpg")
    storage.add(mock_image, "/path/to/image2.jpg")
    return storage


@pytest.fixture
def sample_dataset() -> BaseItemDataset:
    """Create a sample dataset with items."""
    return BaseItemDataset(
        items=[
            BaseDataItem(
                image_path="/path/to/image1.jpg",
                metadata=BaseMetaData(
                    sample_id=1, schematism_name="test", filename="p1.jpg"
                ),
            ),
            BaseDataItem(
                image_path="/path/to/image2.jpg",
                metadata=BaseMetaData(
                    sample_id=2, schematism_name="test", filename="p2.jpg"
                ),
            ),
        ]
    )


class FakeDatasetProcessor(DatasetProcessor[BaseDataItem, Any]):
    """Fake dataset processor that tracks calls."""

    def __init__(self):
        # We don't call super().__init__ because we don't want to provide real dependencies
        self.process_parallel_calls: list[Sequence[BaseDataItem]] = []

    @override
    async def process_parallel_async(
        self, items: Sequence[BaseDataItem], max_concurrent: int = 10
    ) -> Sequence[BaseDataItem]:
        self.process_parallel_calls.append(items)
        return items


class TestEnrichDatasetWithLLMOCR:
    """Test suite for EnrichDatasetWithLLMOCR use case."""

    @pytest.mark.asyncio
    async def test_execute_processes_all_items(
        self,
        sample_dataset: BaseItemDataset,
    ) -> None:
        """Test that execute calls dataset processor."""
        fake_processor = FakeDatasetProcessor()
        use_case = EnrichDatasetWithLLMOCR(dataset_processor=fake_processor)

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert len(fake_processor.process_parallel_calls) == 1
        assert len(fake_processor.process_parallel_calls[0]) == 2

    @pytest.mark.asyncio
    async def test_execute_with_grouping(
        self,
    ) -> None:
        """Test that execute groups by schematism."""
        dataset = BaseItemDataset(
            items=[
                BaseDataItem(
                    image_path="p1.jpg",
                    metadata=BaseMetaData(
                        sample_id=1, schematism_name="A", filename="p1.jpg"
                    ),
                ),
                BaseDataItem(
                    image_path="p2.jpg",
                    metadata=BaseMetaData(
                        sample_id=2, schematism_name="B", filename="p2.jpg"
                    ),
                ),
            ]
        )
        fake_processor = FakeDatasetProcessor()
        use_case = EnrichDatasetWithLLMOCR(dataset_processor=fake_processor)

        request = EnrichWithLLMOCRRequest(
            dataset=dataset, group_by_schematism_name=True
        )
        await use_case.execute(request)

        # Should be called twice, once for each schematism
        assert len(fake_processor.process_parallel_calls) == 2
        assert len(fake_processor.process_parallel_calls[0]) == 1
        assert len(fake_processor.process_parallel_calls[1]) == 1
