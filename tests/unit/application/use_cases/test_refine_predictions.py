"""Tests for RefinePredictionsWithLLM use case."""

import pytest
import asyncio
from typing import Any, Sequence
from collections.abc import Sequence

from notarius.application.use_cases.inference.refine_predictions_using_llm import (
    RefinePredictionsWithLLM,
    RefinePredictionsRequest,
    RefinePredictionsResponse,
)
from notarius.application.services import DatasetProcessor
from notarius.schemas.data.pipeline import (
    PredictionDataItem,
    PredictionItemDataset,
    BaseMetaData,
)
from notarius.domain.entities.schematism import SchematismPage


class FakeSequenceDatasetProcessor(DatasetProcessor[PredictionDataItem, Any]):
    """Fake dataset processor for sequential processing."""

    def __init__(self):
        self.process_sequence_calls: list[Sequence[PredictionDataItem]] = []

    def process_sequence(
        self, items: Sequence[PredictionDataItem]
    ) -> Sequence[PredictionDataItem]:
        self.process_sequence_calls.append(items)
        return items


@pytest.fixture
def sample_prediction_dataset() -> PredictionItemDataset:
    """Create a sample prediction dataset."""
    return PredictionItemDataset(
        items=[
            PredictionDataItem(
                image_path="p1.jpg",
                text="OCR 1",
                predictions=SchematismPage(page_number="1", entries=[]),
                metadata=BaseMetaData(
                    sample_id=1, schematism_name="A", filename="p1.jpg"
                ),
            ),
            PredictionDataItem(
                image_path="p2.jpg",
                text="OCR 2",
                predictions=SchematismPage(page_number="2", entries=[]),
                metadata=BaseMetaData(
                    sample_id=2, schematism_name="B", filename="p2.jpg"
                ),
            ),
        ]
    )


class TestRefinePredictionsWithLLM:
    """Test suite for RefinePredictionsWithLLM use case."""

    @pytest.mark.asyncio
    async def test_execute_processes_all_items(
        self, sample_prediction_dataset: PredictionItemDataset
    ) -> None:
        """Test that execute calls dataset processor."""
        fake_processor = FakeSequenceDatasetProcessor()
        use_case = RefinePredictionsWithLLM(dataset_processor=fake_processor)

        request = RefinePredictionsRequest(dataset=sample_prediction_dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        # If grouping is enabled (default), it calls process_sequence once for each group
        assert len(fake_processor.process_sequence_calls) == 2

    @pytest.mark.asyncio
    async def test_execute_without_grouping(
        self, sample_prediction_dataset: PredictionItemDataset
    ) -> None:
        """Test that execute works without grouping."""
        fake_processor = FakeSequenceDatasetProcessor()
        use_case = RefinePredictionsWithLLM(dataset_processor=fake_processor)

        request = RefinePredictionsRequest(
            dataset=sample_prediction_dataset, group_by_schematism_name=False
        )
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert len(fake_processor.process_sequence_calls) == 1
        assert len(fake_processor.process_sequence_calls[0]) == 2
