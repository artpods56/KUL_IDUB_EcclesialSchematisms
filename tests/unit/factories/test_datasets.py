"""Tests for dataset factories."""

import pytest

from tests.factories.datasets import (
    BaseDataItemFactory,
    BaseDatasetFactory,
    PredictionDataItemFactory,
    PredictionDatasetFactory,
    GroundTruthDataItemFactory,
    GroundTruthDatasetFactory,
)
from notarius.schemas.data.pipeline import (
    BaseDataItem,
    BaseItemDataset,
    PredictionDataItem,
    PredictionItemDataset,
    GroundTruthDataItem,
    GroundTruthItemDataset,
)


class TestBaseDataItemFactory:
    """Tests for BaseDataItemFactory."""

    def test_build_creates_item_with_defaults(self):
        """Test that build() creates an item with sensible defaults."""
        item = BaseDataItemFactory.build()

        assert isinstance(item, BaseDataItem)
        assert item.image_path is not None
        assert item.metadata is not None
        assert item.metadata.sample_id > 0

    def test_build_with_text(self):
        """Test that build_with_text() creates item with text."""
        item = BaseDataItemFactory.build_with_text("Custom OCR text")

        assert item.text == "Custom OCR text"

    def test_build_without_image(self):
        """Test that build_without_image() creates item with no image path."""
        item = BaseDataItemFactory.build_without_image()

        assert item.image_path is None

    def test_build_batch_creates_multiple_items(self):
        """Test that build_batch() creates multiple items."""
        items = BaseDataItemFactory.build_batch(5)

        assert len(items) == 5
        assert all(isinstance(i, BaseDataItem) for i in items)


class TestBaseDatasetFactory:
    """Tests for BaseDatasetFactory."""

    def test_build_with_default_size(self):
        """Test that build() creates dataset with default size."""
        dataset = BaseDatasetFactory.build()

        assert isinstance(dataset, BaseItemDataset)
        assert len(dataset.items) == 3  # default

    def test_build_with_int_creates_items(self):
        """Test that build(items=int) creates that many items."""
        dataset = BaseDatasetFactory.build(items=10)

        assert len(dataset.items) == 10

    def test_build_with_item_list(self):
        """Test that build(items=list) uses provided items."""
        items = [
            BaseDataItemFactory.build(),
            BaseDataItemFactory.build(),
        ]
        dataset = BaseDatasetFactory.build(items=items)

        assert len(dataset.items) == 2
        assert dataset.items == items

    def test_build_empty(self):
        """Test that build_empty() creates empty dataset."""
        dataset = BaseDatasetFactory.build_empty()

        assert len(dataset.items) == 0

    def test_build_large(self):
        """Test that build_large() creates large dataset."""
        dataset = BaseDatasetFactory.build_large(size=100)

        assert len(dataset.items) == 100

    def test_build_with_missing_paths(self):
        """Test that build_with_missing_paths() creates mixed dataset."""
        dataset = BaseDatasetFactory.build_with_missing_paths(total=10, missing=3)

        assert len(dataset.items) == 10

        # Count items without image paths
        missing_count = sum(1 for item in dataset.items if item.image_path is None)
        assert missing_count == 3


class TestPredictionDataItemFactory:
    """Tests for PredictionDataItemFactory."""

    def test_build_creates_item_with_predictions(self):
        """Test that build() creates item with predictions."""
        item = PredictionDataItemFactory.build()

        assert isinstance(item, PredictionDataItem)
        assert item.predictions is not None
        assert len(item.predictions.entries) > 0

    def test_build_with_custom_base_item(self):
        """Test that build() accepts custom base item."""
        base = BaseDataItemFactory.build(text="Custom text")
        item = PredictionDataItemFactory.build(base_item=base)

        assert item.text == "Custom text"

    def test_build_batch_creates_multiple_items(self):
        """Test that build_batch() creates multiple items."""
        items = PredictionDataItemFactory.build_batch(5)

        assert len(items) == 5
        assert all(isinstance(i, PredictionDataItem) for i in items)


class TestPredictionDatasetFactory:
    """Tests for PredictionDatasetFactory."""

    def test_build_with_default_size(self):
        """Test that build() creates dataset with default size."""
        dataset = PredictionDatasetFactory.build()

        assert isinstance(dataset, PredictionItemDataset)
        assert len(dataset.items) == 3

    def test_build_with_int_creates_items(self):
        """Test that build(items=int) creates that many items."""
        dataset = PredictionDatasetFactory.build(items=7)

        assert len(dataset.items) == 7
        assert all(isinstance(i, PredictionDataItem) for i in dataset.items)

    def test_items_have_predictions(self):
        """Test that all items have predictions."""
        dataset = PredictionDatasetFactory.build(items=5)

        assert all(item.predictions is not None for item in dataset.items)


class TestGroundTruthDataItemFactory:
    """Tests for GroundTruthDataItemFactory."""

    def test_build_creates_item_with_ground_truth(self):
        """Test that build() creates item with ground truth."""
        item = GroundTruthDataItemFactory.build()

        assert isinstance(item, GroundTruthDataItem)
        assert item.ground_truth is not None
        assert len(item.ground_truth.entries) > 0

    def test_build_with_custom_base_item(self):
        """Test that build() accepts custom base item."""
        base = BaseDataItemFactory.build(text="Custom text")
        item = GroundTruthDataItemFactory.build(base_item=base)

        assert item.text == "Custom text"


class TestGroundTruthDatasetFactory:
    """Tests for GroundTruthDatasetFactory."""

    def test_build_with_default_size(self):
        """Test that build() creates dataset with default size."""
        dataset = GroundTruthDatasetFactory.build()

        assert isinstance(dataset, GroundTruthItemDataset)
        assert len(dataset.items) == 3

    def test_build_with_int_creates_items(self):
        """Test that build(items=int) creates that many items."""
        dataset = GroundTruthDatasetFactory.build(items=8)

        assert len(dataset.items) == 8
        assert all(isinstance(i, GroundTruthDataItem) for i in dataset.items)

    def test_items_have_ground_truth(self):
        """Test that all items have ground truth."""
        dataset = GroundTruthDatasetFactory.build(items=5)

        assert all(item.ground_truth is not None for item in dataset.items)
