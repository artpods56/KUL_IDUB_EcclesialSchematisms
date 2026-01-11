"""Tests for metadata factories."""

import pytest

from tests.factories.metadata import BaseMetaDataFactory, PageContextFactory
from notarius.schemas.data.pipeline import BaseMetaData
from notarius.domain.entities.schematism import PageContext


class TestBaseMetaDataFactory:
    """Tests for BaseMetaDataFactory."""

    def test_build_creates_metadata_with_defaults(self):
        """Test that build() creates metadata with sensible defaults."""
        metadata = BaseMetaDataFactory.build()

        assert isinstance(metadata, BaseMetaData)
        assert metadata.sample_id > 0
        assert metadata.schematism_name is not None
        assert metadata.filename is not None

    def test_build_with_custom_values(self):
        """Test that build() respects custom values."""
        metadata = BaseMetaDataFactory.build(
            sample_id=42,
            schematism_name="kraków_1930",
            filename="page_001.jpg"
        )

        assert metadata.sample_id == 42
        assert metadata.schematism_name == "kraków_1930"
        assert metadata.filename == "page_001.jpg"

    def test_build_with_schematism(self):
        """Test that build_with_schematism() creates multiple metadata items."""
        items = BaseMetaDataFactory.build_with_schematism("kraków_1930", count=5)

        assert len(items) == 5
        assert all(item.schematism_name == "kraków_1930" for item in items)
        # Sample IDs should be different
        sample_ids = [item.sample_id for item in items]
        assert len(set(sample_ids)) == 5

    def test_build_batch_creates_multiple_items(self):
        """Test that build_batch() creates multiple items."""
        items = BaseMetaDataFactory.build_batch(10)

        assert len(items) == 10
        assert all(isinstance(item, BaseMetaData) for item in items)

    def test_counter_increments(self):
        """Test that internal counter increments."""
        BaseMetaDataFactory.reset_counter()
        meta1 = BaseMetaDataFactory.build()
        meta2 = BaseMetaDataFactory.build()

        assert meta1.sample_id < meta2.sample_id


class TestPageContextFactory:
    """Tests for PageContextFactory."""

    def test_build_creates_context_with_defaults(self):
        """Test that build() creates context with None defaults."""
        context = PageContextFactory.build()

        assert isinstance(context, PageContext)
        # Defaults should be None
        assert context.summary is None
        assert context.note is None
        assert context.active_deanery is None
        assert context.last_page_number is None

    def test_build_with_custom_values(self):
        """Test that build() accepts custom values."""
        context = PageContextFactory.build(
            summary="Page summary",
            note="Special note",
            active_deanery="Krakowski",
            last_page_number="42"
        )

        assert context.summary == "Page summary"
        assert context.note == "Special note"
        assert context.active_deanery == "Krakowski"
        assert context.last_page_number == "42"

    def test_build_empty_creates_all_none(self):
        """Test that build_empty() creates context with all None values."""
        context = PageContextFactory.build_empty()

        assert context.summary is None
        assert context.note is None
        assert context.active_deanery is None
        assert context.last_page_number is None

    def test_build_with_deanery(self):
        """Test that build_with_deanery() sets active_deanery."""
        context = PageContextFactory.build_with_deanery("Krakowski")

        assert context.active_deanery == "Krakowski"

    def test_build_batch_creates_multiple_contexts(self):
        """Test that build_batch() creates multiple contexts."""
        contexts = PageContextFactory.build_batch(5)

        assert len(contexts) == 5
        assert all(isinstance(c, PageContext) for c in contexts)
