"""Tests for entity factories."""

import pytest

from tests.factories.entities import SchematismEntryFactory, SchematismPageFactory
from notarius.domain.entities.schematism import SchematismEntry, SchematismPage


class TestSchematismEntryFactory:
    """Tests for SchematismEntryFactory."""

    def test_build_creates_entry_with_defaults(self):
        """Test that build() creates an entry with sensible defaults."""
        entry = SchematismEntryFactory.build()

        assert isinstance(entry, SchematismEntry)
        assert entry.parish is not None
        assert entry.deanery is not None
        assert entry.dedication is not None
        assert entry.building_material == "mur."

    def test_build_with_custom_values(self):
        """Test that build() respects custom values."""
        entry = SchematismEntryFactory.build(
            parish="Kraków",
            deanery="Krakowski",
            dedication="St. Mary",
            building_material="lig."
        )

        assert entry.parish == "Kraków"
        assert entry.deanery == "Krakowski"
        assert entry.dedication == "St. Mary"
        assert entry.building_material == "lig."

    def test_build_empty_creates_all_none(self):
        """Test that build_empty() creates entry with all None values."""
        entry = SchematismEntryFactory.build_empty()

        assert entry.parish is None
        assert entry.deanery is None
        assert entry.dedication is None
        assert entry.building_material is None

    def test_build_complete_fills_all_fields(self):
        """Test that build_complete() fills all fields."""
        entry = SchematismEntryFactory.build_complete()

        assert entry.parish is not None
        assert entry.deanery is not None
        assert entry.dedication is not None
        assert entry.building_material is not None

    def test_build_multiple_entries_creates_list(self):
        """Test that build_multiple_entries() creates multiple entries."""
        entries = SchematismEntryFactory.build_multiple_entries(count=5)

        assert len(entries) == 5
        assert all(isinstance(e, SchematismEntry) for e in entries)

    def test_build_multiple_entries_with_same_deanery(self):
        """Test that build_multiple_entries() can create entries with same deanery."""
        entries = SchematismEntryFactory.build_multiple_entries(
            count=3,
            same_deanery=True,
            deanery="Krakowski"
        )

        assert len(entries) == 3
        assert all(e.deanery == "Krakowski" for e in entries)
        # But parishes should be different
        parishes = [e.parish for e in entries]
        assert len(set(parishes)) == 3

    def test_build_batch_creates_multiple_entries(self):
        """Test that build_batch() creates multiple entries."""
        entries = SchematismEntryFactory.build_batch(10)

        assert len(entries) == 10
        assert all(isinstance(e, SchematismEntry) for e in entries)

    def test_counter_increments(self):
        """Test that internal counter increments."""
        SchematismEntryFactory.reset_counter()
        entry1 = SchematismEntryFactory.build()
        entry2 = SchematismEntryFactory.build()

        # Entries should have different default values due to counter
        assert entry1.parish != entry2.parish


class TestSchematismPageFactory:
    """Tests for SchematismPageFactory."""

    def test_build_creates_page_with_defaults(self):
        """Test that build() creates a page with sensible defaults."""
        page = SchematismPageFactory.build()

        assert isinstance(page, SchematismPage)
        assert page.page_number is not None
        assert len(page.entries) == 3  # default entry_count
        assert all(isinstance(e, SchematismEntry) for e in page.entries)

    def test_build_with_custom_entry_count(self):
        """Test that build() respects custom entry_count."""
        page = SchematismPageFactory.build(entry_count=10)

        assert len(page.entries) == 10

    def test_build_with_custom_entries(self):
        """Test that build() accepts custom entries list."""
        entries = [
            SchematismEntryFactory.build(parish="A"),
            SchematismEntryFactory.build(parish="B"),
        ]
        page = SchematismPageFactory.build(entries=entries)

        assert len(page.entries) == 2
        assert page.entries[0].parish == "A"
        assert page.entries[1].parish == "B"

    def test_build_empty_creates_no_entries(self):
        """Test that build_empty() creates page with no entries."""
        page = SchematismPageFactory.build_empty()

        assert len(page.entries) == 0

    def test_build_large_creates_many_entries(self):
        """Test that build_large() creates page with many entries."""
        page = SchematismPageFactory.build_large(entry_count=100)

        assert len(page.entries) == 100

    def test_build_with_context(self):
        """Test that build_with_context() creates page with context."""
        page = SchematismPageFactory.build_with_context(
            active_deanery="Krakowski",
            summary="Test summary"
        )

        assert page.context is not None
        assert page.context.active_deanery == "Krakowski"
        assert page.context.summary == "Test summary"

    def test_build_with_same_deanery(self):
        """Test that build_with_same_deanery() creates entries with same deanery."""
        page = SchematismPageFactory.build_with_same_deanery(
            deanery="Krakowski",
            entry_count=5
        )

        assert len(page.entries) == 5
        assert all(e.deanery == "Krakowski" for e in page.entries)

    def test_build_batch_creates_multiple_pages(self):
        """Test that build_batch() creates multiple pages."""
        pages = SchematismPageFactory.build_batch(5)

        assert len(pages) == 5
        assert all(isinstance(p, SchematismPage) for p in pages)
