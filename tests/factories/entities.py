"""Factories for creating domain entity objects.

This module provides factories for creating domain entities like
SchematismEntry and SchematismPage with sensible defaults.
"""

from tests.factories.base import BaseFactory
from tests.factories.metadata import PageContextFactory
from notarius.domain.entities.schematism import SchematismEntry, SchematismPage, PageContext


class SchematismEntryFactory(BaseFactory[SchematismEntry]):
    """Factory for creating SchematismEntry instances."""

    _counter = 0

    @classmethod
    def build(
        cls,
        parish: str | None = None,
        deanery: str | None = None,
        dedication: str | None = None,
        building_material: str | None = None,
        **kwargs
    ) -> SchematismEntry:
        """Build a SchematismEntry instance with sensible defaults.

        Args:
            parish: Parish name
            deanery: Deanery name
            dedication: Church dedication/patron saint
            building_material: Building material (e.g., 'mur.', 'lig.')
            **kwargs: Additional fields to pass to SchematismEntry

        Returns:
            A new SchematismEntry instance

        Example:
            entry = SchematismEntryFactory.build()
            entry = SchematismEntryFactory.build(parish="Kraków", deanery="Krakowski")
        """
        cls._counter += 1

        return SchematismEntry(
            parish=parish or f"Test Parish {cls._counter}",
            deanery=deanery or f"Test Deanery {cls._counter}",
            dedication=dedication or f"St. Test {cls._counter}",
            building_material=building_material or "mur.",
            **kwargs
        )

    @classmethod
    def build_empty(cls) -> SchematismEntry:
        """Build an empty entry with all None values.

        Returns:
            A SchematismEntry with all fields set to None

        Example:
            entry = SchematismEntryFactory.build_empty()
        """
        return SchematismEntry(
            parish=None,
            deanery=None,
            dedication=None,
            building_material=None,
        )

    @classmethod
    def build_complete(cls) -> SchematismEntry:
        """Build a complete entry with all fields populated.

        Returns:
            A SchematismEntry with all fields set

        Example:
            entry = SchematismEntryFactory.build_complete()
        """
        cls._counter += 1
        return cls.build(
            parish=f"Complete Parish {cls._counter}",
            deanery=f"Complete Deanery {cls._counter}",
            dedication=f"St. Complete {cls._counter}",
            building_material="mur."
        )

    @classmethod
    def build_multiple_entries(
        cls,
        count: int = 3,
        same_deanery: bool = False,
        deanery: str | None = None
    ) -> list[SchematismEntry]:
        """Build a list of diverse entries.

        Args:
            count: Number of entries to create
            same_deanery: Whether all entries should have the same deanery
            deanery: Optional deanery name to use (if same_deanery=True)

        Returns:
            A list of SchematismEntry instances

        Example:
            entries = SchematismEntryFactory.build_multiple_entries(5)
            entries = SchematismEntryFactory.build_multiple_entries(
                count=3,
                same_deanery=True,
                deanery="Krakowski"
            )
        """
        if same_deanery:
            deanery_name = deanery or f"Test Deanery {cls._counter}"
            return [
                cls.build(
                    deanery=deanery_name,
                    parish=f"Parish {i}",
                    dedication=f"St. {chr(65 + i)}"  # St. A, St. B, ...
                )
                for i in range(count)
            ]
        else:
            return cls.build_batch(count)


class SchematismPageFactory(BaseFactory[SchematismPage]):
    """Factory for creating SchematismPage instances."""

    _counter = 0

    @classmethod
    def build(
        cls,
        page_number: str | None = None,
        entries: list[SchematismEntry] | None = None,
        context: PageContext | None = None,
        entry_count: int = 3,
        **kwargs
    ) -> SchematismPage:
        """Build a SchematismPage instance with sensible defaults.

        Args:
            page_number: Page number as string
            entries: List of entries (auto-generated if not provided)
            context: Page context (optional)
            entry_count: Number of entries to generate if entries not provided
            **kwargs: Additional fields to pass to SchematismPage

        Returns:
            A new SchematismPage instance

        Example:
            page = SchematismPageFactory.build()
            page = SchematismPageFactory.build(page_number="42", entry_count=5)
            page = SchematismPageFactory.build(entries=[entry1, entry2])
        """
        cls._counter += 1

        if entries is None:
            entries = SchematismEntryFactory.build_batch(entry_count)

        return SchematismPage(
            page_number=page_number or str(cls._counter),
            entries=entries,
            context=context,
            **kwargs
        )

    @classmethod
    def build_empty(cls) -> SchematismPage:
        """Build a page with no entries.

        Returns:
            A SchematismPage with an empty entries list

        Example:
            page = SchematismPageFactory.build_empty()
        """
        return cls.build(entries=[], entry_count=0)

    @classmethod
    def build_large(cls, entry_count: int = 50) -> SchematismPage:
        """Build a page with many entries.

        Args:
            entry_count: Number of entries to create

        Returns:
            A SchematismPage with many entries

        Example:
            page = SchematismPageFactory.build_large(entry_count=100)
        """
        return cls.build(entry_count=entry_count)

    @classmethod
    def build_with_context(
        cls,
        active_deanery: str | None = None,
        summary: str | None = None,
        **kwargs
    ) -> SchematismPage:
        """Build a page with context information.

        Args:
            active_deanery: The active deanery for this page
            summary: Summary of the page
            **kwargs: Additional fields to pass to build()

        Returns:
            A SchematismPage with context set

        Example:
            page = SchematismPageFactory.build_with_context(
                active_deanery="Krakowski",
                summary="Page contains 5 parishes"
            )
        """
        context = PageContextFactory.build(
            active_deanery=active_deanery,
            summary=summary
        )
        return cls.build(context=context, **kwargs)

    @classmethod
    def build_with_same_deanery(cls, deanery: str, entry_count: int = 3) -> SchematismPage:
        """Build a page where all entries have the same deanery.

        Args:
            deanery: The deanery name for all entries
            entry_count: Number of entries to create

        Returns:
            A SchematismPage with all entries having the same deanery

        Example:
            page = SchematismPageFactory.build_with_same_deanery(
                "Krakowski",
                entry_count=5
            )
        """
        entries = SchematismEntryFactory.build_multiple_entries(
            count=entry_count,
            same_deanery=True,
            deanery=deanery
        )
        return cls.build(entries=entries)
