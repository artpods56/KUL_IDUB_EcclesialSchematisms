"""Factories for creating metadata objects.

This module provides factories for creating metadata and context objects
used throughout the pipeline.
"""

from tests.factories.base import BaseFactory
from notarius.schemas.data.pipeline import BaseMetaData
from notarius.domain.entities.schematism import PageContext


class BaseMetaDataFactory(BaseFactory[BaseMetaData]):
    """Factory for creating BaseMetaData instances."""

    _counter = 0

    @classmethod
    def build(
        cls,
        sample_id: int | None = None,
        schematism_name: str | None = None,
        filename: str | None = None,
        **kwargs
    ) -> BaseMetaData:
        """Build a BaseMetaData instance with sensible defaults.

        Args:
            sample_id: Sample ID (auto-incremented if not provided)
            schematism_name: Name of the schematism
            filename: Filename of the source file
            **kwargs: Additional fields to pass to BaseMetaData

        Returns:
            A new BaseMetaData instance

        Example:
            metadata = BaseMetaDataFactory.build()
            metadata = BaseMetaDataFactory.build(sample_id=42, schematism_name="kraków_1930")
        """
        cls._counter += 1

        return BaseMetaData(
            sample_id=sample_id if sample_id is not None else cls._counter,
            schematism_name=schematism_name or f"test_schematism_{cls._counter}",
            filename=filename or f"test_file_{cls._counter}.jpg",
            **kwargs
        )

    @classmethod
    def build_with_schematism(cls, schematism_name: str, count: int = 1) -> list[BaseMetaData]:
        """Build multiple metadata items for the same schematism.

        Args:
            schematism_name: The schematism name to use for all items
            count: Number of metadata items to create

        Returns:
            A list of BaseMetaData instances with sequential sample_ids

        Example:
            items = BaseMetaDataFactory.build_with_schematism("kraków_1930", count=5)
            # All items have schematism_name="kraków_1930" but different sample_ids
        """
        return [
            cls.build(schematism_name=schematism_name)
            for _ in range(count)
        ]


class PageContextFactory(BaseFactory[PageContext]):
    """Factory for creating PageContext instances."""

    @classmethod
    def build(
        cls,
        summary: str | None = None,
        note: str | None = None,
        active_deanery: str | None = None,
        last_page_number: str | None = None,
        **kwargs
    ) -> PageContext:
        """Build a PageContext instance.

        Args:
            summary: Short summary of the page
            note: Specific note about the page
            active_deanery: The deanery active at the end of this page
            last_page_number: The last processed page number
            **kwargs: Additional fields to pass to PageContext

        Returns:
            A new PageContext instance

        Example:
            context = PageContextFactory.build()
            context = PageContextFactory.build(active_deanery="Krakowski")
        """
        return PageContext(
            summary=summary,
            note=note,
            active_deanery=active_deanery,
            last_page_number=last_page_number,
            **kwargs
        )

    @classmethod
    def build_empty(cls) -> PageContext:
        """Build an empty PageContext with all None values.

        Returns:
            A PageContext with all fields set to None

        Example:
            context = PageContextFactory.build_empty()
        """
        return cls.build()

    @classmethod
    def build_with_deanery(cls, deanery: str) -> PageContext:
        """Build a PageContext with an active deanery.

        Args:
            deanery: The active deanery name

        Returns:
            A PageContext with active_deanery set

        Example:
            context = PageContextFactory.build_with_deanery("Krakowski")
        """
        return cls.build(active_deanery=deanery)
