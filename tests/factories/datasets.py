"""Factories for creating dataset and data item objects.

This module provides factories for creating various dataset types and their items,
including base datasets, prediction datasets, ground truth datasets, etc.
"""

from tests.factories.base import BaseFactory
from tests.factories.metadata import BaseMetaDataFactory
from tests.factories.entities import SchematismPageFactory
from notarius.schemas.data.pipeline import (
    BaseMetaData,
    BaseDataItem,
    BaseDataset,
    BaseItemDataset,
    GroundTruthDataItem,
    GroundTruthItemDataset,
    PredictionDataItem,
    PredictionItemDataset,
    AlignedSchematismsDataItem,
    AlignedItemDataset,
    EvaluationDataItem,
    EvaluationItemDataset,
)
from notarius.domain.entities.schematism import SchematismPage


class BaseDataItemFactory(BaseFactory[BaseDataItem]):
    """Factory for creating BaseDataItem instances."""

    _counter = 0

    @classmethod
    def build(
        cls,
        image_path: str | None = None,
        text: str | None = None,
        metadata: BaseMetaData | None = None,
        **kwargs
    ) -> BaseDataItem:
        """Build a BaseDataItem instance with sensible defaults.

        Args:
            image_path: Path to the image file
            text: OCR text from the image
            metadata: Metadata for the item
            **kwargs: Additional fields to pass to BaseDataItem

        Returns:
            A new BaseDataItem instance

        Example:
            item = BaseDataItemFactory.build()
            item = BaseDataItemFactory.build(text="Sample OCR text")
        """
        cls._counter += 1

        if metadata is None:
            metadata = BaseMetaDataFactory.build()

        return BaseDataItem(
            image_path=image_path or f"/path/to/test_image_{cls._counter}.jpg",
            text=text,
            metadata=metadata,
            **kwargs
        )

    @classmethod
    def build_with_text(cls, text: str | None = None) -> BaseDataItem:
        """Build item with OCR text.

        Args:
            text: OCR text to use

        Returns:
            A BaseDataItem with text set

        Example:
            item = BaseDataItemFactory.build_with_text("Sample OCR text from factory")
        """
        return cls.build(text=text or "Sample OCR text from factory")

    @classmethod
    def build_without_image(cls) -> BaseDataItem:
        """Build item without image path.

        Returns:
            A BaseDataItem with image_path=None

        Example:
            item = BaseDataItemFactory.build_without_image()
        """
        cls._counter += 1
        metadata = BaseMetaDataFactory.build()

        return BaseDataItem(
            image_path=None,
            text=None,
            metadata=metadata,
        )


class BaseDatasetFactory(BaseFactory[BaseItemDataset]):
    """Factory for creating BaseDataset instances."""

    @classmethod
    def build(
        cls,
        items: list[BaseDataItem] | int | None = None,
        **kwargs
    ) -> BaseItemDataset:
        """Build a BaseDataset instance.

        Args:
            items: Either a list of items or an int specifying how many to create
            **kwargs: Additional fields to pass to BaseDataset

        Returns:
            A new BaseItemDataset instance

        Example:
            dataset = BaseDatasetFactory.build()  # Default 3 items
            dataset = BaseDatasetFactory.build(items=5)  # 5 auto-generated items
            dataset = BaseDatasetFactory.build(items=[item1, item2])  # Custom items
        """
        # Allow passing either a list of items or a count
        if items is None:
            items_list = BaseDataItemFactory.build_batch(3)
        elif isinstance(items, int):
            items_list = BaseDataItemFactory.build_batch(items)
        else:
            items_list = items

        return BaseItemDataset(items=items_list, **kwargs)

    @classmethod
    def build_empty(cls) -> BaseItemDataset:
        """Build empty dataset.

        Returns:
            A BaseItemDataset with no items

        Example:
            dataset = BaseDatasetFactory.build_empty()
        """
        return cls.build(items=[])

    @classmethod
    def build_large(cls, size: int = 100) -> BaseItemDataset:
        """Build large dataset for performance testing.

        Args:
            size: Number of items to create

        Returns:
            A large BaseItemDataset

        Example:
            dataset = BaseDatasetFactory.build_large(size=1000)
        """
        return cls.build(items=size)

    @classmethod
    def build_with_missing_paths(
        cls,
        total: int = 5,
        missing: int = 2
    ) -> BaseItemDataset:
        """Build dataset with some items missing image paths.

        Args:
            total: Total number of items
            missing: Number of items with missing image paths

        Returns:
            A BaseItemDataset with some items missing image_path

        Example:
            dataset = BaseDatasetFactory.build_with_missing_paths(total=10, missing=3)
        """
        items = BaseDataItemFactory.build_batch(total - missing)
        items.extend([
            BaseDataItemFactory.build_without_image()
            for _ in range(missing)
        ])
        return cls.build(items=items)


class PredictionDataItemFactory(BaseFactory[PredictionDataItem]):
    """Factory for creating PredictionDataItem instances."""

    @classmethod
    def build(
        cls,
        predictions: SchematismPage | None = None,
        base_item: BaseDataItem | None = None,
        image_path: str | None = None,
        text: str | None = None,
        metadata: BaseMetaData | None = None,
        **kwargs
    ) -> PredictionDataItem:
        """Build a PredictionDataItem instance.

        Args:
            predictions: SchematismPage with predictions
            base_item: Base item to copy fields from (if provided, other args ignored)
            image_path: Path to image (used if base_item not provided)
            text: OCR text (used if base_item not provided)
            metadata: Metadata (used if base_item not provided)
            **kwargs: Additional fields to pass to PredictionDataItem

        Returns:
            A new PredictionDataItem instance

        Example:
            item = PredictionDataItemFactory.build()
            item = PredictionDataItemFactory.build(
                predictions=SchematismPageFactory.build(entry_count=5)
            )
        """
        if base_item is None:
            base_item = BaseDataItemFactory.build(
                image_path=image_path,
                text=text,
                metadata=metadata
            )

        if predictions is None:
            predictions = SchematismPageFactory.build()

        return PredictionDataItem(
            image_path=base_item.image_path,
            text=base_item.text,
            metadata=base_item.metadata,
            predictions=predictions,
            **kwargs
        )


class PredictionDatasetFactory(BaseFactory[PredictionItemDataset]):
    """Factory for creating PredictionItemDataset instances."""

    @classmethod
    def build(
        cls,
        items: list[PredictionDataItem] | int | None = None,
        **kwargs
    ) -> PredictionItemDataset:
        """Build a PredictionItemDataset instance.

        Args:
            items: Either a list of items or an int specifying how many to create
            **kwargs: Additional fields to pass to PredictionItemDataset

        Returns:
            A new PredictionItemDataset instance

        Example:
            dataset = PredictionDatasetFactory.build()
            dataset = PredictionDatasetFactory.build(items=10)
        """
        if items is None:
            items_list = PredictionDataItemFactory.build_batch(3)
        elif isinstance(items, int):
            items_list = PredictionDataItemFactory.build_batch(items)
        else:
            items_list = items

        return PredictionItemDataset(items=items_list, **kwargs)


class GroundTruthDataItemFactory(BaseFactory[GroundTruthDataItem]):
    """Factory for creating GroundTruthDataItem instances."""

    @classmethod
    def build(
        cls,
        ground_truth: SchematismPage | None = None,
        base_item: BaseDataItem | None = None,
        image_path: str | None = None,
        text: str | None = None,
        metadata: BaseMetaData | None = None,
        **kwargs
    ) -> GroundTruthDataItem:
        """Build a GroundTruthDataItem instance.

        Args:
            ground_truth: SchematismPage with ground truth
            base_item: Base item to copy fields from (if provided, other args ignored)
            image_path: Path to image (used if base_item not provided)
            text: OCR text (used if base_item not provided)
            metadata: Metadata (used if base_item not provided)
            **kwargs: Additional fields to pass to GroundTruthDataItem

        Returns:
            A new GroundTruthDataItem instance

        Example:
            item = GroundTruthDataItemFactory.build()
            item = GroundTruthDataItemFactory.build(
                ground_truth=SchematismPageFactory.build(entry_count=5)
            )
        """
        if base_item is None:
            base_item = BaseDataItemFactory.build(
                image_path=image_path,
                text=text,
                metadata=metadata
            )

        if ground_truth is None:
            ground_truth = SchematismPageFactory.build()

        return GroundTruthDataItem(
            image_path=base_item.image_path,
            text=base_item.text,
            metadata=base_item.metadata,
            ground_truth=ground_truth,
            **kwargs
        )


class GroundTruthDatasetFactory(BaseFactory[GroundTruthItemDataset]):
    """Factory for creating GroundTruthItemDataset instances."""

    @classmethod
    def build(
        cls,
        items: list[GroundTruthDataItem] | int | None = None,
        **kwargs
    ) -> GroundTruthItemDataset:
        """Build a GroundTruthItemDataset instance.

        Args:
            items: Either a list of items or an int specifying how many to create
            **kwargs: Additional fields to pass to GroundTruthItemDataset

        Returns:
            A new GroundTruthItemDataset instance

        Example:
            dataset = GroundTruthDatasetFactory.build()
            dataset = GroundTruthDatasetFactory.build(items=10)
        """
        if items is None:
            items_list = GroundTruthDataItemFactory.build_batch(3)
        elif isinstance(items, int):
            items_list = GroundTruthDataItemFactory.build_batch(items)
        else:
            items_list = items

        return GroundTruthItemDataset(items=items_list, **kwargs)


class AlignedSchematismsDataItemFactory(BaseFactory[AlignedSchematismsDataItem]):
    """Factory for creating AlignedSchematismsDataItem instances."""

    @classmethod
    def build(
        cls,
        aligned_schematism_pages: tuple[SchematismPage, SchematismPage] | None = None,
        base_item: BaseDataItem | None = None,
        image_path: str | None = None,
        text: str | None = None,
        metadata: BaseMetaData | None = None,
        **kwargs
    ) -> AlignedSchematismsDataItem:
        """Build an AlignedSchematismsDataItem instance.

        Args:
            aligned_schematism_pages: Tuple of (predictions, ground_truth) pages
            base_item: Base item to copy fields from
            image_path: Path to image
            text: OCR text
            metadata: Metadata
            **kwargs: Additional fields

        Returns:
            A new AlignedSchematismsDataItem instance

        Example:
            item = AlignedSchematismsDataItemFactory.build()
        """
        if base_item is None:
            base_item = BaseDataItemFactory.build(
                image_path=image_path,
                text=text,
                metadata=metadata
            )

        if aligned_schematism_pages is None:
            predictions = SchematismPageFactory.build()
            ground_truth = SchematismPageFactory.build()
            aligned_schematism_pages = (predictions, ground_truth)

        return AlignedSchematismsDataItem(
            image_path=base_item.image_path,
            text=base_item.text,
            metadata=base_item.metadata,
            aligned_schematism_pages=aligned_schematism_pages,
            **kwargs
        )


class AlignedDatasetFactory(BaseFactory[AlignedItemDataset]):
    """Factory for creating AlignedItemDataset instances."""

    @classmethod
    def build(
        cls,
        items: list[AlignedSchematismsDataItem] | int | None = None,
        **kwargs
    ) -> AlignedItemDataset:
        """Build an AlignedItemDataset instance.

        Args:
            items: Either a list of items or an int specifying how many to create
            **kwargs: Additional fields

        Returns:
            A new AlignedItemDataset instance

        Example:
            dataset = AlignedDatasetFactory.build()
            dataset = AlignedDatasetFactory.build(items=10)
        """
        if items is None:
            items_list = AlignedSchematismsDataItemFactory.build_batch(3)
        elif isinstance(items, int):
            items_list = AlignedSchematismsDataItemFactory.build_batch(items)
        else:
            items_list = items

        return AlignedItemDataset(items=items_list, **kwargs)


class EvaluationDataItemFactory(BaseFactory[EvaluationDataItem]):
    """Factory for creating EvaluationDataItem instances."""

    @classmethod
    def build(
        cls,
        ground_truth: SchematismPage | None = None,
        base_item: BaseDataItem | None = None,
        image_path: str | None = None,
        text: str | None = None,
        metadata: BaseMetaData | None = None,
        **kwargs
    ) -> EvaluationDataItem:
        """Build an EvaluationDataItem instance.

        Args:
            ground_truth: SchematismPage with ground truth
            base_item: Base item to copy fields from
            image_path: Path to image
            text: OCR text
            metadata: Metadata
            **kwargs: Additional fields

        Returns:
            A new EvaluationDataItem instance

        Example:
            item = EvaluationDataItemFactory.build()
        """
        if base_item is None:
            base_item = BaseDataItemFactory.build(
                image_path=image_path,
                text=text,
                metadata=metadata
            )

        if ground_truth is None:
            ground_truth = SchematismPageFactory.build()

        return EvaluationDataItem(
            image_path=base_item.image_path,
            text=base_item.text,
            metadata=base_item.metadata,
            ground_truth=ground_truth,
            **kwargs
        )


class EvaluationDatasetFactory(BaseFactory[EvaluationItemDataset]):
    """Factory for creating EvaluationItemDataset instances."""

    @classmethod
    def build(
        cls,
        items: list[EvaluationDataItem] | int | None = None,
        **kwargs
    ) -> EvaluationItemDataset:
        """Build an EvaluationItemDataset instance.

        Args:
            items: Either a list of items or an int specifying how many to create
            **kwargs: Additional fields

        Returns:
            A new EvaluationItemDataset instance

        Example:
            dataset = EvaluationDatasetFactory.build()
            dataset = EvaluationDatasetFactory.build(items=10)
        """
        if items is None:
            items_list = EvaluationDataItemFactory.build_batch(3)
        elif isinstance(items, int):
            items_list = EvaluationDataItemFactory.build_batch(items)
        else:
            items_list = items

        return EvaluationItemDataset(items=items_list, **kwargs)
