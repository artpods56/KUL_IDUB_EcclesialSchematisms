from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, Dict, Generic, Literal, TypeVar, Self

from PIL import Image
from pydantic import BaseModel, Field

from notarius.schemas.data.metrics import PageDataMetrics
from notarius.domain.entities.schematism import SchematismPage, SchematismEntry

# Module-level TypeVar for pickle/dill compatibility
# PEP 695 syntax (class Foo[T]) creates scoped type params that can't be pickled
ItemT = TypeVar("ItemT", bound="BaseDataItem")

PageDataSourceField = Literal[
    "ground_truth",
    "source_ground_truth",
    "lmv3_prediction",
    "llm_prediction",
    "parsed_prediction",
]


class BaseMetaData(BaseModel):
    sample_id: int = Field(description="Sample ID")
    schematism_name: str = Field(description="Schematism name")
    filename: str = Field(description="Schematism filename")


class BaseDataItem(BaseModel):
    image_path: str | None = Field(description="Path to the saved image.")
    text: str | None = Field(default=None, description="OCR text extracted from image.")

    metadata: BaseMetaData | None = Field(
        default=None, description="Metadata of the lmv3_dataset item"
    )

    class Config:
        arbitrary_types_allowed = True


class HasGroundTruthMixin(BaseModel):
    ground_truth: SchematismPage = Field(description="Ground truth page")


class HasPredictionsMixin(BaseModel):
    predictions: SchematismPage | None = Field(description="Prediction page")


class HasAlignedPagesMixin(BaseModel):
    aligned_schematism_pages: tuple[SchematismPage, SchematismPage] = Field(
        description="Tuple of aligned predictions and ground truth"
    )


class GroundTruthDataItem(BaseDataItem, HasGroundTruthMixin):
    pass


class PredictionDataItem(BaseDataItem, HasPredictionsMixin):
    pass


class AlignedSchematismsDataItem(BaseDataItem, HasAlignedPagesMixin):
    pass


class BaseDataset(BaseModel, Generic[ItemT]):
    """Generic dataset container that can be pickled/dill serialized.

    Note: Use concrete subclasses (GroundTruthDataset, PredictionDataset, etc.)
    instead of BaseDataset[SomeType] for pickle compatibility.
    """

    items: Sequence[ItemT] = Field(description="List of items")

    def group_by_schematism(self) -> Iterator[tuple[str, Self]]:
        groups: dict[str, list[ItemT]] = defaultdict(list)

        for item in self.items:
            if item.metadata is None:
                raise ValueError("Metadata is required for grouping")
            groups[item.metadata.schematism_name].append(item)

        for key, items in groups.items():
            yield (key, self.__class__(items=items))


# Concrete subclasses for pickle/dill serialization compatibility
# Parameterized generics like BaseDataset[GroundTruthDataItem] cannot be pickled
# because they don't exist as module-level attributes


class BaseItemDataset(BaseDataset[BaseDataItem]):
    """Dataset containing base data items (no ground truth or predictions)."""

    pass


class GroundTruthItemDataset(BaseDataset[GroundTruthDataItem]):
    """Dataset containing ground truth items."""

    pass


class PredictionItemDataset(BaseDataset[PredictionDataItem]):
    """Dataset containing prediction items."""

    @classmethod
    def from_base_dataset(cls, base_dataset: BaseDataset[BaseDataItem]) -> Self:
        return cls(
            items=[
                PredictionDataItem(
                    predictions=None,
                    image_path=item.image_path,
                    text=item.text,
                    metadata=item.metadata,
                )
                for item in base_dataset.items
            ]
        )


class AlignedItemDataset(BaseDataset[AlignedSchematismsDataItem]):
    """Dataset containing aligned prediction/ground truth items."""

    pass


class FlatSchematismEntryWithMetadata(BaseMetaData, SchematismEntry):
    """Flat schematism entry with metadata."""


class FlatSchematismAlignedEntryWithMetadata(BaseMetaData):
    """Flat schematism entry with metadata and aligned ground truth."""

    deanery_a: str | None = None
    deanery_b: str | None = None
    parish_a: str | None = None
    parish_b: str | None = None
    dedication_a: str | None = None
    dedication_b: str | None = None
    building_material_a: str | None = None
    building_material_b: str | None = None
