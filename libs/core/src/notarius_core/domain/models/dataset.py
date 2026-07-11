from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel, Field

ItemT = TypeVar("ItemT", bound="BaseDataItem")


class BaseMetaData(BaseModel):
    sample_id: int = Field(description="Sample ID")
    source_name: str | None = Field(default=None, description="Source name")
    filename: str | None = Field(default=None, description="Source filename")


class BaseDataItem(BaseModel):
    image_path: str | None = Field(default=None, description="Path to the saved image.")
    text: str | None = Field(default=None, description="OCR text extracted from image.")
    metadata: BaseMetaData | None = Field(default=None, description="Item metadata.")

    class Config:
        arbitrary_types_allowed = True


class GroundTruthDataItem(BaseDataItem):
    ground_truth: BaseModel | dict[str, Any] = Field(description="Ground truth output.")


class PredictionDataItem(BaseDataItem):
    predictions: BaseModel | dict[str, Any] | None = Field(
        default=None,
        description="Prediction output.",
    )


class BaseDataset(BaseModel, Generic[ItemT]):
    """Generic dataset container that can be serialized by concrete subclasses."""

    items: Sequence[ItemT] = Field(description="List of items")

    def group_by_source(self) -> Iterator[tuple[str, Self]]:
        groups: dict[str, list[ItemT]] = defaultdict(list)
        for item in self.items:
            if item.metadata is None:
                raise ValueError("Metadata is required for grouping")
            groups[item.metadata.source_name or "unknown"].append(item)
        for key, items in groups.items():
            yield key, self.__class__(items=items)


class BaseItemDataset(BaseDataset[BaseDataItem]):
    pass


class GroundTruthItemDataset(BaseDataset[GroundTruthDataItem]):
    pass


class PredictionItemDataset(BaseDataset[PredictionDataItem]):
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
