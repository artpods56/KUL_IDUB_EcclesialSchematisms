from typing import Self

from pydantic import BaseModel, Field

from notarius_core.domain.models.dataset import (
    BaseDataItem,
    BaseDataset,
    BaseMetaData as CoreBaseMetaData,
)
from notarius_schematisms.domain.models import SchematismEntry, SchematismPage


class BaseMetaData(CoreBaseMetaData):
    schematism_name: str = Field(description="Schematism name")
    filename: str = Field(description="Schematism filename")


class HasGroundTruthMixin(BaseModel):
    ground_truth: SchematismPage = Field(description="Ground truth page")


class HasPredictionsMixin(BaseModel):
    predictions: SchematismPage | None = Field(description="Prediction page")


class HasAlignedPagesMixin(BaseModel):
    aligned_schematism_pages: tuple[SchematismPage, SchematismPage] = Field(
        description="Tuple of aligned predictions and ground truth"
    )


class GroundTruthDataItem(BaseDataItem, HasGroundTruthMixin):
    metadata: BaseMetaData | None = None


class EvaluationDataItem(GroundTruthDataItem):
    pass


class PredictionDataItem(BaseDataItem, HasPredictionsMixin):
    metadata: BaseMetaData | None = None


class AlignedSchematismsDataItem(BaseDataItem, HasAlignedPagesMixin):
    metadata: BaseMetaData | None = None


class BaseItemDataset(BaseDataset[BaseDataItem]):
    pass


class GroundTruthItemDataset(BaseDataset[GroundTruthDataItem]):
    pass


class EvaluationItemDataset(BaseDataset[EvaluationDataItem]):
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


class AlignedItemDataset(BaseDataset[AlignedSchematismsDataItem]):
    pass


class FlatSchematismEntryWithMetadata(BaseMetaData, SchematismEntry):
    pass


class FlatSchematismAlignedEntryWithMetadata(BaseMetaData):
    deanery_a: str | None = None
    deanery_b: str | None = None
    parish_a: str | None = None
    parish_b: str | None = None
    dedication_a: str | None = None
    dedication_b: str | None = None
    building_material_a: str | None = None
    building_material_b: str | None = None
