from typing import TypedDict, NotRequired

from PIL.Image import Image
from pydantic import BaseModel, Field


class ETLSpecificDatasetFields(BaseModel):
    sample_id: str = Field(
        default="sample_id", description="Sample id of the lmv3_dataset."
    )
    ground_truth_source: str = Field(
        default="parsed", description="Ground truth source of the dataset."
    )


class BaseHuggingFaceDatasetSchema(BaseModel):
    image: str = Field(default="image", description="Image used for prediction.")
    source: str = Field(default="source", description="Ground truth data.")
    parsed: str = Field(default="parsed", description="Source truth data.")
    schematism_name: str = Field(
        default="schematism_name", description="Name of schematism."
    )
    filename: str = Field(default="filename", description="Name of file.")


SchematismEntry = TypedDict(
    "SchematismEntry",
    {"parish": str, "deanery": str, "dedication": str, "building_material": str},
)

SchematismEntriesList = list[SchematismEntry]


SchematismPage = TypedDict(
    "SchematismPage", {"entries": SchematismEntriesList, "page_number": int | None}
)


SchematismsDatasetItem = TypedDict(
    "SchematismsDatasetItem",
    {
        "sample_id": NotRequired[int],
        "image": Image,
        "source": SchematismPage,
        "parsed": SchematismPage,
        "schematism_name": str,
        "filename": str,
    },
)

SourceDatasetGenerationItem = TypedDict(
    "SourceDatasetGenerationItem",
    {
        "sample_id": int,
        "filename": str,
        "schematism_name": str,
        "source": SchematismPage,
    },
)

SourceDatasetGenerationResult = TypedDict(
    "SourceDatasetGenerationResult",
    {
        "schematism_name": str,
        "generated_at": str,
        "total_records": int,
        "records": list[SourceDatasetGenerationItem],
    },
)


class FlatPredictionDatasetItem(TypedDict):
    sample_id: str
    filename: str
    schematism_name: str
    deanery: str
    parish: str
    dedication: str
    building_material: str
