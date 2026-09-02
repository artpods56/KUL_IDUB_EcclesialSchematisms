from typing import cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from grafy_core.artifacts import ArtifactTypeKey, ArtifactTypeSpec, JsonObject


class StructuredExtractionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_index: StrictInt = Field(ge=0)
    source_image_id: UUID
    source_filename: StrictStr
    structured_value: JsonObject
    model: StrictStr = Field(min_length=1)
    response_id: StrictStr | None = None
    finish_reason: StrictStr | None = None
    usage: JsonObject = Field(default_factory=dict)


class StructuredExtractionDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    json_schema: StrictStr = Field(min_length=1)
    context_strategy: StrictStr = Field(min_length=1)
    lookahead_images: bool
    items: list[StructuredExtractionItem]


STRUCTURED_EXTRACTION_DATASET = ArtifactTypeSpec(
    key=ArtifactTypeKey("notarius.extraction.dataset", 1),
    title="Structured extraction dataset",
    payload_schema=cast(
        JsonObject,
        StructuredExtractionDataset.model_json_schema(),
    ),
)


__all__ = [
    "STRUCTURED_EXTRACTION_DATASET",
    "StructuredExtractionDataset",
    "StructuredExtractionItem",
]
