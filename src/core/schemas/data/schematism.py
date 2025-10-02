from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class SchematismEntry(BaseModel):
    """Model for individual parish entry data."""

    model_config = ConfigDict(
        extra="forbid",  # This makes Pydantic add additionalProperties: false
    )

    deanery: str | None = Field(
        None, description="Deanery description, null if not on page"
    )
    parish: str = Field(..., description="Name of the parish")
    dedication: str | None = Field(
        None, description="Church dedication/patron saint information"
    )
    building_material: str | None = Field(
        None,
        description="Building material (e.g., 'lig.' for wood, 'mur.' for brick/stone)",
    )


class SchematismPage(BaseModel):
    """Model for page data containing parish entries."""

    model_config = ConfigDict(
        extra="forbid",  # This makes Pydantic add additionalProperties: false
    )

    page_number: str | None = Field(None, description="Page number as string")
    entries: List[SchematismEntry] = Field(
        list(), description="List of parish entries on this page"
    )
