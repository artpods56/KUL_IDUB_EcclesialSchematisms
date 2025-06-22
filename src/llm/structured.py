from typing import List, Optional
from pydantic import BaseModel, Field


class ParishEntry(BaseModel):
    """Model for individual parish entry data."""
    parish: str = Field(..., description="Name of the parish")
    dedication: str = Field(..., description="Church dedication/patron saint information")
    building_material: str = Field(..., description="Building material (e.g., 'lig.' for wood, 'mur.' for brick/stone)")


class PageData(BaseModel):
    """Model for page data containing parish entries."""
    page_number: str = Field(..., description="Page number as string")
    deanery: Optional[str] = Field(None, description="Deanery name, can be null")
    entries: List[ParishEntry] = Field(..., description="List of parish entries on this page")

