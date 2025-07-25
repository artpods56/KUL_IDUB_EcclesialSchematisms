from typing import List, Optional
from pydantic import BaseModel, Field


class ParishEntry(BaseModel):
    """Model for individual parish entry data."""
    deanery: Optional[str] = Field(None, description="Deanery name, null if not on page")
    parish: str = Field(..., description="Name of the parish")
    dedication: Optional[str] = Field(None, description="Church dedication/patron saint information")
    building_material: Optional[str] = Field(None, description="Building material (e.g., 'lig.' for wood, 'mur.' for brick/stone)")


class PageData(BaseModel):
    """Model for page data containing parish entries."""
    page_number: Optional[str] = Field(None, description="Page number as string")
    entries: List[ParishEntry] = Field(..., description="List of parish entries on this page")

