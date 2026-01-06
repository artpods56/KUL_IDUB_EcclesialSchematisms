from pydantic import BaseModel, ConfigDict, Field


class SchematismEntry(BaseModel):
    """Model for individual parish entry data."""

    model_config = ConfigDict(
        extra="forbid",  # This makes Pydantic add additionalProperties: false
    )

    deanery: str | None = Field(
        default=None, description="Deanery description, null if not on page"
    )
    parish: str | None = Field(default=None, description="Name of the parish")
    dedication: str | None = Field(
        default=None, description="Church dedication/patron saint information"
    )
    building_material: str | None = Field(
        default=None,
        description="Building material (e.g., 'lig.' for wood, 'mur.' for brick/stone)",
    )


class PageContext(BaseModel):
    """Context state to carry forward between pages."""

    model_config = ConfigDict(
        extra="forbid",
    )

    summary: str | None = Field(default=None, description="Short summary of the page")

    note: str | None = Field(
        default=None,
        description="Very specific note about the page. Used only to explain difficult predictions or those that require additional context to be explained.",
    )

    active_deanery: str | None = Field(
        default=None,
        description="The deanery that is active at the end of this page",
    )
    last_page_number: str | None = Field(
        default=None,
        description="This page's page number for validation",
    )


class SchematismPage(BaseModel):
    """Model for page data containing parish entries."""

    model_config = ConfigDict(
        extra="forbid",  # This makes Pydantic add additionalProperties: false
    )

    page_number: str | None = Field(None, description="Page number as string")
    entries: list[SchematismEntry] = Field(
        default_factory=list, description="List of parish entries on this page"
    )
    context: PageContext | None = Field(
        default=None,
        description="Context state to pass to the next page for multi-page processing",
    )
