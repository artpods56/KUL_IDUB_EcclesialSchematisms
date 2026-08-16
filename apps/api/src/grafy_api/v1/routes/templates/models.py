from datetime import datetime
from typing import Self
from uuid import UUID

from pydantic import Field, field_validator

from grafy_core.application.templates import TemplateInstantiation
from grafy_core.domain.templates import Template, TemplateState

from grafy_api.v1.models import ApiResponse


class TemplateResponse(ApiResponse):
    id: UUID
    workspace_id: UUID
    source_graph_id: UUID
    source_revision: int = Field(ge=1, strict=True)
    source_graph_name: str
    created_by_user_id: UUID | None = None
    name: str
    description: str | None = None
    state: TemplateState
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_template(cls, template: Template) -> Self:
        return cls(
            id=template.id,
            workspace_id=template.workspace_id,
            source_graph_id=template.source_graph_id,
            source_revision=template.source_revision,
            source_graph_name=template.source_graph_name,
            created_by_user_id=template.created_by_user_id,
            name=template.name,
            description=template.description,
            state=template.state,
            node_count=template.node_count,
            edge_count=template.edge_count,
            created_at=template.created_at,
            updated_at=template.updated_at,
        )


class TemplateListResponse(ApiResponse):
    templates: list[TemplateResponse]

    @classmethod
    def from_templates(cls, templates: list[Template]) -> Self:
        return cls(
            templates=[
                TemplateResponse.from_template(template) for template in templates
            ]
        )


class CreateTemplateRequest(ApiResponse):
    source_graph_id: UUID
    source_revision: int = Field(ge=1, strict=True)
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)

    @field_validator("name", "description", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class UpdateTemplateMetadataRequest(ApiResponse):
    name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=1000)

    @field_validator("name", "description", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class InstantiateTemplateRequest(ApiResponse):
    destination_workspace_id: UUID
    name: str = Field(min_length=1, max_length=160)
    folder_id: UUID | None = None

    @field_validator("name", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class TemplateInstantiationResponse(ApiResponse):
    template_id: UUID
    source_workspace_id: UUID
    destination_workspace_id: UUID
    graph_id: UUID
    graph_name: str
    folder_id: UUID | None = None

    @classmethod
    def from_instantiation(cls, result: TemplateInstantiation) -> Self:
        return cls(
            template_id=result.template_id,
            source_workspace_id=result.source_workspace_id,
            destination_workspace_id=result.destination_workspace_id,
            graph_id=result.graph.id,
            graph_name=result.graph.name,
            folder_id=result.folder_id,
        )


__all__ = [
    "CreateTemplateRequest",
    "InstantiateTemplateRequest",
    "TemplateInstantiationResponse",
    "TemplateListResponse",
    "TemplateResponse",
    "UpdateTemplateMetadataRequest",
]
