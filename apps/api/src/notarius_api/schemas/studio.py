from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from notarius_api.schemas.platform import ArtifactResponse, ArtifactSequenceResponse
from notarius_core.domain.models import (
    ContextTrace,
    Job,
    JobItem,
    JobStatus,
    OutputSchema,
    Project,
    Recipe,
    Source,
    SourceItem,
)


class SourceItemCreate(BaseModel):
    order: int
    text: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None


class SourceCreate(BaseModel):
    name: str
    description: str | None = None
    items: list[SourceItemCreate] = Field(default_factory=list)


class OutputSchemaCreate(BaseModel):
    name: str
    description: str | None = None
    json_schema: dict[str, Any]


class RecipeCreate(BaseModel):
    name: str
    schema_id: UUID
    description: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class JobCreate(BaseModel):
    project_id: UUID
    source_id: UUID
    recipe_id: UUID


class StudioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ProjectResponse(StudioResponse):
    id: UUID
    name: str
    description: str | None
    created_at: datetime

    @classmethod
    def from_project(cls, project: Project) -> "ProjectResponse":
        return cls.model_validate(project)


class SourceResponse(StudioResponse):
    id: UUID
    project_id: UUID
    name: str
    description: str | None
    created_at: datetime

    @classmethod
    def from_source(cls, source: Source) -> "SourceResponse":
        return cls.model_validate(source)


class SourceItemResponse(StudioResponse):
    id: UUID
    source_id: UUID
    order: int
    text: str | None
    image_path: str | None
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_source_item(cls, item: SourceItem) -> "SourceItemResponse":
        return cls.model_validate(item)


class ImageSourceUploadResponse(StudioResponse):
    source: SourceResponse
    items: list[SourceItemResponse]
    artifacts: list[ArtifactResponse]
    sequence: ArtifactSequenceResponse


class PdfSourceUploadResponse(StudioResponse):
    source: SourceResponse
    items: list[SourceItemResponse]
    document_artifact: ArtifactResponse
    artifacts: list[ArtifactResponse]
    sequence: ArtifactSequenceResponse


class OutputSchemaResponse(StudioResponse):
    id: UUID
    project_id: UUID
    name: str
    description: str | None
    json_schema: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_output_schema(cls, schema: OutputSchema) -> "OutputSchemaResponse":
        return cls.model_validate(schema)


class RecipeResponse(StudioResponse):
    id: UUID
    project_id: UUID
    schema_id: UUID
    name: str
    description: str | None
    config: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> "RecipeResponse":
        return cls.model_validate(recipe)


class ContextTraceResponse(StudioResponse):
    rendered_input_context: dict[str, Any]
    previous_domain_context: dict[str, Any] | None
    structured_output: dict[str, Any] | None
    output_domain_context: dict[str, Any] | None
    model_metadata: dict[str, Any]
    error_details: str | None

    @classmethod
    def from_context_trace(cls, trace: ContextTrace | None) -> "ContextTraceResponse | None":
        return cls.model_validate(trace) if trace else None


class JobResponse(StudioResponse):
    id: UUID
    project_id: UUID
    source_id: UUID
    recipe_id: UUID
    status: JobStatus
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_job(cls, job: Job) -> "JobResponse":
        return cls.model_validate(job)


class JobItemResponse(StudioResponse):
    id: UUID
    job_id: UUID
    source_item_id: UUID
    order: int
    status: JobStatus
    structured_output: dict[str, Any] | None
    context_trace: ContextTraceResponse | None
    error: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_job_item(cls, item: JobItem) -> "JobItemResponse":
        data = {
            "id": item.id,
            "job_id": item.job_id,
            "source_item_id": item.source_item_id,
            "order": item.order,
            "status": item.status,
            "structured_output": item.structured_output,
            "context_trace": ContextTraceResponse.from_context_trace(item.context_trace),
            "error": item.error,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
        }
        return cls(**data)
