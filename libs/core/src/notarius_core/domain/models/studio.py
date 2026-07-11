from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(slots=True)
class Project:
    name: str
    description: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Source:
    project_id: UUID
    name: str
    description: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class SourceItem:
    source_id: UUID
    order: int
    text: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class OutputSchema:
    project_id: UUID
    name: str
    json_schema: dict[str, Any]
    description: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Recipe:
    project_id: UUID
    schema_id: UUID
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class ContextTrace:
    rendered_input_context: dict[str, Any] = field(default_factory=dict)
    previous_domain_context: dict[str, Any] | None = None
    structured_output: dict[str, Any] | None = None
    output_domain_context: dict[str, Any] | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)
    error_details: str | None = None


@dataclass(slots=True)
class Job:
    project_id: UUID
    source_id: UUID
    recipe_id: UUID
    status: JobStatus = JobStatus.QUEUED
    id: UUID = field(default_factory=uuid4)
    error: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def mark_running(self) -> None:
        self.status = JobStatus.RUNNING
        self.updated_at = utcnow()

    def mark_succeeded(self) -> None:
        self.status = JobStatus.SUCCEEDED
        self.updated_at = utcnow()

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.error = error
        self.updated_at = utcnow()

    def cancel(self) -> None:
        self.status = JobStatus.CANCELED
        self.updated_at = utcnow()

    def retry(self) -> None:
        self.status = JobStatus.QUEUED
        self.error = None
        self.updated_at = utcnow()


@dataclass(slots=True)
class JobItem:
    job_id: UUID
    source_item_id: UUID
    order: int
    status: JobStatus = JobStatus.QUEUED
    structured_output: dict[str, Any] | None = None
    context_trace: ContextTrace | None = None
    id: UUID = field(default_factory=uuid4)
    error: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

