from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class MessageContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    message_id: UUID = Field(default_factory=uuid4)
    project_id: UUID | None = None
    correlation_id: UUID | None = None


class RequestedMessageContract(MessageContract):
    requested_at: datetime = Field(default_factory=utcnow)
    requested_by: UUID | None = None


class WorkflowCompileRequested(RequestedMessageContract):
    workflow_version_id: UUID
    workflow_run_id: UUID | None = None


class WorkflowRunRequested(RequestedMessageContract):
    workflow_run_id: UUID
    workflow_version_id: UUID | None = None


class NodeRunExecuteRequested(RequestedMessageContract):
    workflow_run_id: UUID
    node_run_id: UUID


class RunEventType(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_PERMANENT = "failed_permanent"
    CANCELLED = "cancelled"


class ArtifactEventType(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"


class ErrorContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: str
    error_code: str
    error_message: str
    retryable: bool
    details: dict[str, str] = Field(default_factory=dict)


class WorkflowRunEvent(MessageContract):
    workflow_run_id: UUID
    event_type: RunEventType
    occurred_at: datetime = Field(default_factory=utcnow)
    error: ErrorContext | None = None


class NodeRunEvent(MessageContract):
    workflow_run_id: UUID
    node_run_id: UUID
    event_type: RunEventType
    occurred_at: datetime = Field(default_factory=utcnow)
    error: ErrorContext | None = None


class ArtifactEvent(MessageContract):
    artifact_id: UUID
    event_type: ArtifactEventType
    artifact_type: str | None = None
    workflow_run_id: UUID | None = None
    node_run_id: UUID | None = None
    occurred_at: datetime = Field(default_factory=utcnow)


class DlqMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_subject: str
    original_message_id: str
    consumer_name: str
    failure: ErrorContext
    failed_at: datetime = Field(default_factory=utcnow)
    attempt_count: int = Field(ge=1)
    workflow_run_id: UUID | None = None
    node_run_id: UUID | None = None
    artifact_id: UUID | None = None


__all__ = [
    "ArtifactEvent",
    "ArtifactEventType",
    "DlqMessage",
    "ErrorContext",
    "MessageContract",
    "NodeRunEvent",
    "NodeRunExecuteRequested",
    "RequestedMessageContract",
    "RunEventType",
    "WorkflowCompileRequested",
    "WorkflowRunEvent",
    "WorkflowRunRequested",
]
