from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeAlias
from uuid import UUID, uuid4


JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def utcnow() -> datetime:
    return datetime.now(UTC)


class ExecutionMode(StrEnum):
    SINGLE = "single"
    MAP = "map"
    REDUCE = "reduce"
    STATEFUL_SEQUENCE = "stateful_sequence"


class WorkflowRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_PERMANENT = "failed_permanent"
    CANCELLED = "cancelled"


class NodeRunStatus(StrEnum):
    QUEUED = "queued"
    BLOCKED = "blocked"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_PERMANENT = "failed_permanent"
    CANCELLED = "cancelled"


class OutboxMessageStatus(StrEnum):
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"


class ExperimentStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: UUID
    artifact_type: str
    schema_version: int
    content_hash: str | None = None


@dataclass(slots=True)
class Artifact:
    artifact_type: str
    schema_version: int
    workflow_run_id: UUID | None
    producer_node_run_id: UUID | None
    payload_ref: str
    id: UUID = field(default_factory=uuid4)
    producer_operator_id: str | None = None
    producer_operator_version: str | None = None
    input_artifact_ids: list[UUID] = field(default_factory=list)
    content_hash: str | None = None
    preview_ref: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    def ref(self) -> ArtifactRef:
        return ArtifactRef(
            artifact_id=self.id,
            artifact_type=self.artifact_type,
            schema_version=self.schema_version,
            content_hash=self.content_hash,
        )


@dataclass(slots=True)
class ArtifactSequence:
    artifact_type: str
    item_refs: list[ArtifactRef]
    schema_version: int
    id: UUID = field(default_factory=uuid4)
    ordered: bool = True
    index_key: str = "sequence_index"
    metadata: JsonObject = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        for item_ref in self.item_refs:
            if item_ref.artifact_type != self.artifact_type:
                raise ValueError(
                    "ArtifactSequence item type mismatch: "
                    f"expected {self.artifact_type}, got {item_ref.artifact_type}"
                )
            if item_ref.schema_version != self.schema_version:
                raise ValueError(
                    "ArtifactSequence schema version mismatch: "
                    f"expected {self.schema_version}, got {item_ref.schema_version}"
                )

    def ref(self) -> "ArtifactSequenceRef":
        return ArtifactSequenceRef(
            sequence_id=self.id,
            artifact_type=self.artifact_type,
            schema_version=self.schema_version,
        )



@dataclass(frozen=True, slots=True)
class ArtifactSequenceRef:
    sequence_id: UUID
    artifact_type: str
    schema_version: int


ArtifactPortRef: TypeAlias = ArtifactRef | ArtifactSequenceRef | list[ArtifactRef]


@dataclass(frozen=True, slots=True)
class PortSpec:
    name: str
    artifact_type: str
    schema_version: int
    sequence: bool = False
    required: bool = True
    description: str | None = None


@dataclass(frozen=True, slots=True)
class NodeSpec:
    id: str
    version: str
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    execution_mode: ExecutionMode
    config_schema: JsonObject = field(default_factory=dict)
    display_name: str | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    id: str
    operator_id: str
    operator_version: str
    config: JsonObject = field(default_factory=dict)
    label: str | None = None
    ui_position: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkflowEdge:
    from_node_id: str
    from_port: str
    to_node_id: str
    to_port: str


@dataclass(slots=True)
class WorkflowDefinition:
    name: str
    nodes: list[WorkflowNode] = field(default_factory=list)
    edges: list[WorkflowEdge] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    description: str | None = None
    declared_inputs: list[PortSpec] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class WorkflowVersion:
    workflow_definition_id: UUID
    version_number: int
    definition_snapshot: WorkflowDefinition
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)
    created_by: str | None = None
    change_note: str | None = None


@dataclass(slots=True)
class WorkflowRun:
    workflow_version_id: UUID
    input_artifact_refs: list[ArtifactRef] = field(default_factory=list)
    input_artifact_sequence_refs: list[ArtifactSequenceRef] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    status: WorkflowRunStatus = WorkflowRunStatus.QUEUED
    output_artifact_refs: list[ArtifactRef] = field(default_factory=list)
    metadata: JsonObject = field(default_factory=dict)
    error: str | None = None
    queued_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            WorkflowRunStatus.SUCCEEDED,
            WorkflowRunStatus.FAILED_PERMANENT,
            WorkflowRunStatus.CANCELLED,
        }

    def mark_running(self) -> None:
        self.status = WorkflowRunStatus.RUNNING
        self.started_at = utcnow()
        self.finished_at = None
        self.error = None

    def mark_queued(self) -> None:
        self.status = WorkflowRunStatus.QUEUED
        self.queued_at = utcnow()
        self.started_at = None
        self.finished_at = None
        self.error = None

    def mark_succeeded(
        self, output_artifact_refs: list[ArtifactRef] | None = None
    ) -> None:
        self.status = WorkflowRunStatus.SUCCEEDED
        if output_artifact_refs is not None:
            self.output_artifact_refs = output_artifact_refs
        self.finished_at = utcnow()
        self.error = None

    def mark_failed(self, error: str, retryable: bool) -> None:
        self.status = (
            WorkflowRunStatus.FAILED_RETRYABLE
            if retryable
            else WorkflowRunStatus.FAILED_PERMANENT
        )
        self.error = error
        self.finished_at = utcnow()

    def mark_cancelled(self) -> None:
        self.status = WorkflowRunStatus.CANCELLED
        self.finished_at = utcnow()


@dataclass(frozen=True, slots=True)
class ExperimentParameter:
    name: str
    node_id: str
    config_path: tuple[str, ...]
    values: tuple[JsonValue, ...]
    description: str | None = None

    def __post_init__(self) -> None:
        if self.name == "":
            raise ValueError("ExperimentParameter.name must be non-empty")
        if self.node_id == "":
            raise ValueError(
                f"ExperimentParameter {self.name!r} requires a non-empty node_id"
            )
        if not self.config_path:
            raise ValueError(
                f"ExperimentParameter {self.name!r} requires a config_path"
            )
        if any(path_part == "" for path_part in self.config_path):
            raise ValueError(
                f"ExperimentParameter {self.name!r} config_path cannot contain "
                "empty parts"
            )
        if not self.values:
            raise ValueError(f"ExperimentParameter {self.name!r} requires values")


@dataclass(slots=True)
class ExperimentVariant:
    key: str
    ordinal: int
    parameter_values: JsonObject
    workflow_run_id: UUID
    id: UUID = field(default_factory=uuid4)
    metadata: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class Experiment:
    name: str
    workflow_version_id: UUID
    parameters: list[ExperimentParameter]
    input_artifact_refs: list[ArtifactRef] = field(default_factory=list)
    input_artifact_sequence_refs: list[ArtifactSequenceRef] = field(default_factory=list)
    variants: list[ExperimentVariant] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    description: str | None = None
    status: ExperimentStatus = ExperimentStatus.QUEUED
    metadata: JsonObject = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    @property
    def workflow_run_ids(self) -> list[UUID]:
        return [variant.workflow_run_id for variant in self.variants]


@dataclass(slots=True)
class NodeRun:
    workflow_run_id: UUID
    workflow_node_id: str
    operator_id: str
    operator_version: str
    input_artifact_refs: dict[str, ArtifactPortRef] = field(
        default_factory=dict
    )
    id: UUID = field(default_factory=uuid4)
    status: NodeRunStatus = NodeRunStatus.QUEUED
    output_artifact_refs: dict[str, ArtifactPortRef] = field(
        default_factory=dict
    )
    attempt_count: int = 0
    max_attempts: int = 5
    metadata: JsonObject = field(default_factory=dict)
    error: str | None = None
    queued_at: datetime = field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            NodeRunStatus.SUCCEEDED,
            NodeRunStatus.FAILED_PERMANENT,
            NodeRunStatus.CANCELLED,
        }

    def mark_running(self) -> None:
        self.status = NodeRunStatus.RUNNING
        self.attempt_count += 1
        self.started_at = utcnow()
        self.finished_at = None
        self.error = None

    def mark_queued(self) -> None:
        self.status = NodeRunStatus.QUEUED
        self.queued_at = utcnow()
        self.started_at = None
        self.finished_at = None
        self.error = None

    def mark_succeeded(
        self,
        output_artifact_refs: dict[str, ArtifactPortRef],
    ) -> None:
        self.status = NodeRunStatus.SUCCEEDED
        self.output_artifact_refs = output_artifact_refs
        self.finished_at = utcnow()
        self.error = None

    def mark_failed(self, error: str, retryable: bool) -> None:
        self.status = (
            NodeRunStatus.FAILED_RETRYABLE
            if retryable
            else NodeRunStatus.FAILED_PERMANENT
        )
        self.error = error
        self.finished_at = utcnow()

    def mark_blocked(self) -> None:
        self.status = NodeRunStatus.BLOCKED
        self.finished_at = None
        self.error = None

    def mark_cancelled(self) -> None:
        self.status = NodeRunStatus.CANCELLED
        self.finished_at = utcnow()


@dataclass(slots=True)
class InputAssemblyTrace:
    node_run_id: UUID
    selected_inputs: dict[str, ArtifactRef | list[ArtifactRef]] = field(
        default_factory=dict
    )
    omitted_inputs: dict[str, str] = field(default_factory=dict)
    policies: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class InvocationTrace:
    node_run_id: UUID
    invocation_type: str
    input_artifact_refs: list[ArtifactRef] = field(default_factory=list)
    output_artifact_refs: list[ArtifactRef] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    provider: str | None = None
    model: str | None = None
    request_ref: str | None = None
    response_ref: str | None = None
    runtime: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)
    error: str | None = None
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class OutboxMessage:
    subject: str
    payload: JsonObject
    message_type: str
    id: UUID = field(default_factory=uuid4)
    status: OutboxMessageStatus = OutboxMessageStatus.PENDING
    attempts: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    published_at: datetime | None = None

    def mark_published(self) -> None:
        self.status = OutboxMessageStatus.PUBLISHED
        self.published_at = utcnow()
        self.error = None

    def mark_failed(self, error: str) -> None:
        self.status = OutboxMessageStatus.PENDING
        self.attempts += 1
        self.error = error

    def mark_permanently_failed(self, error: str) -> None:
        self.status = OutboxMessageStatus.FAILED
        self.error = error

    def requeue(self) -> None:
        self.status = OutboxMessageStatus.PENDING
        self.attempts = 0
        self.error = None
        self.published_at = None
