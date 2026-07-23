import base64
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Final, Literal, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from notarius_core.artifacts import ArtifactRef, ArtifactRefSequence
from notarius_core.conversions import MAX_ARTIFACT_CONVERSION_HOPS
from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionCursor,
    GraphExecutionDetail,
    GraphExecutionNodeResult,
    GraphExecutionNodeStatus,
    GraphExecutionPage,
    GraphExecutionScope,
    GraphExecutionStatus,
)
from notarius_core.nodes import (
    MAX_NODE_PROGRESS_COUNTER,
    MAX_NODE_PROGRESS_MESSAGE_LENGTH,
)

from notarius_api.v1.models import ApiResponse, ArtifactTypeBindingModel
from notarius_api.v1.routes.artifacts.models import ArtifactSummaryResponse


MAX_EXECUTION_NODE_PATH_LENGTH: Final = 64

ExecutionIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
InputPlugIdentifier = ExecutionIdentifier
InvocationIndex = Annotated[int, Field(ge=0)]


class RunInputPlugRequest(BaseModel):
    id: InputPlugIdentifier
    port: InputPlugIdentifier


class RunNodeRequest(BaseModel):
    id: ExecutionIdentifier
    operator_id: str
    operator_version: int
    config: dict[str, object] = Field(default_factory=dict)
    input_plugs: list[RunInputPlugRequest] = Field(default_factory=list)
    artifact_type_bindings: list[ArtifactTypeBindingModel] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_artifact_type_bindings(self) -> "RunNodeRequest":
        variables = [binding.variable for binding in self.artifact_type_bindings]
        if len(variables) != len(set(variables)):
            raise ValueError("Node artifact type binding variables must be unique")
        return self


class FieldProjectionRequest(BaseModel):
    path: list[str] = Field(min_length=1)


class ArtifactConversionRequest(BaseModel):
    id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    version: int = Field(ge=1)


class RunEdgeRequest(BaseModel):
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    to_plug: InputPlugIdentifier | None = None
    projection: FieldProjectionRequest | None = None
    conversion_path: list[ArtifactConversionRequest] = Field(
        default_factory=list,
        max_length=MAX_ARTIFACT_CONVERSION_HOPS,
    )
    collection_mode: Literal["direct", "map"] = "direct"

    @model_validator(mode="before")
    @classmethod
    def normalize_singular_conversion(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        raw = cast(Mapping[object, object], value)
        if "conversion" not in raw:
            return dict(raw)
        if "conversion_path" in raw:
            raise ValueError(
                "Run edge cannot declare both conversion and conversion_path"
            )
        normalized = dict(raw)
        conversion = normalized.pop("conversion")
        normalized["conversion_path"] = [] if conversion is None else [conversion]
        return normalized


class PinnedOutputRequest(BaseModel):
    from_node: str
    from_port: str
    value: ArtifactRef | ArtifactRefSequence


class RunRequest(BaseModel):
    nodes: list[RunNodeRequest]
    edges: list[RunEdgeRequest] = Field(default_factory=list)
    pinned_outputs: list[PinnedOutputRequest] = Field(default_factory=list)
    scope: GraphExecutionScope = "all"
    graph_id: UUID | None = None
    graph_revision: int | None = Field(default=None, ge=1)
    secret_graph_id: UUID | None = None
    secret_graph_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_graph_context(self) -> "RunRequest":
        if (self.graph_id is None) != (self.graph_revision is None):
            raise ValueError("graph_id and graph_revision must be provided together")
        if (self.secret_graph_id is None) != (self.secret_graph_revision is None):
            raise ValueError(
                "secret_graph_id and secret_graph_revision must be provided together"
            )
        if (
            self.graph_id is not None
            and self.secret_graph_id is not None
            and (
                self.graph_id != self.secret_graph_id
                or self.graph_revision != self.secret_graph_revision
            )
        ):
            raise ValueError(
                "graph and secret graph contexts must identify the same saved "
                "graph revision when both are provided"
            )
        return self


class RunPortOutputResponse(ApiResponse):
    port: str
    kind: Literal["single", "sequence"]
    value: ArtifactRef | ArtifactRefSequence
    artifacts: list[ArtifactSummaryResponse]


class RunNodeResponse(ApiResponse):
    node_id: str
    status: Literal["succeeded", "failed", "skipped"]
    error: str | None
    outputs: list[RunPortOutputResponse]


class RunResponse(ApiResponse):
    status: Literal["succeeded", "failed"]
    node_runs: list[RunNodeResponse]


type RunExecutionStatus = Literal[
    "queued",
    "running",
    "cancelling",
    "cancelled",
    "succeeded",
    "failed",
]


class RunExecutionResponse(ApiResponse):
    execution_id: UUID
    status: Literal[
        "queued",
        "running",
        "cancelling",
        "cancelled",
        "succeeded",
        "failed",
    ]
    active_node_id: str | None
    result: RunResponse | None
    error: str | None


class RunExecutionEventBase(ApiResponse):
    sequence: int = Field(ge=1)
    execution_id: UUID
    occurred_at: datetime


class ExecutionStatusEvent(RunExecutionEventBase):
    kind: Literal["execution.status"] = "execution.status"
    status: RunExecutionStatus
    active_node_id: ExecutionIdentifier | None


type NodeExecutionEventStatus = Literal[
    "running",
    "succeeded",
    "failed",
    "skipped",
]


class NodeExecutionEventBase(RunExecutionEventBase):
    node_path: list[ExecutionIdentifier] = Field(
        min_length=1,
        max_length=MAX_EXECUTION_NODE_PATH_LENGTH,
    )
    node_id: ExecutionIdentifier
    node_run_id: UUID | None
    invocation_index: InvocationIndex | None = None
    invocation_path: list[InvocationIndex] = Field(
        default_factory=list,
        max_length=MAX_EXECUTION_NODE_PATH_LENGTH,
    )


class NodeStatusEvent(NodeExecutionEventBase):
    kind: Literal["node.status"] = "node.status"
    status: NodeExecutionEventStatus


class NodeProgressEvent(NodeExecutionEventBase):
    kind: Literal["node.progress"] = "node.progress"
    message: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=MAX_NODE_PROGRESS_MESSAGE_LENGTH,
        ),
    ]
    current: int | None = Field(
        default=None,
        ge=0,
        le=MAX_NODE_PROGRESS_COUNTER,
    )
    total: int | None = Field(
        default=None,
        ge=0,
        le=MAX_NODE_PROGRESS_COUNTER,
    )

    @model_validator(mode="after")
    def validate_progress(self) -> "NodeProgressEvent":
        if (
            self.current is not None
            and self.total is not None
            and self.current > self.total
        ):
            raise ValueError("Node progress current value must not exceed total")
        return self


type RunExecutionEvent = Annotated[
    ExecutionStatusEvent | NodeStatusEvent | NodeProgressEvent,
    Field(discriminator="kind"),
]


class GraphMaterializationsResponse(ApiResponse):
    graph_id: UUID
    graph_revision: int
    node_runs: list[RunNodeResponse]


class GraphExecutionCursorModel(BaseModel):
    created_at: datetime
    execution_id: UUID

    @classmethod
    def decode(cls, value: str) -> GraphExecutionCursor:
        try:
            padding = "=" * (-len(value) % 4)
            payload = base64.urlsafe_b64decode(value + padding)
            cursor = cls.model_validate_json(payload)
            return GraphExecutionCursor(
                created_at=cursor.created_at,
                execution_id=cursor.execution_id,
            )
        except (ValueError, ValidationError) as exc:
            raise ValueError("Invalid execution history cursor") from exc

    @classmethod
    def encode(cls, cursor: GraphExecutionCursor) -> str:
        payload = cls(
            created_at=cursor.created_at,
            execution_id=cursor.execution_id,
        ).model_dump_json()
        return (
            base64.urlsafe_b64encode(payload.encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )


class GraphExecutionSummaryResponse(ApiResponse):
    execution_id: UUID
    graph_id: UUID
    graph_revision: int
    scope: GraphExecutionScope
    status: GraphExecutionStatus
    requested_node_ids: list[str]
    node_count: int
    artifact_count: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    workflow_run_id: UUID | None
    error: str | None

    @classmethod
    def from_execution(
        cls,
        execution: GraphExecution,
        *,
        node_count: int,
        artifact_count: int,
    ) -> "GraphExecutionSummaryResponse":
        return cls(
            execution_id=execution.execution_id,
            graph_id=execution.graph_id,
            graph_revision=execution.graph_revision,
            scope=execution.scope,
            status=execution.status,
            requested_node_ids=list(execution.requested_node_ids),
            node_count=node_count,
            artifact_count=artifact_count,
            created_at=execution.created_at,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            workflow_run_id=execution.workflow_run_id,
            error=execution.error,
        )


class GraphExecutionListResponse(ApiResponse):
    items: list[GraphExecutionSummaryResponse]
    next_cursor: str | None

    @classmethod
    def from_page(cls, page: GraphExecutionPage) -> "GraphExecutionListResponse":
        return cls(
            items=[
                GraphExecutionSummaryResponse.from_execution(
                    item.execution,
                    node_count=item.node_count,
                    artifact_count=item.artifact_count,
                )
                for item in page.items
            ],
            next_cursor=(
                GraphExecutionCursorModel.encode(page.next_cursor)
                if page.next_cursor is not None
                else None
            ),
        )


class GraphExecutionNodeResultResponse(ApiResponse):
    node_id: str
    position: int
    status: GraphExecutionNodeStatus
    error: str | None
    completed_at: datetime
    outputs: list[RunPortOutputResponse]

    @classmethod
    def from_result(
        cls,
        result: GraphExecutionNodeResult,
        *,
        outputs: list[RunPortOutputResponse],
    ) -> "GraphExecutionNodeResultResponse":
        return cls(
            node_id=result.node_id,
            position=result.position,
            status=result.status,
            error=result.error,
            completed_at=result.completed_at,
            outputs=outputs,
        )


class GraphExecutionDetailResponse(GraphExecutionSummaryResponse):
    node_results: list[GraphExecutionNodeResultResponse]

    @classmethod
    def from_detail(
        cls,
        detail: GraphExecutionDetail,
        *,
        node_results: list[GraphExecutionNodeResultResponse],
    ) -> "GraphExecutionDetailResponse":
        execution = detail.execution
        return cls(
            execution_id=execution.execution_id,
            graph_id=execution.graph_id,
            graph_revision=execution.graph_revision,
            scope=execution.scope,
            status=execution.status,
            requested_node_ids=list(execution.requested_node_ids),
            node_count=len(detail.node_results),
            artifact_count=sum(result.artifact_count for result in detail.node_results),
            created_at=execution.created_at,
            started_at=execution.started_at,
            finished_at=execution.finished_at,
            workflow_run_id=execution.workflow_run_id,
            error=execution.error,
            node_results=node_results,
        )


__all__ = [
    "ArtifactConversionRequest",
    "ExecutionStatusEvent",
    "ExecutionIdentifier",
    "FieldProjectionRequest",
    "GraphExecutionCursorModel",
    "GraphExecutionDetailResponse",
    "GraphExecutionListResponse",
    "GraphExecutionNodeResultResponse",
    "GraphExecutionSummaryResponse",
    "GraphMaterializationsResponse",
    "InputPlugIdentifier",
    "InvocationIndex",
    "MAX_EXECUTION_NODE_PATH_LENGTH",
    "NodeExecutionEventBase",
    "NodeExecutionEventStatus",
    "NodeProgressEvent",
    "NodeStatusEvent",
    "PinnedOutputRequest",
    "RunEdgeRequest",
    "RunExecutionEvent",
    "RunExecutionEventBase",
    "RunExecutionResponse",
    "RunExecutionStatus",
    "RunInputPlugRequest",
    "RunNodeRequest",
    "RunNodeResponse",
    "RunPortOutputResponse",
    "RunRequest",
    "RunResponse",
]
