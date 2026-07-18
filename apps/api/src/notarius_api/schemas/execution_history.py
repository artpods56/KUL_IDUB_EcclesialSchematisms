import base64
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ValidationError

from notarius_core.domain.execution_history import (
    GraphExecution,
    GraphExecutionCursor,
    GraphExecutionNodeStatus,
    GraphExecutionScope,
    GraphExecutionStatus,
)

from notarius_api.schemas.workbench import ApiResponse, RunPortOutputResponse


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
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip(
            "="
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


class GraphExecutionNodeResultResponse(ApiResponse):
    node_id: str
    position: int
    status: GraphExecutionNodeStatus
    error: str | None
    completed_at: datetime
    outputs: list[RunPortOutputResponse]


class GraphExecutionDetailResponse(GraphExecutionSummaryResponse):
    node_results: list[GraphExecutionNodeResultResponse]

    @classmethod
    def from_detail(
        cls,
        execution: GraphExecution,
        *,
        node_count: int,
        artifact_count: int,
        node_results: list[GraphExecutionNodeResultResponse],
    ) -> "GraphExecutionDetailResponse":
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
            node_results=node_results,
        )


__all__ = [
    "GraphExecutionDetailResponse",
    "GraphExecutionCursorModel",
    "GraphExecutionListResponse",
    "GraphExecutionNodeResultResponse",
    "GraphExecutionSummaryResponse",
]
