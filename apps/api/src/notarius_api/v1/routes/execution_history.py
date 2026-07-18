from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import StringConstraints

from notarius_core.domain.errors import NotFoundError
from notarius_core.domain.execution_history import GraphExecutionStatus

from notarius_api.schemas.execution_history import (
    GraphExecutionDetailResponse,
    GraphExecutionCursorModel,
    GraphExecutionListResponse,
    GraphExecutionNodeResultResponse,
    GraphExecutionSummaryResponse,
)
from notarius_api.services.execution_history import ExecutionHistoryService
from notarius_api.v1.routes.saved_graphs import SavedGraphDependency
from notarius_api.v1.routes.workbench import RunResultPresenterDependency


router = APIRouter(prefix="/graphs/{graph_id}/executions", tags=["executions"])

ExecutionNodeFilter = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


def execution_history_service(request: Request) -> ExecutionHistoryService:
    service = getattr(request.app.state, "execution_history", None)
    if not isinstance(service, ExecutionHistoryService):
        raise RuntimeError("Execution history service is not initialized")
    return service


ExecutionHistoryDependency = Annotated[
    ExecutionHistoryService,
    Depends(execution_history_service),
]


@router.get("", response_model=GraphExecutionListResponse)
async def list_graph_executions(
    graph_id: UUID,
    service: ExecutionHistoryDependency,
    saved_graphs: SavedGraphDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query()] = None,
    graph_revision: Annotated[int | None, Query(ge=1)] = None,
    execution_status: Annotated[
        GraphExecutionStatus | None,
        Query(alias="status"),
    ] = None,
    node_id: Annotated[ExecutionNodeFilter | None, Query()] = None,
) -> GraphExecutionListResponse:
    try:
        await saved_graphs.get(graph_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    decoded_cursor = None
    if cursor is not None:
        try:
            decoded_cursor = GraphExecutionCursorModel.decode(cursor)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    page = await service.list_for_graph(
        graph_id,
        limit=limit,
        cursor=decoded_cursor,
        graph_revision=graph_revision,
        status=execution_status,
        node_id=node_id,
    )
    return GraphExecutionListResponse(
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


@router.get("/{execution_id}", response_model=GraphExecutionDetailResponse)
async def get_graph_execution_history(
    graph_id: UUID,
    execution_id: UUID,
    service: ExecutionHistoryDependency,
    presenter: RunResultPresenterDependency,
) -> GraphExecutionDetailResponse:
    detail = await service.get_for_graph(graph_id, execution_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Graph execution not found")

    node_results = [
        GraphExecutionNodeResultResponse(
            node_id=node_result.node_id,
            position=node_result.position,
            status=node_result.status,
            error=node_result.error,
            completed_at=node_result.completed_at,
            outputs=[
                await presenter.port_output_response(port_name, value)
                for port_name, value in node_result.outputs.items()
            ],
        )
        for node_result in detail.node_results
    ]
    return GraphExecutionDetailResponse.from_detail(
        detail.execution,
        node_count=len(detail.node_results),
        artifact_count=sum(result.artifact_count for result in detail.node_results),
        node_results=node_results,
    )


__all__ = ["router"]
