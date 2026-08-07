from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from notarius_core.domain.errors import (
    NotFoundError,
    SavedGraphRevisionConflictError,
)
from notarius_core.domain.identity import WorkspaceCapability

from notarius_api.v1.routes.auth.dependencies import require_workspace_capability

from .dependencies import SavedGraphDependency
from .models import (
    CreateSavedGraphRequest,
    SavedGraphListResponse,
    SavedGraphResponse,
    UpdateSavedGraphRequest,
)


router = APIRouter(prefix="/workspaces/{workspace_id}/graphs", tags=["saved graphs"])


@router.get("", response_model=SavedGraphListResponse)
async def list_saved_graphs(
    service: SavedGraphDependency,
    access: require_workspace_capability(WorkspaceCapability.VIEW_GRAPH),
) -> SavedGraphListResponse:
    graphs = await service.list(access.workspace_id)
    return SavedGraphListResponse.from_graphs(graphs)


@router.post(
    "",
    response_model=SavedGraphResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_saved_graph(
    request: CreateSavedGraphRequest,
    service: SavedGraphDependency,
    access: require_workspace_capability(WorkspaceCapability.CREATE_GRAPH),
) -> SavedGraphResponse:
    graph = await service.create(
        workspace_id=access.workspace_id,
        created_by_user_id=access.actor.user_id,
        name=request.name,
        document=request.to_document(),
    )
    return SavedGraphResponse.from_graph(graph)


@router.get("/{graph_id}", response_model=SavedGraphResponse)
async def get_saved_graph(
    graph_id: UUID,
    service: SavedGraphDependency,
    access: require_workspace_capability(WorkspaceCapability.VIEW_GRAPH),
) -> SavedGraphResponse:
    try:
        graph = await service.get(access.workspace_id, graph_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SavedGraphResponse.from_graph(graph)


@router.put("/{graph_id}", response_model=SavedGraphResponse)
async def update_saved_graph(
    graph_id: UUID,
    request: UpdateSavedGraphRequest,
    service: SavedGraphDependency,
    access: require_workspace_capability(WorkspaceCapability.EDIT_GRAPH),
) -> SavedGraphResponse:
    try:
        graph = await service.replace(
            graph_id,
            workspace_id=access.workspace_id,
            name=request.name,
            document=request.to_document(),
            expected_revision=request.expected_revision,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SavedGraphRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SavedGraphResponse.from_graph(graph)


@router.delete("/{graph_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_saved_graph(
    graph_id: UUID,
    service: SavedGraphDependency,
    access: require_workspace_capability(WorkspaceCapability.DELETE_GRAPH),
    expected_revision: Annotated[int, Query(ge=1)],
) -> Response:
    try:
        await service.delete(
            graph_id,
            workspace_id=access.workspace_id,
            expected_revision=expected_revision,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SavedGraphRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
