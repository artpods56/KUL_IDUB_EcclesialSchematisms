from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from notarius_core.domain.errors import (
    NotFoundError,
    SavedGraphRevisionConflictError,
)

from .dependencies import SavedGraphDependency
from .models import (
    CreateSavedGraphRequest,
    SavedGraphListResponse,
    SavedGraphResponse,
    UpdateSavedGraphRequest,
)
from notarius_api.v1.routes.workspace_scope import LegacyWorkspaceDependency


router = APIRouter(prefix="/graphs", tags=["saved graphs"])


@router.get("", response_model=SavedGraphListResponse)
async def list_saved_graphs(
    service: SavedGraphDependency,
    workspace_id: LegacyWorkspaceDependency,
) -> SavedGraphListResponse:
    graphs = await service.list(workspace_id)
    return SavedGraphListResponse.from_graphs(graphs)


@router.post(
    "",
    response_model=SavedGraphResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_saved_graph(
    request: CreateSavedGraphRequest,
    service: SavedGraphDependency,
    workspace_id: LegacyWorkspaceDependency,
) -> SavedGraphResponse:
    graph = await service.create(
        workspace_id=workspace_id,
        created_by_user_id=None,
        name=request.name,
        document=request.to_document(),
    )
    return SavedGraphResponse.from_graph(graph)


@router.get("/{graph_id}", response_model=SavedGraphResponse)
async def get_saved_graph(
    graph_id: UUID,
    service: SavedGraphDependency,
    workspace_id: LegacyWorkspaceDependency,
) -> SavedGraphResponse:
    try:
        graph = await service.get(workspace_id, graph_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SavedGraphResponse.from_graph(graph)


@router.put("/{graph_id}", response_model=SavedGraphResponse)
async def update_saved_graph(
    graph_id: UUID,
    request: UpdateSavedGraphRequest,
    service: SavedGraphDependency,
    workspace_id: LegacyWorkspaceDependency,
) -> SavedGraphResponse:
    try:
        graph = await service.replace(
            graph_id,
            workspace_id=workspace_id,
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
    workspace_id: LegacyWorkspaceDependency,
    expected_revision: Annotated[int, Query(ge=1)],
) -> Response:
    try:
        await service.delete(
            graph_id,
            workspace_id=workspace_id,
            expected_revision=expected_revision,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SavedGraphRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
