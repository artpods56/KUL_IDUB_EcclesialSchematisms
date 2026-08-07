from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from notarius_core.domain.collaboration import ReplaceDocumentCommand
from notarius_core.domain.errors import (
    CollaborationActiveExecutionError,
    CollaborationHeadConflictError,
    CollaborationUncheckpointedError,
    MissingCollaborativeHeadError,
    NotFoundError,
    SavedGraphRevisionConflictError,
)
from notarius_core.domain.identity import WorkspaceCapability

from notarius_api.v1.routes.auth.dependencies import require_workspace_capability

from .dependencies import CollaborationDependency, SavedGraphDependency
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
    collaboration: CollaborationDependency,
    access: require_workspace_capability(WorkspaceCapability.CREATE_GRAPH),
) -> SavedGraphResponse:
    graph, _, _ = await collaboration.bootstrap_graph(
        actor=access.actor,
        workspace_id=access.workspace_id,
        command_id=uuid4(),
        command=ReplaceDocumentCommand(
            name=request.name,
            document=request.to_document(),
        ),
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
    collaboration: CollaborationDependency,
    access: require_workspace_capability(WorkspaceCapability.EDIT_GRAPH),
) -> SavedGraphResponse:
    try:
        graph, _ = await collaboration.replace_complete_document(
            actor=access.actor,
            workspace_id=access.workspace_id,
            graph_id=graph_id,
            name=request.name,
            document=request.to_document(),
            expected_revision=request.expected_revision,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MissingCollaborativeHeadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        SavedGraphRevisionConflictError,
        CollaborationUncheckpointedError,
        CollaborationHeadConflictError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SavedGraphResponse.from_graph(graph)


@router.delete("/{graph_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_saved_graph(
    graph_id: UUID,
    collaboration: CollaborationDependency,
    access: require_workspace_capability(WorkspaceCapability.DELETE_GRAPH),
    expected_revision: Annotated[int, Query(ge=1)],
) -> Response:
    try:
        await collaboration.delete_graph(
            actor=access.actor,
            workspace_id=access.workspace_id,
            graph_id=graph_id,
            expected_revision=expected_revision,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MissingCollaborativeHeadError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (
        SavedGraphRevisionConflictError,
        CollaborationUncheckpointedError,
        CollaborationActiveExecutionError,
        CollaborationHeadConflictError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
