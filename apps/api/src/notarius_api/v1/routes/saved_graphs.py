from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.domain.errors import (
    NotFoundError,
    SavedGraphRevisionConflictError,
)

from notarius_api.schemas.saved_graphs import (
    CreateSavedGraphRequest,
    SavedGraphListResponse,
    SavedGraphResponse,
    SavedGraphSummaryResponse,
    UpdateSavedGraphRequest,
)


router = APIRouter(prefix="/graphs", tags=["saved graphs"])


def saved_graph_service(request: Request) -> SavedGraphService:
    service = getattr(request.app.state, "saved_graphs", None)
    if not isinstance(service, SavedGraphService):
        raise RuntimeError("Saved graph service is not initialized")
    return service


SavedGraphDependency = Annotated[
    SavedGraphService,
    Depends(saved_graph_service),
]


@router.get("", response_model=SavedGraphListResponse)
async def list_saved_graphs(
    service: SavedGraphDependency,
) -> SavedGraphListResponse:
    graphs = await service.list()
    return SavedGraphListResponse(
        graphs=[SavedGraphSummaryResponse.from_graph(graph) for graph in graphs]
    )


@router.post(
    "",
    response_model=SavedGraphResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_saved_graph(
    request: CreateSavedGraphRequest,
    service: SavedGraphDependency,
) -> SavedGraphResponse:
    graph = await service.create(
        name=request.name,
        document=request.to_document(),
    )
    return SavedGraphResponse.from_graph(graph)


@router.get("/{graph_id}", response_model=SavedGraphResponse)
async def get_saved_graph(
    graph_id: UUID,
    service: SavedGraphDependency,
) -> SavedGraphResponse:
    try:
        graph = await service.get(graph_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SavedGraphResponse.from_graph(graph)


@router.put("/{graph_id}", response_model=SavedGraphResponse)
async def update_saved_graph(
    graph_id: UUID,
    request: UpdateSavedGraphRequest,
    service: SavedGraphDependency,
) -> SavedGraphResponse:
    try:
        graph = await service.replace(
            graph_id,
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
    expected_revision: Annotated[int, Query(ge=1)],
) -> Response:
    try:
        await service.delete(graph_id, expected_revision=expected_revision)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SavedGraphRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
