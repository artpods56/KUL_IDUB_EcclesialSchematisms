from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response

from notarius_core.domain.errors import (
    NotFoundError,
    SavedGraphRevisionConflictError,
)

from notarius_api.schemas.node_secrets import (
    ConfigureNodeSecretRequest,
    GraphNodeSecretsResponse,
    NodeSecretStatusResponse,
)
from notarius_api.services.node_secrets import (
    NodeSecretConfigurationError,
    NodeSecretDeclarationError,
    NodeSecretService,
    NodeSecretValueError,
)


router = APIRouter(prefix="/graphs", tags=["node secrets"])


def node_secret_service(request: Request) -> NodeSecretService:
    service = getattr(request.app.state, "node_secrets", None)
    if not isinstance(service, NodeSecretService):
        raise RuntimeError("Node secret service is not initialized")
    return service


NodeSecretDependency = Annotated[NodeSecretService, Depends(node_secret_service)]


@router.get(
    "/{graph_id}/node-secrets",
    response_model=GraphNodeSecretsResponse,
)
async def get_node_secret_status(
    graph_id: UUID,
    service: NodeSecretDependency,
) -> GraphNodeSecretsResponse:
    try:
        state = await service.status(graph_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GraphNodeSecretsResponse(
        graph_id=state.graph_id,
        graph_revision=state.graph_revision,
        secrets=[
            NodeSecretStatusResponse(
                node_id=secret.node_id,
                name=secret.name,
                configured=secret.configured,
            )
            for secret in state.secrets
        ],
    )


@router.put(
    "/{graph_id}/nodes/{node_id}/secrets/{name}",
    response_model=NodeSecretStatusResponse,
)
async def configure_node_secret(
    graph_id: UUID,
    node_id: str,
    name: str,
    request: ConfigureNodeSecretRequest,
    service: NodeSecretDependency,
) -> NodeSecretStatusResponse:
    try:
        state = await service.configure(
            graph_id=graph_id,
            node_id=node_id,
            name=name,
            value=request.value,
            expected_graph_revision=request.expected_graph_revision,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NodeSecretDeclarationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SavedGraphRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NodeSecretConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except NodeSecretValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return NodeSecretStatusResponse(
        node_id=state.node_id,
        name=state.name,
        configured=state.configured,
    )


@router.delete(
    "/{graph_id}/nodes/{node_id}/secrets/{name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_node_secret(
    graph_id: UUID,
    node_id: str,
    name: str,
    service: NodeSecretDependency,
    expected_graph_revision: Annotated[int, Query(ge=1)],
) -> Response:
    try:
        await service.remove(
            graph_id=graph_id,
            node_id=node_id,
            name=name,
            expected_graph_revision=expected_graph_revision,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except NodeSecretDeclarationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SavedGraphRevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
