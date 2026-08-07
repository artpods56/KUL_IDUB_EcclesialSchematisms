from typing import Annotated

from fastapi import Depends, Request

from notarius_api.v1.routes.node_secrets.services import NodeSecretService


def node_secret_service(request: Request) -> NodeSecretService:
    service = getattr(request.app.state, "node_secrets", None)
    if not isinstance(service, NodeSecretService):
        raise RuntimeError("Node secret service is not initialized")
    return service


NodeSecretDependency = Annotated[NodeSecretService, Depends(node_secret_service)]


__all__ = ["NodeSecretDependency", "node_secret_service"]
