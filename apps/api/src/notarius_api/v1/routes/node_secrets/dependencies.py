from typing import Annotated

from fastapi import Depends, Request

from notarius_api.app_state import get_resources
from notarius_api.v1.routes.node_secrets.services import NodeSecretService


def node_secret_service(request: Request) -> NodeSecretService:
    return get_resources(request.app).node_secrets


NodeSecretDependency = Annotated[NodeSecretService, Depends(node_secret_service)]


__all__ = ["NodeSecretDependency", "node_secret_service"]
