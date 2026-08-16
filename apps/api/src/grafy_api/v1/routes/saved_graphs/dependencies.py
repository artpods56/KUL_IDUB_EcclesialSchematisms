from typing import Annotated

from fastapi import Depends, Request

from grafy_core.application.collaboration import CollaborationService
from grafy_core.application.saved_graphs import SavedGraphService

from grafy_api.app_state import get_resources


def saved_graph_service(request: Request) -> SavedGraphService:
    return get_resources(request.app).saved_graphs


def collaboration_service(request: Request) -> CollaborationService:
    return get_resources(request.app).collaboration


SavedGraphDependency = Annotated[
    SavedGraphService,
    Depends(saved_graph_service),
]

CollaborationDependency = Annotated[
    CollaborationService,
    Depends(collaboration_service),
]


__all__ = [
    "CollaborationDependency",
    "SavedGraphDependency",
    "collaboration_service",
    "saved_graph_service",
]
