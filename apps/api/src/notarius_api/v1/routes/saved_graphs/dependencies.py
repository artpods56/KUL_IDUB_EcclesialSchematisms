from typing import Annotated

from fastapi import Depends, Request

from notarius_core.application.collaboration import CollaborationService
from notarius_core.application.saved_graphs import SavedGraphService

from notarius_api.app_state import get_resources


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
