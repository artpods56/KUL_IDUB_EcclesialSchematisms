from typing import Annotated

from fastapi import Depends, Request

from notarius_core.application.collaboration import CollaborationService
from notarius_core.application.saved_graphs import SavedGraphService


def saved_graph_service(request: Request) -> SavedGraphService:
    service = getattr(request.app.state, "saved_graphs", None)
    if not isinstance(service, SavedGraphService):
        raise RuntimeError("Saved graph service is not initialized")
    return service


def collaboration_service(request: Request) -> CollaborationService:
    service = getattr(request.app.state, "collaboration", None)
    if not isinstance(service, CollaborationService):
        raise RuntimeError("Collaboration service is not initialized")
    return service


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
