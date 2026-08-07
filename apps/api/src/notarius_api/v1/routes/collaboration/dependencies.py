from typing import Annotated

from fastapi import Depends, Request, WebSocket

from notarius_core.application.collaboration import CollaborationService

from notarius_api.v1.routes.collaboration.hub import GraphRoomHub


def graph_room_hub(request: Request) -> GraphRoomHub:
    hub = getattr(request.app.state, "graph_room_hub", None)
    if not isinstance(hub, GraphRoomHub):
        raise RuntimeError("Graph room hub is not configured")
    return hub


def collaboration_service(request: Request) -> CollaborationService:
    service = getattr(request.app.state, "collaboration", None)
    if not isinstance(service, CollaborationService):
        raise RuntimeError("Collaboration service is not configured")
    return service


def graph_room_hub_ws(websocket: WebSocket) -> GraphRoomHub:
    hub = getattr(websocket.app.state, "graph_room_hub", None)
    if not isinstance(hub, GraphRoomHub):
        raise RuntimeError("Graph room hub is not configured")
    return hub


def collaboration_service_ws(websocket: WebSocket) -> CollaborationService:
    service = getattr(websocket.app.state, "collaboration", None)
    if not isinstance(service, CollaborationService):
        raise RuntimeError("Collaboration service is not configured")
    return service


GraphRoomHubDependency = Annotated[GraphRoomHub, Depends(graph_room_hub)]
CollaborationDependency = Annotated[
    CollaborationService,
    Depends(collaboration_service),
]
GraphRoomHubWsDependency = Annotated[GraphRoomHub, Depends(graph_room_hub_ws)]
CollaborationWsDependency = Annotated[
    CollaborationService,
    Depends(collaboration_service_ws),
]
