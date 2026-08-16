from typing import Annotated

from fastapi import Depends, Request, WebSocket

from grafy_core.application.collaboration import CollaborationService

from grafy_api.app_state import get_resources
from grafy_api.v1.routes.collaboration.hub import GraphRoomHub


def graph_room_hub(request: Request) -> GraphRoomHub:
    return get_resources(request.app).graph_room_hub


def collaboration_service(request: Request) -> CollaborationService:
    return get_resources(request.app).collaboration


def graph_room_hub_ws(websocket: WebSocket) -> GraphRoomHub:
    return get_resources(websocket.app).graph_room_hub


def collaboration_service_ws(websocket: WebSocket) -> CollaborationService:
    return get_resources(websocket.app).collaboration


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
