"""Post-commit publication helpers for HTTP and room command paths."""

from uuid import UUID

from fastapi import FastAPI, Request

from notarius_core.domain.collaboration import (
    CollaborativeGraphHead,
    GraphCommand,
    GraphCommandReceipt,
)
from notarius_core.domain.identity import ActorContext

from notarius_api.v1.routes.collaboration.hub import (
    CLOSE_ACCESS_REVOKED,
    CLOSE_GRAPH_DELETED,
    CLOSE_PERMISSIONS_CHANGED,
    GraphRoomHub,
)
from notarius_api.v1.routes.collaboration.models import (
    ActorPresentation,
    GraphCommandAcceptedMessage,
    RoomRehydrateMessage,
    actor_display_color,
    bounded_display_name,
)
from notarius_api.v1.routes.saved_graphs.models import CollaborativeHeadResponse


def graph_room_hub_from_request(request: Request) -> GraphRoomHub:
    hub = getattr(request.app.state, "graph_room_hub", None)
    if not isinstance(hub, GraphRoomHub):
        raise RuntimeError("Graph room hub is not configured")
    return hub


async def actor_presentation_for(
    app: FastAPI,
    actor: ActorContext,
) -> ActorPresentation:
    factory = app.state.identity_uow_factory
    async with factory() as unit_of_work:
        user = await unit_of_work.identity.get_user(actor.user_id)
    display_name = "collaborator"
    if user is not None:
        display_name = bounded_display_name(user.display_name, user.email)
    return ActorPresentation(
        actor_id=actor.user_id,
        display_name=display_name,
        color=actor_display_color(actor.user_id),
    )


async def publish_accepted_command(
    request: Request,
    *,
    actor: ActorContext,
    workspace_id: UUID,
    graph_id: UUID,
    command: GraphCommand,
    receipt: GraphCommandReceipt,
    graph_room_session_id: UUID | None = None,
) -> None:
    hub = graph_room_hub_from_request(request)
    presentation = await actor_presentation_for(request.app, actor)
    await hub.publish_accepted(
        workspace_id=workspace_id,
        graph_id=graph_id,
        accepted=GraphCommandAcceptedMessage(
            command_id=receipt.command_id,
            room_epoch=receipt.room_epoch,
            sequence=receipt.accepted_sequence,
            actor=presentation,
            graph_room_session_id=graph_room_session_id,
            command=command,
        ),
    )


async def publish_epoch_reset(
    request: Request,
    *,
    workspace_id: UUID,
    graph_id: UUID,
    head: CollaborativeGraphHead,
) -> None:
    hub = graph_room_hub_from_request(request)
    await hub.publish_rehydrate(
        workspace_id=workspace_id,
        graph_id=graph_id,
        message=RoomRehydrateMessage(
            head=CollaborativeHeadResponse.from_head(head),
        ),
    )


async def close_graph_room(
    request: Request,
    *,
    workspace_id: UUID,
    graph_id: UUID,
) -> None:
    hub = graph_room_hub_from_request(request)
    await hub.close_graph(
        workspace_id=workspace_id,
        graph_id=graph_id,
        code=CLOSE_GRAPH_DELETED[0],
        reason=CLOSE_GRAPH_DELETED[1],
    )


async def close_user_rooms_for_permission_change(
    request: Request,
    *,
    workspace_id: UUID,
    user_id: UUID,
    access_revoked: bool,
) -> None:
    hub = graph_room_hub_from_request(request)
    if access_revoked:
        code, reason = CLOSE_ACCESS_REVOKED
    else:
        code, reason = CLOSE_PERMISSIONS_CHANGED
    await hub.close_workspace_user(
        workspace_id=workspace_id,
        user_id=user_id,
        code=code,
        reason=reason,
    )
