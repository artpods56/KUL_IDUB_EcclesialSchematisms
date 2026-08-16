"""Post-commit publication helpers for HTTP and room command paths."""

from uuid import UUID

from fastapi import FastAPI, Request

from grafy_core.domain.collaboration import (
    CollaborativeGraphHead,
    GraphCommand,
    GraphCommandReceipt,
)
from grafy_core.domain.identity import ActorContext

from grafy_api.app_state import get_identity, get_resources
from grafy_api.v1.routes.collaboration.hub import (
    CLOSE_ACCESS_REVOKED,
    CLOSE_GRAPH_DELETED,
    CLOSE_PERMISSIONS_CHANGED,
    GraphRoomHub,
)
from grafy_api.v1.routes.collaboration.models import (
    ActiveExecutionSummary,
    ActorPresentation,
    ExecutionActiveMessage,
    ExecutionClearedMessage,
    GraphCommandAcceptedMessage,
    RoomRehydrateMessage,
    TerminalExecutionStatus,
    actor_display_color,
    bounded_display_name,
)
from grafy_api.v1.routes.saved_graphs.models import CollaborativeHeadResponse


def graph_room_hub_from_request(request: Request) -> GraphRoomHub:
    return get_resources(request.app).graph_room_hub


async def actor_presentation_for(
    app: FastAPI,
    actor: ActorContext,
) -> ActorPresentation:
    factory = get_identity(app).identity_uow_factory
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


class ActiveExecutionRoomPublisher:
    """Post-commit fanout of active-execution discovery to graph rooms."""

    def __init__(self, hub: GraphRoomHub) -> None:
        self._hub = hub

    async def publish_active(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID,
        execution: ActiveExecutionSummary,
    ) -> None:
        await self._hub.publish_execution_active(
            workspace_id=workspace_id,
            graph_id=graph_id,
            message=ExecutionActiveMessage(execution=execution),
        )

    async def publish_cleared(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID,
        execution_id: UUID,
        status: TerminalExecutionStatus,
        graph_revision: int,
        error: str | None,
    ) -> None:
        bounded_error = None if error is None else error[:2000]
        await self._hub.publish_execution_cleared(
            workspace_id=workspace_id,
            graph_id=graph_id,
            message=ExecutionClearedMessage(
                execution_id=execution_id,
                status=status,
                graph_revision=graph_revision,
                error=bounded_error,
            ),
        )
