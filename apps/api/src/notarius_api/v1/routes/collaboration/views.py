import asyncio
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, status
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from notarius_core.application.collaboration import CollaborationService
from notarius_core.domain.collaboration import CommandReceiptOutcome
from notarius_core.domain.errors import (
    CapabilityDeniedError,
    CollaborationCommandRejectedError,
    CollaborationHeadConflictError,
    CollaborationIdempotencyMismatchError,
    MissingCollaborativeHeadError,
    NotFoundError,
    UserDisabledError,
)
from notarius_core.domain.identity import (
    ActorContext,
    WorkspaceCapability,
)

from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_api.v1.routes.auth.services import SESSION_COOKIE
from notarius_api.v1.routes.collaboration.dependencies import (
    CollaborationWsDependency,
    GraphRoomHubWsDependency,
)
from notarius_api.v1.routes.collaboration.hub import (
    CLOSE_ACCESS_REVOKED,
    CLOSE_PERMISSIONS_CHANGED,
    CLOSE_PROTOCOL_ERROR,
    GraphRoomHub,
    GraphRoomSession,
)
from notarius_api.v1.routes.collaboration.models import (
    CLIENT_ROOM_MESSAGE_ADAPTER,
    CapabilitySnapshot,
    GraphCommandAcceptedMessage,
    GraphCommandReceiptMessage,
    GraphCommandRejectedMessage,
    GraphCommandSubmitMessage,
    RoomHeartbeatMessage,
    RoomReadyMessage,
    command_receipt_outcome,
)
from notarius_api.v1.routes.collaboration.publish import actor_presentation_for
from notarius_api.v1.routes.saved_graphs.models import CollaborativeHeadResponse


logger = logging.getLogger(__name__)

router = APIRouter(tags=["collaboration"])


def _http_request_from_websocket(websocket: WebSocket) -> Request:
    """Build a GET Request view of the handshake for cookie/session helpers."""

    scope = dict(websocket.scope)
    scope["type"] = "http"
    scope["method"] = "GET"
    scope.setdefault("extensions", {})
    return Request(scope)


async def _websocket_browser_actor(websocket: WebSocket) -> ActorContext:
    """Resolve the browser actor for WebSocket admission.

    Honors the same ``browser_actor`` dependency override used by HTTP tests.
    """

    override = websocket.app.dependency_overrides.get(browser_actor)
    if override is not None:
        result = override()
        if hasattr(result, "__await__"):
            return await result
        return result
    request = _http_request_from_websocket(websocket)
    return await browser_actor(
        request,
        request.cookies.get(SESSION_COOKIE),
    )


def _require_websocket_origin(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    public_origin = websocket.app.state.settings.public_origin
    if origin != public_origin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Origin validation failed",
        )


@router.websocket("/workspaces/{workspace_id}/graphs/{graph_id}/room")
async def graph_room(
    websocket: WebSocket,
    workspace_id: UUID,
    graph_id: UUID,
    hub: GraphRoomHubWsDependency,
    collaboration: CollaborationWsDependency,
    actor: Annotated[ActorContext, Depends(_websocket_browser_actor)],
) -> None:
    _require_websocket_origin(websocket)
    try:
        access = await websocket.app.state.identity_service.authorize(
            actor=actor,
            workspace_id=workspace_id,
            capability=WorkspaceCapability.JOIN_GRAPH_ROOM,
        )
        access.require(WorkspaceCapability.VIEW_GRAPH)
        head = await collaboration.get_head(
            actor=actor,
            workspace_id=workspace_id,
            graph_id=graph_id,
        )
        presentation = await actor_presentation_for(websocket.app, actor)
    except HTTPException:
        raise
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except MissingCollaborativeHeadError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except CapabilityDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except UserDisabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        ) from exc

    await websocket.accept()
    session = GraphRoomSession(
        workspace_id=workspace_id,
        graph_id=graph_id,
        graph_room_session_id=hub.new_session_id(),
        actor_user_id=actor.user_id,
        credential_reference=actor.credential_reference,
        authorization_version=access.membership.authorization_version,
        actor_presentation=presentation,
        websocket=websocket,
    )
    await hub.join(session)
    ready = RoomReadyMessage(
        workspace_id=workspace_id,
        graph_id=graph_id,
        graph_room_session_id=session.graph_room_session_id,
        actor=presentation,
        capabilities=CapabilitySnapshot(
            capabilities=tuple(
                sorted(access.capabilities, key=lambda item: item.value)
            ),
            authorization_version=access.membership.authorization_version,
        ),
        head=CollaborativeHeadResponse.from_head(head),
    )
    heartbeat_seconds = websocket.app.state.settings.graph_room_heartbeat_seconds
    try:
        await websocket.send_json(ready.model_dump(mode="json"))
        while True:
            if heartbeat_seconds > 0:
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=heartbeat_seconds,
                    )
                except TimeoutError:
                    still_open = await _revalidate_and_heartbeat(
                        websocket=websocket,
                        session=session,
                        actor=actor,
                        hub=hub,
                    )
                    if not still_open:
                        return
                    continue
            else:
                raw = await websocket.receive_json()
            try:
                message = CLIENT_ROOM_MESSAGE_ADAPTER.validate_python(raw)
            except ValidationError:
                await hub.close_session(
                    session,
                    code=CLOSE_PROTOCOL_ERROR[0],
                    reason=CLOSE_PROTOCOL_ERROR[1],
                )
                return
            if isinstance(message, GraphCommandSubmitMessage):
                await _handle_command_submit(
                    session=session,
                    actor=actor,
                    collaboration=collaboration,
                    hub=hub,
                    message=message,
                )
    except WebSocketDisconnect:
        await hub.leave(session)
    except Exception:
        logger.exception(
            "graph_room_failed workspace_id=%s graph_id=%s "
            "graph_room_session_id=%s",
            workspace_id,
            graph_id,
            session.graph_room_session_id,
        )
        await hub.leave(session)


async def _revalidate_and_heartbeat(
    *,
    websocket: WebSocket,
    session: GraphRoomSession,
    actor: ActorContext,
    hub: GraphRoomHub,
) -> bool:
    """Reauthorize membership and emit ``room.heartbeat``.

    Covers lost post-commit invalidation (auth tenancy design). Role or view
    changes close with the stable protocol reasons rather than updating
    capabilities in place.
    """

    if session.closed:
        return False
    try:
        access = await websocket.app.state.identity_service.authorize(
            actor=actor,
            workspace_id=session.workspace_id,
            capability=WorkspaceCapability.JOIN_GRAPH_ROOM,
        )
        access.require(WorkspaceCapability.VIEW_GRAPH)
    except (NotFoundError, CapabilityDeniedError, UserDisabledError):
        await hub.close_session(
            session,
            code=CLOSE_ACCESS_REVOKED[0],
            reason=CLOSE_ACCESS_REVOKED[1],
        )
        return False
    if access.membership.authorization_version != session.authorization_version:
        await hub.close_session(
            session,
            code=CLOSE_PERMISSIONS_CHANGED[0],
            reason=CLOSE_PERMISSIONS_CHANGED[1],
        )
        return False
    await hub.deliver_private(
        session,
        RoomHeartbeatMessage(
            authorization_version=access.membership.authorization_version,
        ),
    )
    return True


async def _handle_command_submit(
    *,
    session: GraphRoomSession,
    actor: ActorContext,
    collaboration: CollaborationService,
    hub: GraphRoomHub,
    message: GraphCommandSubmitMessage,
) -> None:
    try:
        head, receipt = await collaboration.accept_command(
            actor=actor,
            workspace_id=session.workspace_id,
            graph_id=session.graph_id,
            command_id=message.command_id,
            observed_sequence=message.observed_sequence,
            observed_room_epoch=message.room_epoch,
            command=message.command,
            graph_room_session_id=session.graph_room_session_id,
        )
    except CapabilityDeniedError as exc:
        await session.websocket.send_json(
            GraphCommandRejectedMessage(
                command_id=message.command_id,
                error_code="forbidden",
                detail=str(exc),
            ).model_dump(mode="json")
        )
        return
    except NotFoundError as exc:
        await session.websocket.send_json(
            GraphCommandRejectedMessage(
                command_id=message.command_id,
                error_code="not_found",
                detail=str(exc),
            ).model_dump(mode="json")
        )
        return
    except MissingCollaborativeHeadError as exc:
        await session.websocket.send_json(
            GraphCommandRejectedMessage(
                command_id=message.command_id,
                error_code="missing_head",
                detail=str(exc),
            ).model_dump(mode="json")
        )
        return
    except CollaborationCommandRejectedError as exc:
        await session.websocket.send_json(
            GraphCommandRejectedMessage(
                command_id=message.command_id,
                error_code="command_rejected",
                detail=str(exc),
            ).model_dump(mode="json")
        )
        return
    except CollaborationHeadConflictError as exc:
        await session.websocket.send_json(
            GraphCommandRejectedMessage(
                command_id=message.command_id,
                error_code="head_conflict",
                detail=str(exc),
                current_room_epoch=exc.room_epoch,
                current_sequence=exc.actual_sequence,
            ).model_dump(mode="json")
        )
        return
    except CollaborationIdempotencyMismatchError as exc:
        await session.websocket.send_json(
            GraphCommandRejectedMessage(
                command_id=message.command_id,
                error_code="idempotency_mismatch",
                detail=str(exc),
            ).model_dump(mode="json")
        )
        return

    deduplicated = receipt.outcome is CommandReceiptOutcome.IDEMPOTENT_REPLAY
    receipt_message = GraphCommandReceiptMessage(
        command_id=message.command_id,
        outcome=command_receipt_outcome(deduplicated=deduplicated),
        accepted_room_epoch=receipt.room_epoch,
        accepted_sequence=receipt.accepted_sequence,
        current_room_epoch=head.room_epoch,
        current_sequence=head.collaboration_sequence,
        deduplicated=deduplicated,
        requires_head_rehydration=receipt.room_epoch != head.room_epoch,
    )
    if deduplicated:
        # Design: receipt answers an idempotent retry and is never rebroadcast.
        await hub.deliver_private(session, receipt_message)
        return

    accepted = GraphCommandAcceptedMessage(
        command_id=message.command_id,
        room_epoch=receipt.room_epoch,
        sequence=receipt.accepted_sequence,
        actor=session.actor_presentation,
        graph_room_session_id=session.graph_room_session_id,
        command=message.command,
    )
    await hub.publish_accepted(
        workspace_id=session.workspace_id,
        graph_id=session.graph_id,
        accepted=accepted,
        receipt=receipt_message,
        receipt_session_id=session.graph_room_session_id,
    )
