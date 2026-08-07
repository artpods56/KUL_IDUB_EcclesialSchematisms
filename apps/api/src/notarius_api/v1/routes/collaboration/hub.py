import asyncio
import logging
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from fastapi import WebSocket
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect, WebSocketState

from notarius_api.v1.routes.collaboration.models import (
    ActorPresentation,
    GraphCommandAcceptedMessage,
    GraphCommandReceiptMessage,
    RoomRehydrateMessage,
)


logger = logging.getLogger(__name__)

OUTBOUND_QUEUE_MAXSIZE = 64

CLOSE_PERMISSIONS_CHANGED = (4003, "permissions_changed")
CLOSE_ACCESS_REVOKED = (4004, "access_revoked")
CLOSE_PROTOCOL_ERROR = (4008, "protocol_error")
CLOSE_SLOW_CONSUMER = (4009, "slow_consumer")
CLOSE_GRAPH_DELETED = (4010, "graph_deleted")


@dataclass(slots=True)
class GraphRoomSession:
    """Ephemeral WebSocket connection identity — not an authorization grant."""

    workspace_id: UUID
    graph_id: UUID
    graph_room_session_id: UUID
    actor_user_id: UUID
    credential_reference: str | None
    authorization_version: int
    actor_presentation: ActorPresentation
    websocket: WebSocket
    outbound: asyncio.Queue[BaseModel | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=OUTBOUND_QUEUE_MAXSIZE)
    )
    sender_task: asyncio.Task[None] | None = None
    closed: bool = False


class GraphRoomHub:
    """In-process hub keyed by (workspace_id, graph_id)."""

    def __init__(self) -> None:
        self._rooms: dict[tuple[UUID, UUID], dict[UUID, GraphRoomSession]] = {}
        self._lock = asyncio.Lock()

    async def join(self, session: GraphRoomSession) -> None:
        key = (session.workspace_id, session.graph_id)
        async with self._lock:
            room = self._rooms.setdefault(key, {})
            room[session.graph_room_session_id] = session
        session.sender_task = asyncio.create_task(self._sender_loop(session))

    async def leave(self, session: GraphRoomSession) -> None:
        await self._close_session(session, code=1000, reason="left")

    async def close_session(
        self,
        session: GraphRoomSession,
        *,
        code: int,
        reason: str,
    ) -> None:
        await self._close_session(session, code=code, reason=reason)

    def new_session_id(self) -> UUID:
        return uuid4()

    async def publish_accepted(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID,
        accepted: GraphCommandAcceptedMessage,
        receipt: GraphCommandReceiptMessage | None = None,
        receipt_session_id: UUID | None = None,
    ) -> None:
        sessions = await self._sessions_for(workspace_id, graph_id)
        for session in sessions:
            await self._enqueue(session, accepted)
            if (
                receipt is not None
                and receipt_session_id is not None
                and session.graph_room_session_id == receipt_session_id
            ):
                await self._enqueue(session, receipt)

    async def deliver_private(
        self,
        session: GraphRoomSession,
        message: BaseModel,
    ) -> None:
        """Deliver a private message (e.g. idempotent receipt) without fanout."""

        await self._enqueue(session, message)

    async def publish_rehydrate(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID,
        message: RoomRehydrateMessage,
    ) -> None:
        for session in await self._sessions_for(workspace_id, graph_id):
            await self._enqueue(session, message)

    async def close_graph(
        self,
        *,
        workspace_id: UUID,
        graph_id: UUID,
        code: int = CLOSE_GRAPH_DELETED[0],
        reason: str = CLOSE_GRAPH_DELETED[1],
    ) -> None:
        for session in await self._sessions_for(workspace_id, graph_id):
            await self._close_session(session, code=code, reason=reason)

    async def close_workspace_user(
        self,
        *,
        workspace_id: UUID,
        user_id: UUID,
        code: int,
        reason: str,
    ) -> None:
        sessions = await self._sessions_for_workspace_user(workspace_id, user_id)
        for session in sessions:
            await self._close_session(session, code=code, reason=reason)

    async def shutdown(self) -> None:
        async with self._lock:
            sessions = [
                session
                for room in self._rooms.values()
                for session in room.values()
            ]
            self._rooms.clear()
        for session in sessions:
            await self._close_session(session, code=1001, reason="shutdown")

    async def _sessions_for(
        self,
        workspace_id: UUID,
        graph_id: UUID,
    ) -> list[GraphRoomSession]:
        async with self._lock:
            room = self._rooms.get((workspace_id, graph_id), {})
            return list(room.values())

    async def _sessions_for_workspace_user(
        self,
        workspace_id: UUID,
        user_id: UUID,
    ) -> list[GraphRoomSession]:
        async with self._lock:
            matched: list[GraphRoomSession] = []
            for (room_workspace_id, _graph_id), room in self._rooms.items():
                if room_workspace_id != workspace_id:
                    continue
                for session in room.values():
                    if session.actor_user_id == user_id:
                        matched.append(session)
            return matched

    async def _enqueue(self, session: GraphRoomSession, message: BaseModel) -> None:
        if session.closed:
            return
        try:
            session.outbound.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning(
                "graph_room_slow_consumer workspace_id=%s graph_id=%s "
                "graph_room_session_id=%s",
                session.workspace_id,
                session.graph_id,
                session.graph_room_session_id,
            )
            await self._close_session(
                session,
                code=CLOSE_SLOW_CONSUMER[0],
                reason=CLOSE_SLOW_CONSUMER[1],
            )

    async def _sender_loop(self, session: GraphRoomSession) -> None:
        try:
            while True:
                message = await session.outbound.get()
                if message is None or session.closed:
                    return
                if session.websocket.application_state != WebSocketState.CONNECTED:
                    return
                await session.websocket.send_json(message.model_dump(mode="json"))
        except WebSocketDisconnect:
            return
        except Exception:
            logger.exception(
                "graph_room_sender_failed workspace_id=%s graph_id=%s "
                "graph_room_session_id=%s",
                session.workspace_id,
                session.graph_id,
                session.graph_room_session_id,
            )
            await self._close_session(
                session,
                code=CLOSE_PROTOCOL_ERROR[0],
                reason=CLOSE_PROTOCOL_ERROR[1],
            )

    async def _close_session(
        self,
        session: GraphRoomSession,
        *,
        code: int,
        reason: str,
    ) -> None:
        if session.closed:
            return
        session.closed = True
        key = (session.workspace_id, session.graph_id)
        async with self._lock:
            room = self._rooms.get(key)
            if room is not None:
                room.pop(session.graph_room_session_id, None)
                if not room:
                    self._rooms.pop(key, None)
        if session.sender_task is not None and session.sender_task is not asyncio.current_task():
            session.sender_task.cancel()
        try:
            session.outbound.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if session.websocket.application_state == WebSocketState.CONNECTED:
            try:
                await session.websocket.close(code=code, reason=reason[:123])
            except Exception:
                logger.debug(
                    "graph_room_close_failed workspace_id=%s graph_id=%s "
                    "graph_room_session_id=%s",
                    session.workspace_id,
                    session.graph_id,
                    session.graph_room_session_id,
                    exc_info=True,
                )
