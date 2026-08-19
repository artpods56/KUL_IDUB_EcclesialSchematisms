"""Phase 4 authenticated graph-room WebSocket protocol tests."""

import asyncio
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from queue import Queue
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from grafy_api.main import create_app
from grafy_api.settings import Settings
from grafy_api.v1.routes.auth.dependencies import browser_actor
from grafy_api.v1.routes.collaboration.hub import (
    CLOSE_SLOW_CONSUMER,
    OUTBOUND_QUEUE_MAXSIZE,
    GraphRoomHub,
    GraphRoomSession,
)
from grafy_api.v1.routes.collaboration.models import (
    ActorPresentation,
    PresenceUpdateSubmitMessage,
    RoomHeartbeatMessage,
    RoomReadyMessage,
)
from grafy_core.domain.identity import (
    ActorContext,
    User,
    Workspace,
    WorkspaceMembership,
    WorkspaceRole,
)
from grafy_persistence.database import create_database
from grafy_persistence.orm import metadata
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork

from tests.unit.api.conftest import (
    TEST_USER_ID,
    WORKSPACE_ID,
    workspace_api_path,
)

FIXTURES = Path(__file__).parent / "fixtures"


PUBLIC_ORIGIN = "http://localhost:3000"
EDITOR_USER_ID = UUID(int=11)
VIEWER_USER_ID = UUID(int=12)
STRANGER_USER_ID = UUID(int=13)


class ActorSwitcher:
    def __init__(self, user_id: UUID) -> None:
        self.user_id = user_id

    def install(self, application: FastAPI) -> None:
        switcher = self

        def override() -> ActorContext:
            return ActorContext(
                user_id=switcher.user_id,
                credential_reference="test-session",
            )

        application.dependency_overrides[browser_actor] = override

    def as_user(self, user_id: UUID) -> None:
        self.user_id = user_id


async def _seed_room_users(database_url: str) -> None:
    database = create_database(database_url)
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            for user_id, email, name in (
                (TEST_USER_ID, "owner@example.test", "Owner"),
                (EDITOR_USER_ID, "editor@example.test", "Editor"),
                (VIEWER_USER_ID, "viewer@example.test", "Viewer"),
                (STRANGER_USER_ID, "stranger@example.test", "Stranger"),
            ):
                await unit_of_work.identity.add_user(
                    User(id=user_id, email=email, display_name=name)
                )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=WORKSPACE_ID,
                    slug="local",
                    name="Local workspace",
                    kind="shared",
                )
            )
            for user_id, role in (
                (TEST_USER_ID, WorkspaceRole.OWNER),
                (EDITOR_USER_ID, WorkspaceRole.EDITOR),
                (VIEWER_USER_ID, WorkspaceRole.VIEWER),
            ):
                await unit_of_work.identity.add_membership(
                    WorkspaceMembership(
                        workspace_id=WORKSPACE_ID,
                        user_id=user_id,
                        role=role,
                    )
                )
            await unit_of_work.commit()
    finally:
        await database.dispose()


def _build_room_client(
    tmp_path: Path,
    *,
    graph_room_heartbeat_seconds: float = 0.0,
) -> tuple[TestClient, ActorSwitcher, FastAPI]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'room.sqlite3'}"
    asyncio.run(_seed_room_users(database_url))
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
            public_origin=PUBLIC_ORIGIN,
            graph_room_heartbeat_seconds=graph_room_heartbeat_seconds,
        )
    )
    switcher = ActorSwitcher(TEST_USER_ID)
    switcher.install(application)
    return TestClient(application), switcher, application


@pytest.fixture
def room_client(tmp_path: Path) -> Iterator[tuple[TestClient, ActorSwitcher]]:
    client, switcher, _application = _build_room_client(tmp_path)
    with client:
        yield client, switcher


@pytest.fixture
def room_app_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, ActorSwitcher, FastAPI]]:
    client, switcher, application = _build_room_client(tmp_path)
    with client:
        yield client, switcher, application


@pytest.fixture
def heartbeat_room_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, ActorSwitcher]]:
    client, switcher, _application = _build_room_client(
        tmp_path,
        graph_room_heartbeat_seconds=0.05,
    )
    with client:
        yield client, switcher


def _create_graph(client: TestClient, name: str = "Room graph") -> UUID:
    response = client.post(
        workspace_api_path("/graphs"),
        json={"name": name, "nodes": [], "edges": []},
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["id"])


def _room_path(graph_id: UUID) -> str:
    return f"/v1/workspaces/{WORKSPACE_ID}/graphs/{graph_id}/room"


def _connect(client: TestClient, graph_id: UUID):
    return client.websocket_connect(
        _room_path(graph_id),
        headers={"Origin": PUBLIC_ORIGIN},
    )


def _receive_until(websocket, message_type: str, *, limit: int = 20) -> dict:
    for _ in range(limit):
        message = websocket.receive_json()
        if message["type"] == message_type:
            return message
    raise AssertionError(f"did not receive {message_type!r} within {limit} messages")


def _receive_execution_active_status(
    websocket,
    status: str,
    *,
    limit: int = 20,
) -> dict:
    for _ in range(limit):
        message = websocket.receive_json()
        if (
            message.get("type") == "execution.active"
            and message.get("execution", {}).get("status") == status
        ):
            return message
    raise AssertionError(
        f"did not receive execution.active status={status!r} within {limit} messages"
    )


def test_room_ready_fixture_matches_protocol_model() -> None:
    payload = json.loads((FIXTURES / "graph_room_ready.v1.json").read_text())
    ready = RoomReadyMessage.model_validate(payload)
    assert ready.type == "room.ready"
    assert ready.protocol_version == 1
    assert ready.active_execution is None


def test_room_ready_admits_authenticated_member(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, _switcher = room_client
    graph_id = _create_graph(client)

    with _connect(client, graph_id) as websocket:
        ready = websocket.receive_json()

    assert ready["type"] == "room.ready"
    assert ready["protocol_version"] == 1
    assert ready["workspace_id"] == str(WORKSPACE_ID)
    assert ready["graph_id"] == str(graph_id)
    assert ready["actor"]["actor_id"] == str(TEST_USER_ID)
    assert ready["actor"]["display_name"] == "Owner"
    assert ready["capabilities"]["authorization_version"] >= 1
    assert "edit_graph" in ready["capabilities"]["capabilities"]
    assert ready["head"]["graph_id"] == str(graph_id)
    assert ready["head"]["collaboration_sequence"] == 1
    assert ready["graph_room_session_id"]
    assert len(ready["participants"]) == 1
    assert ready["participants"][0]["graph_room_session_id"] == (
        ready["graph_room_session_id"]
    )
    assert ready["participants"][0]["actor"]["actor_id"] == str(TEST_USER_ID)
    assert ready["active_execution"] is None


def test_join_ready_precedes_command_committed_after_head_snapshot(
    room_app_client: tuple[TestClient, ActorSwitcher, FastAPI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command committed across the join snapshot window cannot be missed."""

    client, _switcher, application = room_app_client
    graph_id = _create_graph(client)
    initial_head_response = client.get(
        workspace_api_path(f"/graphs/{graph_id}/head")
    )
    assert initial_head_response.status_code == 200, initial_head_response.text
    initial_head = initial_head_response.json()

    collaboration = application.state.resources.collaboration
    original_get_head = collaboration.get_head
    snapshot_captured = threading.Event()
    release_snapshot = threading.Event()

    async def get_head_after_snapshot(
        *,
        actor: ActorContext,
        workspace_id: UUID,
        graph_id: UUID,
    ):
        head = await original_get_head(
            actor=actor,
            workspace_id=workspace_id,
            graph_id=graph_id,
        )
        snapshot_captured.set()
        await asyncio.to_thread(release_snapshot.wait)
        return head

    monkeypatch.setattr(collaboration, "get_head", get_head_after_snapshot)

    received: Queue[dict] = Queue()
    failures: Queue[Exception] = Queue()
    first_message_received = threading.Event()

    def receive_join_messages() -> None:
        try:
            with _connect(client, graph_id) as websocket:
                received.put(websocket.receive_json())
                first_message_received.set()
                received.put(websocket.receive_json())
        except Exception as exc:
            failures.put(exc)
            first_message_received.set()

    receiver = threading.Thread(target=receive_join_messages, daemon=True)
    receiver.start()
    try:
        assert snapshot_captured.wait(timeout=5)
        raced_command_id = str(uuid4())
        raced_response = client.post(
            workspace_api_path(f"/graphs/{graph_id}/commands"),
            json={
                "command_id": raced_command_id,
                "room_epoch": initial_head["room_epoch"],
                "observed_sequence": initial_head["collaboration_sequence"],
                "command": {
                    "kind": "rename_graph",
                    "name": "Committed during join",
                    "expected_name": initial_head["name"],
                },
            },
        )
        assert raced_response.status_code == 200, raced_response.text
        raced_head = raced_response.json()["head"]
    finally:
        release_snapshot.set()

    assert first_message_received.wait(timeout=5)
    if not failures.empty():
        raise failures.get_nowait()
    first = received.get(timeout=1)

    # This later fanout guarantees the receiver terminates even if the raced
    # accepted event was lost; the assertion below then identifies the loss.
    marker_response = client.post(
        workspace_api_path(f"/graphs/{graph_id}/commands"),
        json={
            "command_id": str(uuid4()),
            "room_epoch": raced_head["room_epoch"],
            "observed_sequence": raced_head["collaboration_sequence"],
            "command": {
                "kind": "rename_graph",
                "name": "Post-ready marker",
                "expected_name": raced_head["name"],
            },
        },
    )
    assert marker_response.status_code == 200, marker_response.text

    receiver.join(timeout=5)
    assert not receiver.is_alive()
    if not failures.empty():
        raise failures.get_nowait()
    second = received.get(timeout=1)

    assert first["type"] == "room.ready"
    assert first["head"]["collaboration_sequence"] == (
        initial_head["collaboration_sequence"]
    )
    assert second["type"] == "graph.command.accepted"
    assert second["command_id"] == raced_command_id
    assert second["sequence"] == raced_head["collaboration_sequence"]
    assert second["command"]["name"] == "Committed during join"


def test_room_rejects_invalid_origin(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, _switcher = room_client
    graph_id = _create_graph(client)

    with pytest.raises(WebSocketDenialResponse) as denied:
        with client.websocket_connect(
            _room_path(graph_id),
            headers={"Origin": "http://evil.example"},
        ):
            pass
    assert denied.value.status_code == 403


def test_room_rejects_non_member(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, switcher = room_client
    graph_id = _create_graph(client)
    switcher.as_user(STRANGER_USER_ID)

    with pytest.raises(WebSocketDenialResponse) as denied:
        with _connect(client, graph_id):
            pass
    assert denied.value.status_code == 404


def test_two_sessions_converge_on_accepted_sequence_and_head(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    """Phase 4 exit: two sessions on the same room observe one accepted head."""

    client, switcher = room_client
    graph_id = _create_graph(client)

    with _connect(client, graph_id) as owner_ws:
        owner_ready = owner_ws.receive_json()
        switcher.as_user(EDITOR_USER_ID)
        with _connect(client, graph_id) as editor_ws:
            editor_ready = editor_ws.receive_json()
            assert editor_ready["type"] == "room.ready"
            assert editor_ready["workspace_id"] == owner_ready["workspace_id"]
            assert editor_ready["graph_id"] == owner_ready["graph_id"]
            assert (
                editor_ready["head"]["collaboration_sequence"]
                == owner_ready["head"]["collaboration_sequence"]
            )

            command_id = str(uuid4())
            expected_sequence = editor_ready["head"]["collaboration_sequence"] + 1
            editor_ws.send_json(
                {
                    "protocol_version": 1,
                    "type": "graph.command.submit",
                    "command_id": command_id,
                    "room_epoch": editor_ready["head"]["room_epoch"],
                    "observed_sequence": editor_ready["head"]["collaboration_sequence"],
                    "command": {
                        "kind": "rename_graph",
                        "name": "Renamed live",
                        "expected_name": editor_ready["head"]["name"],
                    },
                }
            )

            editor_accepted = _receive_until(editor_ws, "graph.command.accepted")
            editor_receipt = _receive_until(editor_ws, "graph.command.receipt")
            assert editor_accepted["sequence"] == expected_sequence
            assert editor_receipt["accepted_sequence"] == expected_sequence
            assert editor_receipt["current_sequence"] == expected_sequence
            assert editor_receipt["deduplicated"] is False

            owner_accepted = _receive_until(owner_ws, "graph.command.accepted")
            assert owner_accepted["command_id"] == command_id
            assert owner_accepted["sequence"] == expected_sequence
            assert owner_accepted["command"]["name"] == "Renamed live"
            assert owner_accepted["actor"]["actor_id"] == str(EDITOR_USER_ID)

    head = client.get(workspace_api_path(f"/graphs/{graph_id}/head"))
    assert head.status_code == 200, head.text
    assert head.json()["collaboration_sequence"] == expected_sequence
    assert head.json()["name"] == "Renamed live"


def test_reconnect_idempotent_retry_does_not_double_apply(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    """Phase 4 exit: reconnect + same command id returns receipt without rebroadcast."""

    client, switcher = room_client
    graph_id = _create_graph(client)
    command_id = str(uuid4())

    with _connect(client, graph_id) as first_ws:
        ready = first_ws.receive_json()
        submit = {
            "protocol_version": 1,
            "type": "graph.command.submit",
            "command_id": command_id,
            "room_epoch": ready["head"]["room_epoch"],
            "observed_sequence": ready["head"]["collaboration_sequence"],
            "command": {
                "kind": "rename_graph",
                "name": "Once only",
                "expected_name": ready["head"]["name"],
            },
        }
        first_ws.send_json(submit)
        first_accepted = _receive_until(first_ws, "graph.command.accepted")
        first_receipt = _receive_until(first_ws, "graph.command.receipt")
        assert first_receipt["outcome"] == "accepted"
        accepted_sequence = first_accepted["sequence"]

    switcher.as_user(EDITOR_USER_ID)
    with _connect(client, graph_id) as peer_ws:
        peer_ready = peer_ws.receive_json()
        assert peer_ready["head"]["collaboration_sequence"] == accepted_sequence
        assert peer_ready["head"]["name"] == "Once only"

        switcher.as_user(TEST_USER_ID)
        with _connect(client, graph_id) as retry_ws:
            retry_ready = retry_ws.receive_json()
            assert retry_ready["head"]["collaboration_sequence"] == accepted_sequence
            retry_ws.send_json(submit)
            receipt = _receive_until(retry_ws, "graph.command.receipt")
            assert receipt["outcome"] == "idempotent_replay"
            assert receipt["deduplicated"] is True
            assert receipt["accepted_sequence"] == accepted_sequence
            assert receipt["current_sequence"] == accepted_sequence

            # Peer must not observe a second accepted fanout for the retry.
            peer_ws.send_json(
                {
                    "protocol_version": 1,
                    "type": "graph.command.submit",
                    "command_id": str(uuid4()),
                    "room_epoch": peer_ready["head"]["room_epoch"],
                    "observed_sequence": accepted_sequence,
                    "command": {
                        "kind": "rename_graph",
                        "name": "Peer marker",
                        "expected_name": "Once only",
                    },
                }
            )
            accepted = _receive_until(peer_ws, "graph.command.accepted")
            receipt = _receive_until(peer_ws, "graph.command.receipt")
            assert accepted["command"]["name"] == "Peer marker"
            assert accepted["sequence"] == accepted_sequence + 1
            assert receipt["accepted_sequence"] == accepted_sequence + 1

    head = client.get(workspace_api_path(f"/graphs/{graph_id}/head"))
    assert head.status_code == 200, head.text
    assert head.json()["collaboration_sequence"] == accepted_sequence + 1
    assert head.json()["name"] == "Peer marker"


def test_viewer_command_is_rejected_without_fanout(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, switcher = room_client
    graph_id = _create_graph(client)

    with _connect(client, graph_id) as owner_ws:
        owner_ws.receive_json()
        switcher.as_user(VIEWER_USER_ID)
        with _connect(client, graph_id) as viewer_ws:
            ready = viewer_ws.receive_json()
            assert "edit_graph" not in ready["capabilities"]["capabilities"]
            viewer_ws.send_json(
                {
                    "protocol_version": 1,
                    "type": "graph.command.submit",
                    "command_id": str(uuid4()),
                    "room_epoch": ready["head"]["room_epoch"],
                    "observed_sequence": ready["head"]["collaboration_sequence"],
                    "command": {
                        "kind": "rename_graph",
                        "name": "Nope",
                        "expected_name": ready["head"]["name"],
                    },
                }
            )
            rejected = _receive_until(viewer_ws, "graph.command.rejected")
            assert rejected["error_code"] == "forbidden"


def test_role_change_closes_with_permissions_changed(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, switcher = room_client
    graph_id = _create_graph(client)
    switcher.as_user(EDITOR_USER_ID)

    with _connect(client, graph_id) as editor_ws:
        editor_ws.receive_json()
        switcher.as_user(TEST_USER_ID)
        response = client.patch(
            workspace_api_path(f"/members/{EDITOR_USER_ID}"),
            json={"role": "viewer"},
        )
        assert response.status_code == 200, response.text

        with pytest.raises(WebSocketDisconnect) as closed:
            editor_ws.receive_json()
        assert closed.value.code == 4003
        assert closed.value.reason == "permissions_changed"

    # Reconnect as viewer and confirm fresh capability snapshot.
    switcher.as_user(EDITOR_USER_ID)
    with _connect(client, graph_id) as viewer_ws:
        ready = viewer_ws.receive_json()
    assert "edit_graph" not in ready["capabilities"]["capabilities"]
    assert "view_graph" in ready["capabilities"]["capabilities"]
    assert ready["capabilities"]["authorization_version"] >= 2


def test_http_epoch_reset_rehydrates_connected_sessions(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, _switcher = room_client
    graph_id = _create_graph(client)
    detail = client.get(workspace_api_path(f"/graphs/{graph_id}"))
    assert detail.status_code == 200
    revision = detail.json()["revision"]

    with _connect(client, graph_id) as websocket:
        ready = websocket.receive_json()
        replace = client.put(
            workspace_api_path(f"/graphs/{graph_id}"),
            json={
                "name": "Replaced document",
                "nodes": [],
                "edges": [],
                "expected_revision": revision,
            },
        )
        assert replace.status_code == 200, replace.text
        rehydrate = _receive_until(websocket, "room.rehydrate")
        assert rehydrate["reason"] == "epoch_reset"
        assert rehydrate["head"]["name"] == "Replaced document"
        assert rehydrate["head"]["collaboration_sequence"] == 0
        assert rehydrate["head"]["room_epoch"] != ready["head"]["room_epoch"]


def test_http_command_publishes_to_room(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, _switcher = room_client
    graph_id = _create_graph(client)

    with _connect(client, graph_id) as websocket:
        ready = websocket.receive_json()
        command_id = str(uuid4())
        response = client.post(
            workspace_api_path(f"/graphs/{graph_id}/commands"),
            json={
                "command_id": command_id,
                "room_epoch": ready["head"]["room_epoch"],
                "observed_sequence": ready["head"]["collaboration_sequence"],
                "command": {
                    "kind": "rename_graph",
                    "name": "HTTP rename",
                    "expected_name": ready["head"]["name"],
                },
            },
        )
        assert response.status_code == 200, response.text
        accepted = _receive_until(websocket, "graph.command.accepted")
        assert accepted["command_id"] == command_id
        assert accepted["command"]["name"] == "HTTP rename"


def test_room_sends_application_heartbeat(
    heartbeat_room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, _switcher = heartbeat_room_client
    graph_id = _create_graph(client)

    with _connect(client, graph_id) as websocket:
        ready = websocket.receive_json()
        heartbeat = websocket.receive_json()

    assert heartbeat["type"] == "room.heartbeat"
    assert heartbeat["protocol_version"] == 1
    assert (
        heartbeat["authorization_version"]
        == ready["capabilities"]["authorization_version"]
    )
    RoomHeartbeatMessage.model_validate(heartbeat)


def test_heartbeat_revalidation_closes_on_lost_role_invalidation(
    heartbeat_room_client: tuple[TestClient, ActorSwitcher],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Heartbeat covers lost post-commit room invalidation (auth tenancy design)."""

    async def _skip_close(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "grafy_api.v1.routes.workspaces.views.close_user_rooms_for_permission_change",
        _skip_close,
    )

    client, switcher = heartbeat_room_client
    graph_id = _create_graph(client)
    switcher.as_user(EDITOR_USER_ID)

    with _connect(client, graph_id) as editor_ws:
        editor_ws.receive_json()
        switcher.as_user(TEST_USER_ID)
        response = client.patch(
            workspace_api_path(f"/members/{EDITOR_USER_ID}"),
            json={"role": "viewer"},
        )
        assert response.status_code == 200, response.text

        with pytest.raises(WebSocketDisconnect) as closed:
            while True:
                message = editor_ws.receive_json()
                # A heartbeat enqueued before the role commit may still arrive.
                assert message["type"] == "room.heartbeat"
        assert closed.value.code == 4003
        assert closed.value.reason == "permissions_changed"


def test_presence_join_leave_fanout_and_room_ready_participants(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, switcher = room_client
    graph_id = _create_graph(client)

    with _connect(client, graph_id) as owner_ws:
        owner_ready = owner_ws.receive_json()
        assert len(owner_ready["participants"]) == 1
        assert owner_ready["participants"][0]["graph_room_session_id"] == (
            owner_ready["graph_room_session_id"]
        )

        switcher.as_user(EDITOR_USER_ID)
        with _connect(client, graph_id) as editor_ws:
            editor_ready = editor_ws.receive_json()
            participant_ids = {
                item["graph_room_session_id"] for item in editor_ready["participants"]
            }
            assert participant_ids == {
                owner_ready["graph_room_session_id"],
                editor_ready["graph_room_session_id"],
            }

            join = _receive_until(owner_ws, "presence.join")
            assert join["participant"]["graph_room_session_id"] == (
                editor_ready["graph_room_session_id"]
            )
            assert join["participant"]["actor"]["actor_id"] == str(EDITOR_USER_ID)

            editor_ws.send_json(
                {
                    "protocol_version": 1,
                    "type": "presence.update",
                    "presence_sequence": 1,
                    "cursor": {"x": 12.5, "y": -4.0},
                    "selected_node_ids": ["node-a"],
                    "selected_edge_ids": [],
                    "activity": "editing_node",
                    "activity_target_ids": ["node-a"],
                    "transient_node_positions": [],
                }
            )
            update = _receive_until(owner_ws, "presence.update")
            assert update["participant"]["presence_sequence"] == 1
            assert update["participant"]["cursor"] == {"x": 12.5, "y": -4.0}
            assert update["participant"]["selected_node_ids"] == ["node-a"]
            assert update["participant"]["activity"] == "editing_node"

        leave = _receive_until(owner_ws, "presence.leave")
        assert leave["graph_room_session_id"] == editor_ready["graph_room_session_id"]


def test_presence_cleared_on_access_revoked(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, switcher = room_client
    graph_id = _create_graph(client)

    with _connect(client, graph_id) as owner_ws:
        owner_ws.receive_json()
        switcher.as_user(EDITOR_USER_ID)
        with _connect(client, graph_id) as editor_ws:
            editor_ready = editor_ws.receive_json()
            _receive_until(owner_ws, "presence.join")

            switcher.as_user(TEST_USER_ID)
            response = client.delete(workspace_api_path(f"/members/{EDITOR_USER_ID}"))
            assert response.status_code == 204, response.text

            with pytest.raises(WebSocketDisconnect) as closed:
                while True:
                    message = editor_ws.receive_json()
                    assert message["type"] in {"presence.update", "room.heartbeat"}
            assert closed.value.code == 4004
            assert closed.value.reason == "access_revoked"

            leave = _receive_until(owner_ws, "presence.leave")
            assert leave["graph_room_session_id"] == editor_ready["graph_room_session_id"]


def test_presence_rate_limit_and_stale_sequence_drop() -> None:
    async def _exercise() -> None:
        hub = GraphRoomHub(presence_max_updates_per_second=20.0)
        graph_id = uuid4()
        websocket = AsyncMock()
        websocket.application_state = WebSocketState.CONNECTED
        peer_ws = AsyncMock()
        peer_ws.application_state = WebSocketState.CONNECTED
        session = GraphRoomSession(
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            graph_room_session_id=uuid4(),
            actor_user_id=TEST_USER_ID,
            credential_reference="test-session",
            authorization_version=1,
            actor_presentation=ActorPresentation(
                actor_id=TEST_USER_ID,
                display_name="Owner",
                color="indigo",
            ),
            websocket=websocket,
        )
        peer = GraphRoomSession(
            workspace_id=WORKSPACE_ID,
            graph_id=graph_id,
            graph_room_session_id=uuid4(),
            actor_user_id=EDITOR_USER_ID,
            credential_reference="test-session",
            authorization_version=1,
            actor_presentation=ActorPresentation(
                actor_id=EDITOR_USER_ID,
                display_name="Editor",
                color="emerald",
            ),
            websocket=peer_ws,
        )
        await hub.join(session)
        await hub.join(peer)
        await hub.register_presence(session)
        await hub.register_presence(peer)
        # Drain join messages.
        while not peer.outbound.empty():
            peer.outbound.get_nowait()

        from grafy_api.v1.routes.collaboration.models import (
            PresenceUpdateSubmitMessage,
        )

        first = await hub.apply_presence_update(
            session,
            PresenceUpdateSubmitMessage(
                presence_sequence=1,
                cursor={"x": 1.0, "y": 2.0},
            ),
        )
        assert first is not None
        stale = await hub.apply_presence_update(
            session,
            PresenceUpdateSubmitMessage(
                presence_sequence=1,
                cursor={"x": 9.0, "y": 9.0},
            ),
        )
        assert stale is None
        burst = await hub.apply_presence_update(
            session,
            PresenceUpdateSubmitMessage(
                presence_sequence=2,
                cursor={"x": 3.0, "y": 4.0},
            ),
        )
        assert burst is None
        await hub.shutdown()

    asyncio.run(_exercise())


def _room_session(websocket: AsyncMock) -> GraphRoomSession:
    return GraphRoomSession(
        workspace_id=WORKSPACE_ID,
        graph_id=uuid4(),
        graph_room_session_id=uuid4(),
        actor_user_id=TEST_USER_ID,
        credential_reference="test-session",
        authorization_version=1,
        actor_presentation=ActorPresentation(
            actor_id=TEST_USER_ID,
            display_name="Owner",
            color="indigo",
        ),
        websocket=websocket,
    )


def test_presence_update_after_close_cannot_recreate_membership() -> None:
    """A delayed presence update after close is dropped; it must not recreate
    presence for an already-closed (unregistered) participant."""

    async def _exercise() -> None:
        hub = GraphRoomHub()
        websocket = AsyncMock()
        websocket.application_state = WebSocketState.CONNECTED
        session = _room_session(websocket)
        await hub.join(session)
        await hub.register_presence(session)
        await hub.close_session(session, code=1000, reason="left")

        # A stale update racing after close: no member remains, so it is dropped.
        result = await hub.apply_presence_update(
            session,
            PresenceUpdateSubmitMessage(
                presence_sequence=1,
                cursor={"x": 1.0, "y": 2.0},
            ),
        )
        assert result is None
        # Presence cannot be recreated: no participant is present for the graph.
        assert await hub.participants_for(
            workspace_id=WORKSPACE_ID,
            graph_id=session.graph_id,
        ) == []
        await hub.shutdown()

    asyncio.run(_exercise())


def test_presence_update_race_with_close_never_recreates_after_close() -> None:
    """Even when an update arrives between join and close, a subsequent close
    removes the member; a later update cannot resurrect presence."""

    async def _exercise() -> None:
        hub = GraphRoomHub()
        websocket = AsyncMock()
        websocket.application_state = WebSocketState.CONNECTED
        session = _room_session(websocket)
        await hub.join(session)
        await hub.register_presence(session)

        first = await hub.apply_presence_update(
            session,
            PresenceUpdateSubmitMessage(
                presence_sequence=1,
                cursor={"x": 3.0, "y": 4.0},
            ),
        )
        assert first is not None
        await hub.close_session(session, code=1000, reason="left")

        # After close the exact session is gone; a late update is dropped.
        late = await hub.apply_presence_update(
            session,
            PresenceUpdateSubmitMessage(
                presence_sequence=2,
                cursor={"x": 9.0, "y": 9.0},
            ),
        )
        assert late is None
        assert await hub.participants_for(
            workspace_id=WORKSPACE_ID,
            graph_id=session.graph_id,
        ) == []
        await hub.shutdown()

    asyncio.run(_exercise())


def test_slow_consumer_is_disconnected_instead_of_unbounded_queue() -> None:
    """Phase 4 exit: one slow connection cannot grow an unbounded send queue."""

    async def _exercise() -> None:
        hub = GraphRoomHub()
        websocket = AsyncMock()
        websocket.application_state = WebSocketState.CONNECTED
        session = GraphRoomSession(
            workspace_id=WORKSPACE_ID,
            graph_id=uuid4(),
            graph_room_session_id=uuid4(),
            actor_user_id=TEST_USER_ID,
            credential_reference="test-session",
            authorization_version=1,
            actor_presentation=ActorPresentation(
                actor_id=TEST_USER_ID,
                display_name="Owner",
                color="indigo",
            ),
            websocket=websocket,
        )
        await hub.join(session)
        filler = RoomHeartbeatMessage(authorization_version=1)
        for _ in range(OUTBOUND_QUEUE_MAXSIZE):
            session.outbound.put_nowait(filler)

        await hub.deliver_private(session, RoomHeartbeatMessage(authorization_version=1))

        assert session.closed is True
        websocket.close.assert_awaited()
        close_kwargs = websocket.close.await_args.kwargs
        assert close_kwargs["code"] == CLOSE_SLOW_CONSUMER[0]
        assert close_kwargs["reason"] == CLOSE_SLOW_CONSUMER[1]
        await hub.shutdown()

    asyncio.run(_exercise())


def test_activated_session_has_one_fifo_post_activation_writer() -> None:
    """After activation every message routes through the hub's single sender.

    A blocked socket must not allow direct writes to overtake queued messages;
    the activated session owns exactly one sender task, so ``deliver_private``
    messages are delivered strictly in FIFO order.
    """

    async def _exercise() -> None:
        hub = GraphRoomHub()
        websocket = AsyncMock()
        websocket.application_state = WebSocketState.CONNECTED
        session = GraphRoomSession(
            workspace_id=WORKSPACE_ID,
            graph_id=uuid4(),
            graph_room_session_id=uuid4(),
            actor_user_id=TEST_USER_ID,
            credential_reference="test-session",
            authorization_version=1,
            actor_presentation=ActorPresentation(
                actor_id=TEST_USER_ID,
                display_name="Owner",
                color="indigo",
            ),
            websocket=websocket,
        )
        await hub.join(session)
        await hub.activate(session)

        # Simulate a blocked socket: the sender task awaits an event that never
        # fires until after both messages have been enqueued, so the queue is
        # the only path and delivery is strictly FIFO.
        blocked = asyncio.Event()

        async def _blocking_send(payload: dict) -> None:
            await blocked.wait()

        websocket.send_json.side_effect = _blocking_send

        first = RoomHeartbeatMessage(authorization_version=1)
        second = RoomHeartbeatMessage(authorization_version=2)
        await hub.deliver_private(session, first)
        await hub.deliver_private(session, second)

        await asyncio.sleep(0.05)
        # Unblock the single sender so it drains the queue in order.
        blocked.set()
        await asyncio.sleep(0.05)
        calls = websocket.send_json.await_args_list
        sent = [call.args[0]["authorization_version"] for call in calls]
        assert sent == [1, 2], sent
        assert len(calls) == 2

        # Exactly one sender task exists for this activated session.
        assert session.sender_task is not None
        await hub.shutdown()

    asyncio.run(_exercise())


def _create_graph_with_revision(

    client: TestClient,
    name: str = "Execution room graph",
) -> tuple[UUID, int]:
    response = client.post(
        workspace_api_path("/graphs"),
        json={"name": name, "nodes": [], "edges": []},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    return UUID(payload["id"]), int(payload["revision"])


def _start_saved_execution(
    client: TestClient,
    *,
    graph_id: UUID,
    graph_revision: int,
) -> dict:
    response = client.post(
        workspace_api_path("/executions"),
        json={
            "nodes": [],
            "edges": [],
            "graph_id": str(graph_id),
            "graph_revision": graph_revision,
        },
    )
    assert response.status_code == 202, response.text
    return response.json()


def test_two_sessions_see_execution_start_and_complete(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    """Phase 5B: room advertises active execution lifecycle to every session."""

    client, switcher = room_client
    graph_id, revision = _create_graph_with_revision(client)

    with _connect(client, graph_id) as owner_ws:
        owner_ready = owner_ws.receive_json()
        assert owner_ready["active_execution"] is None
        switcher.as_user(EDITOR_USER_ID)
        with _connect(client, graph_id) as editor_ws:
            editor_ready = editor_ws.receive_json()
            assert editor_ready["active_execution"] is None

            switcher.as_user(TEST_USER_ID)
            started = _start_saved_execution(
                client,
                graph_id=graph_id,
                graph_revision=revision,
            )
            execution_id = started["execution_id"]

            owner_active = _receive_until(owner_ws, "execution.active")
            editor_active = _receive_until(editor_ws, "execution.active")
            assert owner_active["execution"]["execution_id"] == execution_id
            assert editor_active["execution"]["execution_id"] == execution_id
            assert owner_active["execution"]["status"] in {
                "queued",
                "running",
            }
            assert owner_active["execution"]["starter"]["actor_id"] == str(
                TEST_USER_ID
            )
            assert owner_active["execution"]["cancellable"] is True

            owner_cleared = _receive_until(owner_ws, "execution.cleared")
            editor_cleared = _receive_until(editor_ws, "execution.cleared")
            assert owner_cleared["execution_id"] == execution_id
            assert editor_cleared["execution_id"] == execution_id
            assert owner_cleared["status"] == "succeeded"
            assert editor_cleared["status"] == "succeeded"


def test_viewer_discovers_active_execution_on_room_ready_and_cannot_cancel(
    room_app_client: tuple[TestClient, ActorSwitcher, FastAPI],
) -> None:
    client, switcher, application = room_app_client
    graph_id, revision = _create_graph_with_revision(client)

    original_run = application.state.resources.run_graph.run

    async def blocking_run(*args, **kwargs):
        del args, kwargs
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    application.state.resources.run_graph.run = blocking_run
    try:
        started = _start_saved_execution(
            client,
            graph_id=graph_id,
            graph_revision=revision,
        )
        execution_id = started["execution_id"]

        switcher.as_user(VIEWER_USER_ID)
        with _connect(client, graph_id) as viewer_ws:
            ready = viewer_ws.receive_json()
            assert ready["active_execution"] is not None
            assert ready["active_execution"]["execution_id"] == execution_id
            assert ready["active_execution"]["status"] in {"queued", "running"}
            assert "execute_graph" not in ready["capabilities"]["capabilities"]
            assert "cancel_execution" not in ready["capabilities"]["capabilities"]

            observed = client.get(workspace_api_path(f"/executions/{execution_id}"))
            assert observed.status_code == 200

            denied = client.delete(workspace_api_path(f"/executions/{execution_id}"))
            assert denied.status_code == 403

        switcher.as_user(EDITOR_USER_ID)
        cancel = client.delete(workspace_api_path(f"/executions/{execution_id}"))
        assert cancel.status_code == 200
        assert cancel.json()["status"] in {"cancelling", "cancelled"}
    finally:
        application.state.resources.run_graph.run = original_run


def test_second_saved_execution_conflicts_while_active(
    room_app_client: tuple[TestClient, ActorSwitcher, FastAPI],
) -> None:
    client, switcher, application = room_app_client
    del switcher
    graph_id, revision = _create_graph_with_revision(client)

    original_run = application.state.resources.run_graph.run

    async def blocking_run(*args, **kwargs):
        del args, kwargs
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    application.state.resources.run_graph.run = blocking_run
    try:
        first = _start_saved_execution(
            client,
            graph_id=graph_id,
            graph_revision=revision,
        )
        conflict = client.post(
            workspace_api_path("/executions"),
            json={
                "nodes": [],
                "edges": [],
                "graph_id": str(graph_id),
                "graph_revision": revision,
            },
        )
        assert conflict.status_code == 409, conflict.text
        detail = conflict.json()["detail"]
        assert detail["error_code"] == "active_execution"
        assert detail["execution_id"] == first["execution_id"]

        cancel = client.delete(
            workspace_api_path(f"/executions/{first['execution_id']}")
        )
        assert cancel.status_code == 200
    finally:
        application.state.resources.run_graph.run = original_run


def test_two_sessions_see_execution_cancel(
    room_app_client: tuple[TestClient, ActorSwitcher, FastAPI],
) -> None:
    client, switcher, application = room_app_client
    graph_id, revision = _create_graph_with_revision(client)

    original_run = application.state.resources.run_graph.run

    async def blocking_run(*args, **kwargs):
        del args, kwargs
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    application.state.resources.run_graph.run = blocking_run
    try:
        with _connect(client, graph_id) as owner_ws:
            owner_ws.receive_json()
            switcher.as_user(EDITOR_USER_ID)
            with _connect(client, graph_id) as editor_ws:
                editor_ws.receive_json()

                switcher.as_user(TEST_USER_ID)
                started = _start_saved_execution(
                    client,
                    graph_id=graph_id,
                    graph_revision=revision,
                )
                execution_id = started["execution_id"]
                _receive_until(owner_ws, "execution.active")
                _receive_until(editor_ws, "execution.active")

                cancel = client.delete(
                    workspace_api_path(f"/executions/{execution_id}")
                )
                assert cancel.status_code == 200

                owner_cancelling = _receive_execution_active_status(
                    owner_ws,
                    "cancelling",
                )
                editor_cancelling = _receive_execution_active_status(
                    editor_ws,
                    "cancelling",
                )
                assert owner_cancelling["execution"]["cancellable"] is False
                assert editor_cancelling["execution"]["cancellable"] is False

                owner_cleared = _receive_until(owner_ws, "execution.cleared")
                editor_cleared = _receive_until(editor_ws, "execution.cleared")
                assert owner_cleared["status"] == "cancelled"
                assert editor_cleared["status"] == "cancelled"
    finally:
        application.state.resources.run_graph.run = original_run
