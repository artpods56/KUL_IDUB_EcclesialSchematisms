"""Phase 4 authenticated graph-room WebSocket protocol tests."""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from notarius_api.main import create_app
from notarius_api.settings import Settings
from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_api.v1.routes.collaboration.hub import (
    CLOSE_SLOW_CONSUMER,
    OUTBOUND_QUEUE_MAXSIZE,
    GraphRoomHub,
    GraphRoomSession,
)
from notarius_api.v1.routes.collaboration.models import (
    ActorPresentation,
    RoomHeartbeatMessage,
    RoomReadyMessage,
)
from notarius_core.domain.identity import (
    ActorContext,
    User,
    Workspace,
    WorkspaceMembership,
    WorkspaceRole,
)
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork

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
    assert ready["participants"] == []
    assert ready["active_execution"] is None


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

            editor_messages = [
                editor_ws.receive_json(),
                editor_ws.receive_json(),
            ]
            editor_by_type = {message["type"]: message for message in editor_messages}
            assert set(editor_by_type) == {
                "graph.command.accepted",
                "graph.command.receipt",
            }
            assert editor_by_type["graph.command.accepted"]["sequence"] == (
                expected_sequence
            )
            assert editor_by_type["graph.command.receipt"]["accepted_sequence"] == (
                expected_sequence
            )
            assert editor_by_type["graph.command.receipt"]["current_sequence"] == (
                expected_sequence
            )
            assert editor_by_type["graph.command.receipt"]["deduplicated"] is False

            owner_accepted = owner_ws.receive_json()
            assert owner_accepted["type"] == "graph.command.accepted"
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
        first_messages = [first_ws.receive_json(), first_ws.receive_json()]
        first_by_type = {message["type"]: message for message in first_messages}
        assert first_by_type["graph.command.receipt"]["outcome"] == "accepted"
        accepted_sequence = first_by_type["graph.command.accepted"]["sequence"]

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
            receipt = retry_ws.receive_json()
            assert receipt["type"] == "graph.command.receipt"
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
            peer_messages = [peer_ws.receive_json(), peer_ws.receive_json()]
            peer_types = {message["type"] for message in peer_messages}
            assert peer_types == {
                "graph.command.accepted",
                "graph.command.receipt",
            }
            accepted = next(
                message
                for message in peer_messages
                if message["type"] == "graph.command.accepted"
            )
            assert accepted["command"]["name"] == "Peer marker"
            assert accepted["sequence"] == accepted_sequence + 1

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
            rejected = viewer_ws.receive_json()
            assert rejected["type"] == "graph.command.rejected"
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
        rehydrate = websocket.receive_json()
        assert rehydrate["type"] == "room.rehydrate"
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
        accepted = websocket.receive_json()
        assert accepted["type"] == "graph.command.accepted"
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
        "notarius_api.v1.routes.workspaces.views.close_user_rooms_for_permission_change",
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
