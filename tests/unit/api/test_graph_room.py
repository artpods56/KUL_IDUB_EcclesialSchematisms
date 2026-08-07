"""Phase 4 authenticated graph-room WebSocket protocol tests."""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from notarius_api.main import create_app
from notarius_api.settings import Settings
from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_api.v1.routes.collaboration.models import RoomReadyMessage
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


@pytest.fixture
def room_client(tmp_path: Path) -> Iterator[tuple[TestClient, ActorSwitcher]]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'room.sqlite3'}"
    asyncio.run(_seed_room_users(database_url))
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
            public_origin=PUBLIC_ORIGIN,
        )
    )
    switcher = ActorSwitcher(TEST_USER_ID)
    switcher.install(application)
    with TestClient(application) as client:
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


def test_command_fanout_to_connected_sessions(
    room_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, switcher = room_client
    graph_id = _create_graph(client)

    with _connect(client, graph_id) as owner_ws:
        owner_ready = owner_ws.receive_json()
        switcher.as_user(EDITOR_USER_ID)
        with _connect(client, graph_id) as editor_ws:
            editor_ready = editor_ws.receive_json()
            assert editor_ready["type"] == "room.ready"

            command_id = str(uuid4())
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
            editor_types = {message["type"] for message in editor_messages}
            assert editor_types == {
                "graph.command.accepted",
                "graph.command.receipt",
            }

            owner_accepted = owner_ws.receive_json()
            assert owner_accepted["type"] == "graph.command.accepted"
            assert owner_accepted["command_id"] == command_id
            assert owner_accepted["sequence"] == owner_ready["head"][
                "collaboration_sequence"
            ] + 1
            assert owner_accepted["command"]["name"] == "Renamed live"
            assert owner_accepted["actor"]["actor_id"] == str(EDITOR_USER_ID)


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
