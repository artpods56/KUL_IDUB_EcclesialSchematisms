"""Phase 7 two-session collaboration acceptance at the API/WebSocket boundary.

These tests are the automatable stand-in for the plan's two-browser journey:
owner invites collaborators, two sessions converge and share a run, a viewer
observes without mutating, membership revoke closes the victim room, and a
personal graph stays invisible. Live OIDC/SSH rehearsal remains an operator gate.
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.websockets import WebSocketDisconnect

from notarius_api.main import create_app
from notarius_api.settings import Settings
from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_core.domain.identity import (
    ActorContext,
    User,
    Workspace,
    WorkspaceMembership,
    WorkspaceRole,
    personal_workspace_slug,
)
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork


PUBLIC_ORIGIN = "http://localhost:3000"
SHARED_WORKSPACE_ID = UUID("00000000-0000-0000-0000-0000000000a1")
PERSONAL_WORKSPACE_ID = UUID("00000000-0000-0000-0000-0000000000a2")
OWNER_ID = UUID(int=21)
EDITOR_ID = UUID(int=22)
VIEWER_ID = UUID(int=23)


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


async def _seed_acceptance_users(database_url: str) -> None:
    database = create_database(database_url)
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            for user_id, email, name in (
                (OWNER_ID, "owner@acceptance.test", "Owner"),
                (EDITOR_ID, "editor@acceptance.test", "Editor"),
                (VIEWER_ID, "viewer@acceptance.test", "Viewer"),
            ):
                await unit_of_work.identity.add_user(
                    User(id=user_id, email=email, display_name=name)
                )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=PERSONAL_WORKSPACE_ID,
                    slug=personal_workspace_slug(OWNER_ID),
                    name="Owner personal",
                    kind="personal",
                    personal_owner_user_id=OWNER_ID,
                )
            )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=SHARED_WORKSPACE_ID,
                    slug="acceptance-team",
                    name="Acceptance team",
                    kind="shared",
                )
            )
            await unit_of_work.identity.add_membership(
                WorkspaceMembership(
                    workspace_id=PERSONAL_WORKSPACE_ID,
                    user_id=OWNER_ID,
                    role=WorkspaceRole.OWNER,
                )
            )
            await unit_of_work.identity.add_membership(
                WorkspaceMembership(
                    workspace_id=SHARED_WORKSPACE_ID,
                    user_id=OWNER_ID,
                    role=WorkspaceRole.OWNER,
                )
            )
            await unit_of_work.commit()
    finally:
        await database.dispose()


@pytest.fixture
def acceptance_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, ActorSwitcher]]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'acceptance.sqlite3'}"
    asyncio.run(_seed_acceptance_users(database_url))
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
            public_origin=PUBLIC_ORIGIN,
            graph_room_heartbeat_seconds=0.0,
        )
    )
    switcher = ActorSwitcher(OWNER_ID)
    switcher.install(application)
    with TestClient(application) as client:
        yield client, switcher


def _api(workspace_id: UUID, suffix: str) -> str:
    normalized = suffix if suffix.startswith("/") else f"/{suffix}"
    return f"/v1/workspaces/{workspace_id}{normalized}"


def _room_path(workspace_id: UUID, graph_id: UUID) -> str:
    return f"/v1/workspaces/{workspace_id}/graphs/{graph_id}/room"


def _connect(client: TestClient, workspace_id: UUID, graph_id: UUID):
    return client.websocket_connect(
        _room_path(workspace_id, graph_id),
        headers={"Origin": PUBLIC_ORIGIN},
    )


def _receive_until(websocket, message_type: str, *, limit: int = 30) -> dict:
    for _ in range(limit):
        message = websocket.receive_json()
        if message["type"] == message_type:
            return message
    raise AssertionError(f"did not receive {message_type!r} within {limit} messages")


def _create_graph(
    client: TestClient,
    *,
    workspace_id: UUID,
    name: str,
) -> tuple[UUID, int]:
    response = client.post(
        _api(workspace_id, "/graphs"),
        json={"name": name, "nodes": [], "edges": []},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return UUID(body["id"]), int(body["revision"])


def _add_member(
    client: TestClient,
    *,
    user_id: UUID,
    role: WorkspaceRole,
) -> None:
    response = client.post(
        _api(SHARED_WORKSPACE_ID, "/members"),
        json={"user_id": str(user_id), "role": role.value},
    )
    assert response.status_code == 200, response.text
    assert response.json()["role"] == role.value


def test_phase7_two_session_collaboration_acceptance_journey(
    acceptance_client: tuple[TestClient, ActorSwitcher],
) -> None:
    """Owner invites peers; sessions converge, share a run, and revoke cleanly."""

    client, switcher = acceptance_client

    switcher.as_user(OWNER_ID)
    personal_graph_id, _ = _create_graph(
        client,
        workspace_id=PERSONAL_WORKSPACE_ID,
        name="Private draft",
    )
    switcher.as_user(EDITOR_ID)
    denied = client.get(_api(PERSONAL_WORKSPACE_ID, f"/graphs/{personal_graph_id}"))
    assert denied.status_code == 404, denied.text
    listed = client.get(_api(PERSONAL_WORKSPACE_ID, "/graphs"))
    assert listed.status_code == 404, listed.text

    switcher.as_user(OWNER_ID)
    shared_graph_id, revision = _create_graph(
        client,
        workspace_id=SHARED_WORKSPACE_ID,
        name="Shared acceptance graph",
    )
    _add_member(client, user_id=EDITOR_ID, role=WorkspaceRole.EDITOR)
    _add_member(client, user_id=VIEWER_ID, role=WorkspaceRole.VIEWER)

    members = client.get(_api(SHARED_WORKSPACE_ID, "/members"))
    assert members.status_code == 200, members.text
    member_ids = {UUID(item["user"]["id"]) for item in members.json()}
    assert member_ids == {OWNER_ID, EDITOR_ID, VIEWER_ID}

    with _connect(client, SHARED_WORKSPACE_ID, shared_graph_id) as owner_ws:
        owner_ready = owner_ws.receive_json()
        assert owner_ready["type"] == "room.ready"
        assert owner_ready["active_execution"] is None

        switcher.as_user(EDITOR_ID)
        with _connect(client, SHARED_WORKSPACE_ID, shared_graph_id) as editor_ws:
            editor_ready = editor_ws.receive_json()
            assert editor_ready["type"] == "room.ready"
            assert {
                item["graph_room_session_id"] for item in editor_ready["participants"]
            } == {
                owner_ready["graph_room_session_id"],
                editor_ready["graph_room_session_id"],
            }

            join = _receive_until(owner_ws, "presence.join")
            assert join["participant"]["actor"]["actor_id"] == str(EDITOR_ID)
            assert join["participant"]["graph_room_session_id"] == (
                editor_ready["graph_room_session_id"]
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
                        "name": "Converged name",
                        "expected_name": editor_ready["head"]["name"],
                    },
                }
            )
            editor_accepted = _receive_until(editor_ws, "graph.command.accepted")
            editor_receipt = _receive_until(editor_ws, "graph.command.receipt")
            assert editor_accepted["sequence"] == expected_sequence
            assert editor_receipt["accepted_sequence"] == expected_sequence

            owner_accepted = _receive_until(owner_ws, "graph.command.accepted")
            assert owner_accepted["command_id"] == command_id
            assert owner_accepted["sequence"] == expected_sequence
            assert owner_accepted["command"]["name"] == "Converged name"
            assert owner_accepted["actor"]["actor_id"] == str(EDITOR_ID)

            switcher.as_user(OWNER_ID)
            started = client.post(
                _api(SHARED_WORKSPACE_ID, "/executions"),
                json={
                    "nodes": [],
                    "edges": [],
                    "graph_id": str(shared_graph_id),
                    "graph_revision": revision,
                },
            )
            assert started.status_code == 202, started.text
            execution_id = started.json()["execution_id"]

            owner_active = _receive_until(owner_ws, "execution.active")
            editor_active = _receive_until(editor_ws, "execution.active")
            assert owner_active["execution"]["execution_id"] == execution_id
            assert editor_active["execution"]["execution_id"] == execution_id
            assert owner_active["execution"]["starter"]["actor_id"] == str(OWNER_ID)

            owner_cleared = _receive_until(owner_ws, "execution.cleared")
            editor_cleared = _receive_until(editor_ws, "execution.cleared")
            assert owner_cleared["execution_id"] == execution_id
            assert editor_cleared["execution_id"] == execution_id

            switcher.as_user(VIEWER_ID)
            with _connect(client, SHARED_WORKSPACE_ID, shared_graph_id) as viewer_ws:
                viewer_ready = viewer_ws.receive_json()
                assert viewer_ready["type"] == "room.ready"
                assert viewer_ready["head"]["name"] == "Converged name"
                assert "edit_graph" not in viewer_ready["capabilities"]["capabilities"]
                assert "execute_graph" not in viewer_ready["capabilities"]["capabilities"]

                viewer_ws.send_json(
                    {
                        "protocol_version": 1,
                        "type": "graph.command.submit",
                        "command_id": str(uuid4()),
                        "room_epoch": viewer_ready["head"]["room_epoch"],
                        "observed_sequence": viewer_ready["head"][
                            "collaboration_sequence"
                        ],
                        "command": {
                            "kind": "rename_graph",
                            "name": "Viewer should fail",
                            "expected_name": viewer_ready["head"]["name"],
                        },
                    }
                )
                rejected = _receive_until(viewer_ws, "graph.command.rejected")
                assert rejected["error_code"] == "forbidden"

            switcher.as_user(OWNER_ID)
            revoke = client.delete(_api(SHARED_WORKSPACE_ID, f"/members/{EDITOR_ID}"))
            assert revoke.status_code == 204, revoke.text

            with pytest.raises(WebSocketDisconnect) as closed:
                while True:
                    message = editor_ws.receive_json()
                    assert message["type"] in {
                        "presence.update",
                        "presence.leave",
                        "presence.join",
                        "room.heartbeat",
                    }
            assert closed.value.code == 4004
            assert closed.value.reason == "access_revoked"

            for _ in range(30):
                leave = owner_ws.receive_json()
                if leave["type"] != "presence.leave":
                    continue
                if leave["graph_room_session_id"] == editor_ready["graph_room_session_id"]:
                    break
            else:
                raise AssertionError("owner did not observe editor presence.leave")

    switcher.as_user(EDITOR_ID)
    after_revoke = client.get(
        _api(SHARED_WORKSPACE_ID, f"/graphs/{shared_graph_id}/head")
    )
    assert after_revoke.status_code == 404, after_revoke.text

    switcher.as_user(OWNER_ID)
    head = client.get(_api(SHARED_WORKSPACE_ID, f"/graphs/{shared_graph_id}/head"))
    assert head.status_code == 200, head.text
    assert head.json()["name"] == "Converged name"
    assert head.json()["collaboration_sequence"] == expected_sequence
