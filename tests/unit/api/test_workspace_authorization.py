"""Phase 2 tenant IDOR and role/capability matrix for workspace-qualified routes."""

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import SecretStr

from notarius_api.main import create_app
from notarius_api.settings import Settings
from notarius_api.v1.routes.auth.dependencies import browser_actor
from notarius_core.artifacts import ArtifactObject
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

from tests.unit.api.conftest import WORKSPACE_ID


WORKSPACE_A = WORKSPACE_ID
WORKSPACE_B = UUID("00000000-0000-0000-0000-00000000000b")

OWNER_A_ID = UUID(int=1)
VIEWER_A_ID = UUID(int=2)
EDITOR_A_ID = UUID(int=3)
OWNER_B_ID = UUID(int=4)
BOTH_ID = UUID(int=5)


def _api(workspace_id: UUID, suffix: str) -> str:
    normalized = suffix if suffix.startswith("/") else f"/{suffix}"
    return f"/v1/workspaces/{workspace_id}{normalized}"


def _empty_graph_payload(name: str = "Authz graph") -> dict[str, object]:
    return {"name": name, "nodes": [], "edges": []}


@dataclass
class ActorSwitcher:
    user_id: UUID

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


async def _seed_authorization_matrix(database_url: str) -> UUID:
    """Seed two workspaces, role matrix users, and one inline artifact in A.

    Returns the seeded artifact id owned by workspace A.
    """

    database = create_database(database_url)
    artifact = ArtifactObject(
        workspace_id=WORKSPACE_A,
        artifact_type="scalar.text",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"value": "workspace-a-secret-payload"},
    )
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            for user_id, email, display_name in (
                (OWNER_A_ID, "owner-a@example.test", "Owner A"),
                (VIEWER_A_ID, "viewer-a@example.test", "Viewer A"),
                (EDITOR_A_ID, "editor-a@example.test", "Editor A"),
                (OWNER_B_ID, "owner-b@example.test", "Owner B"),
                (BOTH_ID, "both@example.test", "Both Workspaces"),
            ):
                await unit_of_work.identity.add_user(
                    User(id=user_id, email=email, display_name=display_name)
                )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=WORKSPACE_A,
                    slug="workspace-a",
                    name="Workspace A",
                    kind="shared",
                )
            )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=WORKSPACE_B,
                    slug="workspace-b",
                    name="Workspace B",
                    kind="shared",
                )
            )
            for workspace_id, user_id, role in (
                (WORKSPACE_A, OWNER_A_ID, WorkspaceRole.OWNER),
                (WORKSPACE_A, VIEWER_A_ID, WorkspaceRole.VIEWER),
                (WORKSPACE_A, EDITOR_A_ID, WorkspaceRole.EDITOR),
                (WORKSPACE_A, BOTH_ID, WorkspaceRole.EDITOR),
                (WORKSPACE_B, OWNER_B_ID, WorkspaceRole.OWNER),
                (WORKSPACE_B, BOTH_ID, WorkspaceRole.OWNER),
            ):
                await unit_of_work.identity.add_membership(
                    WorkspaceMembership(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        role=role,
                    )
                )
            await unit_of_work.artifacts.add(artifact)
            await unit_of_work.commit()
    finally:
        await database.dispose()
    return artifact.id


@pytest.fixture
def authz_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, ActorSwitcher, UUID]]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'workspace-authz.sqlite3'}"
    artifact_id = asyncio.run(_seed_authorization_matrix(database_url))
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
            auth_cookie_secure=False,
        )
    )
    actor = ActorSwitcher(user_id=OWNER_A_ID)
    actor.install(application)
    with TestClient(application) as client:
        yield client, actor, artifact_id


def _create_graph(
    client: TestClient,
    actor: ActorSwitcher,
    *,
    workspace_id: UUID,
    user_id: UUID,
    name: str,
) -> tuple[UUID, int]:
    actor.as_user(user_id)
    response = client.post(
        _api(workspace_id, "/graphs"),
        json=_empty_graph_payload(name),
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return UUID(body["id"]), int(body["revision"])


def _start_execution(
    client: TestClient,
    actor: ActorSwitcher,
    user_id: UUID,
    workspace_id: UUID,
) -> UUID:
    actor.as_user(user_id)
    response = client.post(
        _api(workspace_id, "/executions"),
        json={"nodes": [], "edges": []},
    )
    assert response.status_code == 202, response.text
    return UUID(response.json()["execution_id"])


def _assert_not_found(response: Response, *, context: object) -> None:
    assert response.status_code == 404, (context, response.status_code, response.text)
    assert response.json() == {"detail": "Not found"}


def _assert_forbidden(response: Response, *, context: object) -> None:
    assert response.status_code == 403, (context, response.status_code, response.text)
    assert response.json() == {"detail": "Forbidden"}


def test_global_graph_browser_is_authorized_and_keeps_user_state_private(
    authz_client: tuple[TestClient, ActorSwitcher, UUID],
) -> None:
    client, actor, _ = authz_client
    graph_a_id, _ = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_A,
        user_id=OWNER_A_ID,
        name="Workspace A draft",
    )
    graph_b_id, _ = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_B,
        user_id=OWNER_B_ID,
        name="Workspace B private",
    )

    actor.as_user(OWNER_A_ID)
    head = client.get(_api(WORKSPACE_A, f"/graphs/{graph_a_id}/head")).json()
    renamed = client.post(
        _api(WORKSPACE_A, f"/graphs/{graph_a_id}/commands"),
        json={
            "command_id": str(uuid4()),
            "room_epoch": head["room_epoch"],
            "observed_sequence": head["collaboration_sequence"],
            "command": {
                "kind": "rename_graph",
                "name": "Current live-head name",
                "expected_name": "Workspace A draft",
            },
        },
    )
    assert renamed.status_code == 200, renamed.text

    created_folder = client.post(
        _api(WORKSPACE_A, "/graph-folders"),
        json={"name": "Research"},
    )
    assert created_folder.status_code == 201, created_folder.text
    folder_id = created_folder.json()["id"]
    assigned = client.put(
        _api(WORKSPACE_A, f"/graphs/{graph_a_id}/folder"),
        json={"folder_id": folder_id},
    )
    assert assigned.status_code == 200, assigned.text
    assert assigned.json()["folder_id"] == folder_id
    assert client.put(_api(WORKSPACE_A, f"/graphs/{graph_a_id}/star")).status_code == 200
    opened = client.post(_api(WORKSPACE_A, f"/graphs/{graph_a_id}/opened"))
    assert opened.status_code == 200, opened.text
    assert opened.json()["last_opened_at"] is not None

    owner_a_browser = client.get("/v1/me/graphs")
    assert owner_a_browser.status_code == 200, owner_a_browser.text
    assert [graph["id"] for graph in owner_a_browser.json()["graphs"]] == [
        str(graph_a_id)
    ]
    owner_a_row = owner_a_browser.json()["graphs"][0]
    assert owner_a_row["location"] == {
        "id": str(WORKSPACE_A),
        "slug": "workspace-a",
        "name": "Workspace A",
        "kind": "shared",
    }
    assert owner_a_row["folder"] == {"id": folder_id, "name": "Research"}
    assert owner_a_row["starred"] is True
    assert owner_a_row["last_opened_at"] is not None
    assert owner_a_row["draft"] == {
        "name": "Current live-head name",
        "head_sequence": 2,
        "checkpoint_sequence": 1,
        "checkpoint_revision": 1,
        "updated_at": renamed.json()["head"]["updated_at"],
        "node_count": 0,
        "edge_count": 0,
    }
    assert owner_a_row["creator"] == {
        "id": str(OWNER_A_ID),
        "display_name": "Owner A",
    }

    actor.as_user(VIEWER_A_ID)
    viewer_browser = client.get("/v1/me/graphs")
    assert viewer_browser.status_code == 200, viewer_browser.text
    viewer_row = viewer_browser.json()["graphs"][0]
    assert viewer_row["id"] == str(graph_a_id)
    assert viewer_row["starred"] is False
    assert viewer_row["last_opened_at"] is None
    _assert_forbidden(
        client.put(_api(WORKSPACE_A, f"/graphs/{graph_a_id}/archive")),
        context="viewer archive graph",
    )
    _assert_forbidden(
        client.post(
            _api(WORKSPACE_A, "/graph-folders"),
            json={"name": "Viewer folder"},
        ),
        context="viewer create folder",
    )
    assert client.put(_api(WORKSPACE_A, f"/graphs/{graph_a_id}/star")).status_code == 200

    actor.as_user(OWNER_A_ID)
    owner_a_row_after_viewer_star = client.get("/v1/me/graphs").json()["graphs"][0]
    assert owner_a_row_after_viewer_star["starred"] is True
    unstarred = client.delete(_api(WORKSPACE_A, f"/graphs/{graph_a_id}/star"))
    assert unstarred.status_code == 200, unstarred.text
    assert unstarred.json()["starred"] is False
    assert client.get("/v1/me/graphs").json()["graphs"][0]["starred"] is False

    archived = client.put(_api(WORKSPACE_A, f"/graphs/{graph_a_id}/archive"))
    assert archived.status_code == 200, archived.text
    assert archived.json()["archived"] is True
    assert client.get("/v1/me/graphs").json()["graphs"][0]["archived"] is True
    restored = client.delete(_api(WORKSPACE_A, f"/graphs/{graph_a_id}/archive"))
    assert restored.status_code == 200, restored.text
    assert restored.json()["archived"] is False

    actor.as_user(BOTH_ID)
    both_browser = client.get("/v1/me/graphs")
    assert both_browser.status_code == 200, both_browser.text
    assert {graph["id"] for graph in both_browser.json()["graphs"]} == {
        str(graph_a_id),
        str(graph_b_id),
    }

    actor.as_user(OWNER_B_ID)
    revoked = client.delete(_api(WORKSPACE_B, f"/members/{BOTH_ID}"))
    assert revoked.status_code == 204, revoked.text
    actor.as_user(BOTH_ID)
    after_revocation = client.get("/v1/me/graphs")
    assert after_revocation.status_code == 200, after_revocation.text
    assert [graph["id"] for graph in after_revocation.json()["graphs"]] == [
        str(graph_a_id)
    ]


def test_folder_assignment_cannot_cross_workspace_and_delete_unfiles_graphs(
    authz_client: tuple[TestClient, ActorSwitcher, UUID],
) -> None:
    client, actor, _ = authz_client
    graph_a_id, _ = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_A,
        user_id=OWNER_A_ID,
        name="Folder boundary",
    )

    actor.as_user(OWNER_B_ID)
    foreign_folder = client.post(
        _api(WORKSPACE_B, "/graph-folders"),
        json={"name": "Foreign"},
    )
    assert foreign_folder.status_code == 201, foreign_folder.text

    actor.as_user(OWNER_A_ID)
    rejected = client.put(
        _api(WORKSPACE_A, f"/graphs/{graph_a_id}/folder"),
        json={"folder_id": foreign_folder.json()["id"]},
    )
    _assert_not_found(rejected, context="cross-workspace folder assignment")

    own_folder = client.post(
        _api(WORKSPACE_A, "/graph-folders"),
        json={"name": "Temporary"},
    )
    assert own_folder.status_code == 201, own_folder.text
    own_folder_id = own_folder.json()["id"]
    renamed = client.patch(
        _api(WORKSPACE_A, f"/graph-folders/{own_folder_id}"),
        json={"name": "  Renamed  "},
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["name"] == "Renamed"
    listed_folders = client.get(_api(WORKSPACE_A, "/graph-folders"))
    assert listed_folders.status_code == 200, listed_folders.text
    assert listed_folders.json()["folders"] == [renamed.json()]
    duplicate = client.post(
        _api(WORKSPACE_A, "/graph-folders"),
        json={"name": "Renamed"},
    )
    assert duplicate.status_code == 409, duplicate.text
    assert (
        client.put(
            _api(WORKSPACE_A, f"/graphs/{graph_a_id}/folder"),
            json={"folder_id": own_folder_id},
        ).status_code
        == 200
    )
    unfiled = client.put(
        _api(WORKSPACE_A, f"/graphs/{graph_a_id}/folder"),
        json={"folder_id": None},
    )
    assert unfiled.status_code == 200, unfiled.text
    assert unfiled.json()["folder_id"] is None
    assert (
        client.put(
            _api(WORKSPACE_A, f"/graphs/{graph_a_id}/folder"),
            json={"folder_id": own_folder_id},
        ).status_code
        == 200
    )

    deleted = client.delete(_api(WORKSPACE_A, f"/graph-folders/{own_folder_id}"))
    assert deleted.status_code == 204, deleted.text
    row = client.get("/v1/me/graphs").json()["graphs"][0]
    assert row["id"] == str(graph_a_id)
    assert row["folder"] is None


def test_non_member_cannot_read_or_write_other_workspace_by_uuid(
    authz_client: tuple[TestClient, ActorSwitcher, UUID],
) -> None:
    client, actor, artifact_id = authz_client
    graph_id, revision = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_B,
        user_id=OWNER_B_ID,
        name="B private graph",
    )
    execution_id = _start_execution(client, actor, OWNER_B_ID, WORKSPACE_B)

    actor.as_user(OWNER_A_ID)
    _assert_not_found(client.get(_api(WORKSPACE_B, "/nodes")), context="nodes")
    _assert_not_found(client.get(_api(WORKSPACE_B, "/graphs")), context="list graphs")
    _assert_not_found(
        client.get(_api(WORKSPACE_B, f"/graphs/{graph_id}")),
        context="get graph",
    )
    _assert_not_found(
        client.post(_api(WORKSPACE_B, "/graphs"), json=_empty_graph_payload()),
        context="create graph",
    )
    _assert_not_found(
        client.put(
            _api(WORKSPACE_B, f"/graphs/{graph_id}"),
            json={**_empty_graph_payload(), "expected_revision": revision},
        ),
        context="update graph",
    )
    _assert_not_found(
        client.delete(
            _api(WORKSPACE_B, f"/graphs/{graph_id}"),
            params={"expected_revision": revision},
        ),
        context="delete graph",
    )
    _assert_not_found(
        client.get(_api(WORKSPACE_B, f"/graphs/{graph_id}/node-secrets")),
        context="list secrets",
    )
    _assert_not_found(
        client.put(
            _api(WORKSPACE_B, f"/graphs/{graph_id}/nodes/llm/secrets/api_key"),
            json={"value": "secret", "expected_graph_revision": revision},
        ),
        context="put secret",
    )
    _assert_not_found(
        client.post(_api(WORKSPACE_B, "/runs"), json={"nodes": [], "edges": []}),
        context="run",
    )
    _assert_not_found(
        client.post(_api(WORKSPACE_B, "/executions"), json={"nodes": [], "edges": []}),
        context="start execution",
    )
    _assert_not_found(
        client.get(_api(WORKSPACE_B, f"/executions/{execution_id}")),
        context="get execution",
    )
    _assert_not_found(
        client.get(_api(WORKSPACE_B, f"/executions/{execution_id}/events")),
        context="sse",
    )
    _assert_not_found(
        client.delete(_api(WORKSPACE_B, f"/executions/{execution_id}")),
        context="cancel execution",
    )
    _assert_not_found(
        client.get(
            _api(WORKSPACE_B, f"/graphs/{graph_id}/materializations"),
            params={"graph_revision": revision},
        ),
        context="materializations",
    )
    _assert_not_found(
        client.get(_api(WORKSPACE_B, f"/graphs/{graph_id}/executions")),
        context="history",
    )
    _assert_not_found(
        client.get(_api(WORKSPACE_B, f"/artifacts/{artifact_id}/content")),
        context="artifact content",
    )
    _assert_not_found(
        client.post(
            _api(WORKSPACE_B, "/uploads"),
            files={"file": ("sample.png", BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
        ),
        context="upload",
    )
    _assert_not_found(
        client.post(_api(WORKSPACE_B, "/samples"), json={"count": 1}),
        context="samples",
    )
    _assert_not_found(client.get(_api(WORKSPACE_B, "/members")), context="list members")
    _assert_not_found(
        client.post(
            _api(WORKSPACE_B, "/members"),
            json={"user_id": str(VIEWER_A_ID), "role": "viewer"},
        ),
        context="add member",
    )


def test_viewer_can_read_but_cannot_mutate_execute_or_manage_secrets(
    authz_client: tuple[TestClient, ActorSwitcher, UUID],
) -> None:
    client, actor, artifact_id = authz_client
    graph_id, revision = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_A,
        user_id=OWNER_A_ID,
        name="Shared readable graph",
    )
    execution_id = _start_execution(client, actor, EDITOR_A_ID, WORKSPACE_A)

    actor.as_user(VIEWER_A_ID)
    assert client.get(_api(WORKSPACE_A, "/nodes")).status_code == 200
    assert client.get(_api(WORKSPACE_A, "/graphs")).status_code == 200
    assert client.get(_api(WORKSPACE_A, f"/graphs/{graph_id}")).status_code == 200
    assert (
        client.get(_api(WORKSPACE_A, f"/graphs/{graph_id}/node-secrets")).status_code
        == 200
    )
    assert (
        client.get(_api(WORKSPACE_A, f"/artifacts/{artifact_id}/content")).status_code
        == 200
    )
    assert (
        client.get(_api(WORKSPACE_A, f"/executions/{execution_id}")).status_code == 200
    )
    assert (
        client.get(
            _api(WORKSPACE_A, f"/graphs/{graph_id}/materializations"),
            params={"graph_revision": revision},
        ).status_code
        == 200
    )
    assert (
        client.get(_api(WORKSPACE_A, f"/graphs/{graph_id}/executions")).status_code
        == 200
    )

    _assert_forbidden(
        client.post(_api(WORKSPACE_A, "/graphs"), json=_empty_graph_payload("viewer")),
        context="viewer create graph",
    )
    _assert_forbidden(
        client.put(
            _api(WORKSPACE_A, f"/graphs/{graph_id}"),
            json={
                **_empty_graph_payload("viewer edit"),
                "expected_revision": revision,
            },
        ),
        context="viewer update graph",
    )
    _assert_forbidden(
        client.delete(
            _api(WORKSPACE_A, f"/graphs/{graph_id}"),
            params={"expected_revision": revision},
        ),
        context="viewer delete graph",
    )
    _assert_forbidden(
        client.put(
            _api(WORKSPACE_A, f"/graphs/{graph_id}/nodes/llm/secrets/api_key"),
            json={"value": "secret", "expected_graph_revision": revision},
        ),
        context="viewer put secret",
    )
    _assert_forbidden(
        client.delete(
            _api(WORKSPACE_A, f"/graphs/{graph_id}/nodes/llm/secrets/api_key"),
            params={"expected_graph_revision": revision},
        ),
        context="viewer delete secret",
    )
    _assert_forbidden(
        client.post(_api(WORKSPACE_A, "/runs"), json={"nodes": [], "edges": []}),
        context="viewer run",
    )
    _assert_forbidden(
        client.post(_api(WORKSPACE_A, "/executions"), json={"nodes": [], "edges": []}),
        context="viewer start execution",
    )
    _assert_forbidden(
        client.delete(_api(WORKSPACE_A, f"/executions/{execution_id}")),
        context="viewer cancel",
    )
    _assert_forbidden(
        client.post(
            _api(WORKSPACE_A, "/uploads"),
            files={"file": ("sample.png", BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
        ),
        context="viewer upload",
    )
    _assert_forbidden(
        client.post(_api(WORKSPACE_A, "/samples"), json={"count": 1}),
        context="viewer samples",
    )
    _assert_forbidden(
        client.post(
            _api(WORKSPACE_A, "/members"),
            json={"user_id": str(OWNER_B_ID), "role": "viewer"},
        ),
        context="viewer add member",
    )


def test_editor_can_edit_and_execute_but_not_manage_secrets_delete_or_members(
    authz_client: tuple[TestClient, ActorSwitcher, UUID],
) -> None:
    client, actor, _artifact_id = authz_client
    graph_id, revision = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_A,
        user_id=EDITOR_A_ID,
        name="Editor draft",
    )

    actor.as_user(EDITOR_A_ID)
    update = client.put(
        _api(WORKSPACE_A, f"/graphs/{graph_id}"),
        json={**_empty_graph_payload("Editor updated"), "expected_revision": revision},
    )
    assert update.status_code == 200
    revision = int(update.json()["revision"])

    run = client.post(_api(WORKSPACE_A, "/runs"), json={"nodes": [], "edges": []})
    assert run.status_code == 200
    execution = client.post(
        _api(WORKSPACE_A, "/executions"),
        json={"nodes": [], "edges": []},
    )
    assert execution.status_code == 202
    execution_id = UUID(execution.json()["execution_id"])
    assert client.get(_api(WORKSPACE_A, f"/executions/{execution_id}")).status_code == 200
    with client.stream(
        "GET", _api(WORKSPACE_A, f"/executions/{execution_id}/events")
    ) as events:
        assert events.status_code == 200
        assert "text/event-stream" in events.headers["content-type"]
        # Drain a bounded prefix so the stream can finish without hanging the suite.
        body = events.read()
        assert body  # at least one lifecycle frame for the empty graph run

    upload = client.post(
        _api(WORKSPACE_A, "/uploads"),
        files={"file": ("sample.png", BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
    )
    assert upload.status_code == 200

    _assert_forbidden(
        client.put(
            _api(WORKSPACE_A, f"/graphs/{graph_id}/nodes/llm/secrets/api_key"),
            json={"value": "secret", "expected_graph_revision": revision},
        ),
        context="editor put secret",
    )
    _assert_forbidden(
        client.delete(
            _api(WORKSPACE_A, f"/graphs/{graph_id}/nodes/llm/secrets/api_key"),
            params={"expected_graph_revision": revision},
        ),
        context="editor delete secret",
    )
    _assert_forbidden(
        client.delete(
            _api(WORKSPACE_A, f"/graphs/{graph_id}"),
            params={"expected_revision": revision},
        ),
        context="editor delete graph",
    )
    _assert_forbidden(
        client.get(_api(WORKSPACE_A, "/members")),
        context="editor list members",
    )
    _assert_forbidden(
        client.post(
            _api(WORKSPACE_A, "/members"),
            json={"user_id": str(OWNER_B_ID), "role": "viewer"},
        ),
        context="editor add member",
    )


def test_owner_can_manage_secrets_delete_graph_and_members(
    authz_client: tuple[TestClient, ActorSwitcher, UUID],
) -> None:
    client, actor, _artifact_id = authz_client
    graph_id, revision = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_A,
        user_id=OWNER_A_ID,
        name="Owner managed graph",
    )

    actor.as_user(OWNER_A_ID)
    # Capability gate is before resource lookup: owner clears MANAGE_SECRETS
    # (404 for undeclared secret) while editor/viewer receive 403 above.
    secret_put = client.put(
        _api(WORKSPACE_A, f"/graphs/{graph_id}/nodes/missing/secrets/api_key"),
        json={"value": "secret", "expected_graph_revision": revision},
    )
    assert secret_put.status_code == 404

    members = client.get(_api(WORKSPACE_A, "/members"))
    assert members.status_code == 200
    member_ids = {UUID(item["user"]["id"]) for item in members.json()}
    assert OWNER_A_ID in member_ids
    assert VIEWER_A_ID in member_ids
    assert EDITOR_A_ID in member_ids

    deleted = client.delete(
        _api(WORKSPACE_A, f"/graphs/{graph_id}"),
        params={"expected_revision": revision},
    )
    assert deleted.status_code == 204
    assert client.get(_api(WORKSPACE_A, f"/graphs/{graph_id}")).status_code == 404


def test_cross_workspace_resource_ids_do_not_authorize_via_wrong_path(
    authz_client: tuple[TestClient, ActorSwitcher, UUID],
) -> None:
    client, actor, artifact_id = authz_client
    graph_a, revision_a = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_A,
        user_id=OWNER_A_ID,
        name="Graph in A",
    )
    graph_b, revision_b = _create_graph(
        client,
        actor,
        workspace_id=WORKSPACE_B,
        user_id=OWNER_B_ID,
        name="Graph in B",
    )
    execution_a = _start_execution(client, actor, OWNER_A_ID, WORKSPACE_A)

    # Member of both workspaces still cannot reach A resources under B's path.
    actor.as_user(BOTH_ID)
    assert client.get(_api(WORKSPACE_A, f"/graphs/{graph_a}")).status_code == 200
    assert client.get(_api(WORKSPACE_B, f"/graphs/{graph_b}")).status_code == 200

    wrong_path_reads = (
        _api(WORKSPACE_B, f"/graphs/{graph_a}"),
        _api(WORKSPACE_A, f"/graphs/{graph_b}"),
        _api(WORKSPACE_B, f"/graphs/{graph_a}/node-secrets"),
        _api(WORKSPACE_B, f"/artifacts/{artifact_id}/content"),
        _api(WORKSPACE_B, f"/executions/{execution_a}"),
        _api(WORKSPACE_B, f"/executions/{execution_a}/events"),
        _api(WORKSPACE_B, f"/graphs/{graph_a}/executions"),
        _api(WORKSPACE_A, f"/graphs/{graph_b}/executions"),
    )
    for path in wrong_path_reads:
        response = client.get(path)
        assert response.status_code == 404, (path, response.status_code, response.text)

    assert (
        client.get(
            _api(WORKSPACE_B, f"/graphs/{graph_a}/materializations"),
            params={"graph_revision": revision_a},
        ).status_code
        == 404
    )
    assert (
        client.get(
            _api(WORKSPACE_A, f"/graphs/{graph_b}/materializations"),
            params={"graph_revision": revision_b},
        ).status_code
        == 404
    )

    wrong_secret = client.put(
        _api(WORKSPACE_B, f"/graphs/{graph_a}/nodes/llm/secrets/api_key"),
        json={"value": "secret", "expected_graph_revision": revision_a},
    )
    assert wrong_secret.status_code == 404

    wrong_delete = client.delete(
        _api(WORKSPACE_B, f"/graphs/{graph_a}"),
        params={"expected_revision": revision_a},
    )
    assert wrong_delete.status_code == 404

    # Resource remains readable under its owning workspace path.
    assert client.get(_api(WORKSPACE_A, f"/graphs/{graph_a}")).status_code == 200
    assert (
        client.get(_api(WORKSPACE_A, f"/artifacts/{artifact_id}/content")).status_code
        == 200
    )
