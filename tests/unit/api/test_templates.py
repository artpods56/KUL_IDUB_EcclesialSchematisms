import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from grafy_api.main import create_app
from grafy_api.settings import Settings
from grafy_api.v1.routes.auth.dependencies import browser_actor
from grafy_core.artifacts import ArtifactObject
from grafy_core.domain.collaboration import CollaborativeGraphHead
from grafy_core.domain.execution_history import GraphExecution
from grafy_core.domain.identity import (
    ActorContext,
    User,
    Workspace,
    WorkspaceKind,
    WorkspaceMembership,
    WorkspaceRole,
)
from grafy_core.domain.node_secrets import EncryptedNodeSecret
from grafy_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraph,
    SavedGraphDocument,
    SavedGraphNode,
    SavedGraphRevision,
)
from grafy_persistence.database import create_database
from grafy_persistence.orm import metadata
from grafy_persistence.unit_of_work import SqlAlchemyUnitOfWork


SOURCE_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000701")
DESTINATION_WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000702")
OWNER_ID = UUID("00000000-0000-0000-0000-000000000703")
EDITOR_ID = UUID("00000000-0000-0000-0000-000000000704")
VIEWER_ID = UUID("00000000-0000-0000-0000-000000000705")
CROSS_LOCATION_ID = UUID("00000000-0000-0000-0000-000000000706")
VIEWER_BOTH_ID = UUID("00000000-0000-0000-0000-000000000707")
OUTSIDER_ID = UUID("00000000-0000-0000-0000-000000000708")
SOURCE_GRAPH_ID = UUID("00000000-0000-0000-0000-000000000709")
SOURCE_ARTIFACT_ID = UUID("00000000-0000-0000-0000-000000000710")
CAPABILITY_GRAPH_ID = UUID("00000000-0000-0000-0000-000000000712")


def _api(workspace_id: UUID, suffix: str) -> str:
    normalized = suffix if suffix.startswith("/") else f"/{suffix}"
    return f"/v1/workspaces/{workspace_id}{normalized}"


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


def _source_revision_document() -> SavedGraphDocument:
    return SavedGraphDocument(
        nodes=(
            SavedGraphNode(
                id="source",
                operator_id="example.operator",
                operator_version=1,
                config={
                    "safe_setting": "preserved",
                    "artifact_id": str(SOURCE_ARTIFACT_ID),
                    "upload_key": "source-upload",
                    "uploads": ["source-upload"],
                    "nested": {
                        "artifact_id": str(SOURCE_ARTIFACT_ID),
                        "safe": True,
                    },
                },
                position=GraphPoint(x=1, y=2),
            ),
        )
    )


async def _seed_templates(database_url: str) -> None:
    database = create_database(database_url)
    now = datetime.now(UTC)
    revision_one_document = _source_revision_document()
    revision_two_document = SavedGraphDocument(
        nodes=(
            SavedGraphNode(
                id="changed-later",
                operator_id="example.operator",
                operator_version=1,
                config={"safe_setting": "new source value"},
                position=GraphPoint(x=5, y=6),
            ),
        )
    )
    source_graph = SavedGraph(
        id=SOURCE_GRAPH_ID,
        workspace_id=SOURCE_WORKSPACE_ID,
        created_by_user_id=OWNER_ID,
        name="Changed source",
        document=revision_two_document,
        revision=2,
        created_at=now,
        updated_at=now,
    )
    capability_graph = SavedGraph(
        id=CAPABILITY_GRAPH_ID,
        workspace_id=SOURCE_WORKSPACE_ID,
        created_by_user_id=OWNER_ID,
        name="Graph with a workspace-bound module call",
        document=SavedGraphDocument(
            nodes=(
                SavedGraphNode(
                    id="module-call",
                    operator_id=f"graph.module.{DESTINATION_WORKSPACE_ID}",
                    operator_version=1,
                    config={},
                    position=GraphPoint(x=0, y=0),
                ),
            )
        ),
        created_at=now,
        updated_at=now,
    )
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            for user_id in (
                OWNER_ID,
                EDITOR_ID,
                VIEWER_ID,
                CROSS_LOCATION_ID,
                VIEWER_BOTH_ID,
                OUTSIDER_ID,
            ):
                await unit_of_work.identity.add_user(
                    User(
                        id=user_id,
                        email=f"{user_id.hex}@example.test",
                        display_name=user_id.hex[-4:],
                    )
                )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=SOURCE_WORKSPACE_ID,
                    slug="my-graphs",
                    name="My graphs",
                    kind=WorkspaceKind.SHARED,
                )
            )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=DESTINATION_WORKSPACE_ID,
                    slug="research-team",
                    name="Research team",
                    kind=WorkspaceKind.SHARED,
                )
            )
            for workspace_id, user_id, role in (
                (SOURCE_WORKSPACE_ID, OWNER_ID, WorkspaceRole.OWNER),
                (DESTINATION_WORKSPACE_ID, OWNER_ID, WorkspaceRole.OWNER),
                (SOURCE_WORKSPACE_ID, EDITOR_ID, WorkspaceRole.EDITOR),
                (SOURCE_WORKSPACE_ID, VIEWER_ID, WorkspaceRole.VIEWER),
                (SOURCE_WORKSPACE_ID, CROSS_LOCATION_ID, WorkspaceRole.VIEWER),
                (
                    DESTINATION_WORKSPACE_ID,
                    CROSS_LOCATION_ID,
                    WorkspaceRole.EDITOR,
                ),
                (SOURCE_WORKSPACE_ID, VIEWER_BOTH_ID, WorkspaceRole.VIEWER),
                (DESTINATION_WORKSPACE_ID, VIEWER_BOTH_ID, WorkspaceRole.VIEWER),
            ):
                await unit_of_work.identity.add_membership(
                    WorkspaceMembership(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        role=role,
                    )
                )
            await unit_of_work.graphs.add(source_graph)
            await unit_of_work.graphs.add(capability_graph)
            await unit_of_work.graphs.add_revision(
                SavedGraphRevision(
                    workspace_id=SOURCE_WORKSPACE_ID,
                    graph_id=SOURCE_GRAPH_ID,
                    revision=1,
                    name="Original source",
                    document=revision_one_document,
                    created_at=now,
                )
            )
            await unit_of_work.graphs.add_revision(source_graph.snapshot())
            await unit_of_work.graphs.add_revision(capability_graph.snapshot())
            await unit_of_work.collaboration.add_head(
                CollaborativeGraphHead.for_existing_saved_graph(
                    workspace_id=SOURCE_WORKSPACE_ID,
                    graph_id=SOURCE_GRAPH_ID,
                    name=source_graph.name,
                    document=source_graph.document,
                    checkpoint_revision=source_graph.revision,
                )
            )
            await unit_of_work.collaboration.add_head(
                CollaborativeGraphHead.for_existing_saved_graph(
                    workspace_id=SOURCE_WORKSPACE_ID,
                    graph_id=CAPABILITY_GRAPH_ID,
                    name=capability_graph.name,
                    document=capability_graph.document,
                    checkpoint_revision=capability_graph.revision,
                )
            )
            await unit_of_work.artifacts.add(
                ArtifactObject(
                    id=SOURCE_ARTIFACT_ID,
                    workspace_id=SOURCE_WORKSPACE_ID,
                    artifact_type="scalar.text",
                    schema_version=1,
                    content_type="application/json",
                    storage_backend="inline",
                    inline_payload={"value": "source-only"},
                )
            )
            await unit_of_work.node_secrets.upsert(
                EncryptedNodeSecret(
                    workspace_id=SOURCE_WORKSPACE_ID,
                    graph_id=SOURCE_GRAPH_ID,
                    node_id="source",
                    name="api_key",
                    operator_id="example.operator",
                    operator_version=1,
                    key_id="test-key",
                    aad_version=2,
                    dependency_sha256="d" * 64,
                    nonce=b"n" * 12,
                    ciphertext=b"source-secret",
                    created_at=now,
                    updated_at=now,
                )
            )
            await unit_of_work.execution_history.add(
                GraphExecution(
                    workspace_id=SOURCE_WORKSPACE_ID,
                    execution_id=UUID("00000000-0000-0000-0000-000000000711"),
                    graph_id=SOURCE_GRAPH_ID,
                    graph_revision=1,
                    requested_node_ids=("source",),
                    status="succeeded",
                    created_at=now,
                    started_at=now,
                    finished_at=now,
                )
            )
            await unit_of_work.commit()
    finally:
        await database.dispose()


@pytest.fixture
def template_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, ActorSwitcher]]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'templates.sqlite3'}"
    asyncio.run(_seed_templates(database_url))
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
            auth_cookie_secure=False,
        )
    )
    actor = ActorSwitcher(user_id=OWNER_ID)
    actor.install(application)
    with TestClient(application) as client:
        yield client, actor


def _create_template(
    client: TestClient, *, name: str = "Starter analysis"
) -> dict[str, object]:
    response = client.post(
        _api(SOURCE_WORKSPACE_ID, "/templates"),
        json={
            "source_graph_id": str(SOURCE_GRAPH_ID),
            "source_revision": 1,
            "name": name,
            "description": "A reusable research starting point",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_template_snapshot_and_instantiations_remain_independent_and_safe(
    template_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, actor = template_client
    actor.as_user(OWNER_ID)
    template = _create_template(client)
    folder_response = client.post(
        _api(DESTINATION_WORKSPACE_ID, "/graph-folders"),
        json={"name": "Fieldwork"},
    )
    assert folder_response.status_code == 201, folder_response.text
    folder_id = folder_response.json()["id"]

    first = client.post(
        _api(SOURCE_WORKSPACE_ID, f"/templates/{template['id']}/instantiate"),
        json={
            "destination_workspace_id": str(DESTINATION_WORKSPACE_ID),
            "name": "Climate review",
            "folder_id": folder_id,
        },
    )
    second = client.post(
        _api(SOURCE_WORKSPACE_ID, f"/templates/{template['id']}/instantiate"),
        json={
            "destination_workspace_id": str(DESTINATION_WORKSPACE_ID),
            "name": "Second review",
            "folder_id": folder_id,
        },
    )
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    first_copy = first.json()
    second_copy = second.json()
    assert first_copy["graph_id"] != second_copy["graph_id"]
    assert first_copy["folder_id"] == folder_id

    graph = client.get(
        _api(DESTINATION_WORKSPACE_ID, f"/graphs/{first_copy['graph_id']}")
    )
    assert graph.status_code == 200, graph.text
    body = graph.json()
    assert body["name"] == "Climate review"
    assert [node["id"] for node in body["nodes"]] == ["source"]
    config = body["nodes"][0]["config"]
    assert config == {
        "safe_setting": "preserved",
        "nested": {"safe": True},
    }
    assert (
        client.get(
            _api(
                DESTINATION_WORKSPACE_ID,
                f"/graphs/{first_copy['graph_id']}/node-secrets",
            )
        ).json()["secrets"]
        == []
    )
    assert (
        client.get(
            _api(
                DESTINATION_WORKSPACE_ID,
                f"/graphs/{first_copy['graph_id']}/executions",
            )
        ).json()["items"]
        == []
    )
    assert (
        client.get(
            _api(DESTINATION_WORKSPACE_ID, f"/artifacts/{SOURCE_ARTIFACT_ID}")
        ).status_code
        == 404
    )

    mutated_nodes = body["nodes"]
    mutated_nodes[0]["config"]["safe_setting"] = "first copy only"
    updated_first = client.put(
        _api(DESTINATION_WORKSPACE_ID, f"/graphs/{first_copy['graph_id']}"),
        json={
            "expected_revision": 1,
            "name": body["name"],
            "nodes": mutated_nodes,
            "edges": body["edges"],
            "presentation": body["presentation"],
        },
    )
    assert updated_first.status_code == 200, updated_first.text
    unchanged_second = client.get(
        _api(DESTINATION_WORKSPACE_ID, f"/graphs/{second_copy['graph_id']}")
    )
    assert unchanged_second.status_code == 200
    assert unchanged_second.json()["nodes"][0]["config"]["safe_setting"] == (
        "preserved"
    )

    source = client.get(_api(SOURCE_WORKSPACE_ID, f"/graphs/{SOURCE_GRAPH_ID}"))
    assert source.status_code == 200
    assert [node["id"] for node in source.json()["nodes"]] == ["changed-later"]

    invalid_capability = client.post(
        _api(SOURCE_WORKSPACE_ID, "/templates"),
        json={
            "source_graph_id": str(CAPABILITY_GRAPH_ID),
            "source_revision": 1,
            "name": "Unsafe capability",
        },
    )
    assert invalid_capability.status_code == 422
    assert "cannot include module operator" in invalid_capability.text


def test_template_search_metadata_archive_and_role_authorization(
    template_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, actor = template_client
    actor.as_user(EDITOR_ID)
    template = _create_template(client, name="Survey starter")
    template_id = template["id"]

    search = client.get(
        _api(SOURCE_WORKSPACE_ID, "/templates"),
        params={"q": "research"},
    )
    assert search.status_code == 200
    assert [item["id"] for item in search.json()["templates"]] == [template_id]

    updated = client.put(
        _api(SOURCE_WORKSPACE_ID, f"/templates/{template_id}"),
        json={"name": "Survey field kit", "description": None},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Survey field kit"
    assert updated.json()["description"] is None
    assert (
        client.post(
            _api(SOURCE_WORKSPACE_ID, f"/templates/{template_id}/archive")
        ).status_code
        == 403
    )

    actor.as_user(OWNER_ID)
    archived = client.post(
        _api(SOURCE_WORKSPACE_ID, f"/templates/{template_id}/archive")
    )
    assert archived.status_code == 200
    assert archived.json()["state"] == "archived"
    assert client.get(_api(SOURCE_WORKSPACE_ID, "/templates")).json()["templates"] == []
    use_archived = client.post(
        _api(SOURCE_WORKSPACE_ID, f"/templates/{template_id}/instantiate"),
        json={
            "destination_workspace_id": str(DESTINATION_WORKSPACE_ID),
            "name": "Should fail",
        },
    )
    assert use_archived.status_code == 422
    assert "Archived templates cannot be used" in use_archived.text


def test_using_template_requires_source_read_and_destination_create(
    template_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, actor = template_client
    actor.as_user(OWNER_ID)
    template = _create_template(client)
    template_id = template["id"]

    actor.as_user(OUTSIDER_ID)
    assert client.get(_api(SOURCE_WORKSPACE_ID, "/templates")).status_code == 404

    actor.as_user(VIEWER_ID)
    assert (
        client.get(_api(SOURCE_WORKSPACE_ID, f"/templates/{template_id}")).status_code
        == 200
    )
    assert (
        client.post(
            _api(SOURCE_WORKSPACE_ID, "/templates"),
            json={
                "source_graph_id": str(SOURCE_GRAPH_ID),
                "source_revision": 1,
                "name": "Denied",
            },
        ).status_code
        == 403
    )

    actor.as_user(VIEWER_BOTH_ID)
    denied = client.post(
        _api(SOURCE_WORKSPACE_ID, f"/templates/{template_id}/instantiate"),
        json={
            "destination_workspace_id": str(DESTINATION_WORKSPACE_ID),
            "name": "No destination create",
        },
    )
    assert denied.status_code == 403

    actor.as_user(CROSS_LOCATION_ID)
    allowed = client.post(
        _api(SOURCE_WORKSPACE_ID, f"/templates/{template_id}/instantiate"),
        json={
            "destination_workspace_id": str(DESTINATION_WORKSPACE_ID),
            "name": "Cross-location copy",
        },
    )
    assert allowed.status_code == 201, allowed.text
    assert allowed.json()["destination_workspace_id"] == str(DESTINATION_WORKSPACE_ID)


def test_template_destination_folder_must_exist_in_destination_workspace(
    template_client: tuple[TestClient, ActorSwitcher],
) -> None:
    client, actor = template_client
    actor.as_user(OWNER_ID)
    template = _create_template(client)
    response = client.post(
        _api(SOURCE_WORKSPACE_ID, f"/templates/{template['id']}/instantiate"),
        json={
            "destination_workspace_id": str(DESTINATION_WORKSPACE_ID),
            "name": "Missing folder rejected",
            "folder_id": "00000000-0000-0000-0000-000000000999",
        },
    )
    assert response.status_code == 404
    assert "Graph folder" in response.text
