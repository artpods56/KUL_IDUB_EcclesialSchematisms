import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from grafy_core.application.saved_graphs import SavedGraphService
from grafy_core.domain.execution_history import GraphExecution
from grafy_core.domain.identity import (
    User,
    Workspace,
    WorkspaceMembership,
    WorkspaceRole,
)
from grafy_core.domain.saved_graphs import SavedGraphDocument
from grafy_persistence.unit_of_work import (
    SqlAlchemySavedGraphUnitOfWork,
    SqlAlchemyUnitOfWork,
)

from grafy_api.builtins import builtin_plugins
from tests.support.identity import browser_actor_override
from grafy_api.v1.routes.auth.dependencies import browser_actor
from grafy_api.plugin_discovery import build_plugin_registry
from grafy_api.v1.routes.executions.models import (
    RunExecutionResponse,
    RunNodeRequest,
    RunRequest,
)
from tests.support.clients import GrafyApi
from grafy_api.v1.routes.saved_graphs.models import (
    CreateSavedGraphRequest,
    GraphPointModel,
    SavedGraphNodeModel,
    SavedGraphResponse,
    UpdateSavedGraphRequest,
)
from grafy_api.settings import Settings

from tests.testkit import client_with_overrides, create_db_url, db


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000007")


def _saved_text_graph_nodes(text: str) -> list[SavedGraphNodeModel]:
    return [
        SavedGraphNodeModel(
            id="text",
            operator_id="text.input",
            operator_version=1,
            config={"text": text},
            position=GraphPointModel(x=0, y=0),
        )
    ]


def _create_text_graph_request(name: str, text: str) -> CreateSavedGraphRequest:
    return CreateSavedGraphRequest(
        name=name, nodes=_saved_text_graph_nodes(text), edges=[]
    )


def _update_text_graph_request(
    name: str, text: str, *, expected_revision: int
) -> UpdateSavedGraphRequest:
    return UpdateSavedGraphRequest(
        name=name,
        expected_revision=expected_revision,
        nodes=_saved_text_graph_nodes(text),
        edges=[],
    )


def _start_saved_text_execution(
    client: TestClient,
    graph: SavedGraphResponse,
) -> RunExecutionResponse:
    api = GrafyApi(client)
    executions = api.workspace(WORKSPACE_ID).executions
    started = executions.start_execution_ok(
        RunRequest(
            nodes=[
                RunNodeRequest(
                    id="text",
                    operator_id="text.input",
                    operator_version=1,
                    config={"text": "remember this"},
                )
            ],
            edges=[],
            scope="selected-with-dependencies",
            graph_id=graph.id,
            graph_revision=graph.revision,
        )
    )
    for _ in range(100):
        current = executions.get_execution_ok(started.execution_id)
        if current.status in {"cancelled", "succeeded", "failed"}:
            return current
    raise AssertionError("Execution did not reach a terminal status")


def test_saved_graph_execution_history_lists_filters_and_renders_artifacts(
    builtin_client: TestClient,
) -> None:
    api = GrafyApi(builtin_client)
    executions = api.workspace(WORKSPACE_ID).executions
    graphs = api.workspace(WORKSPACE_ID).graphs
    created_response = graphs.create(
        _create_text_graph_request("History", "remember this")
    )
    assert created_response.status_code == 201
    graph = SavedGraphResponse.model_validate(created_response.json())

    first = _start_saved_text_execution(builtin_client, graph)
    assert first.status == "succeeded"

    listing = executions.list_graph_executions_ok(graph.id)
    assert len(listing.items) == 1
    summary = listing.items[0]
    assert summary.execution_id == first.execution_id
    assert summary.graph_revision == graph.revision
    assert summary.scope == "selected-with-dependencies"
    assert summary.status == "succeeded"
    assert summary.requested_node_ids == ["text"]
    assert summary.node_count == 1
    assert summary.artifact_count == 1
    assert summary.started_at is not None
    assert summary.finished_at is not None
    assert summary.workflow_run_id is not None

    detail = executions.get_graph_execution_ok(graph.id, first.execution_id)
    assert [(result.node_id, result.status) for result in detail.node_results] == [
        ("text", "succeeded")
    ]
    output = detail.node_results[0].outputs[0]
    assert output.port == "text"
    assert output.artifacts[0].text == '"remember this"'

    succeeded = executions.list_graph_executions_ok(
        graph.id, status="succeeded", node_id="text"
    )
    assert len(succeeded.items) == 1
    no_match = executions.list_graph_executions_ok(
        graph.id, status="failed", node_id="missing"
    )
    assert no_match.items == []
    # Values the route rejects (whitespace node filter, undecodable cursor)
    # are deliberate boundary probes; keep them on the raw client.
    assert (
        builtin_client.get(
            f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph.id}/executions",
            params={"node_id": "   "},
        ).status_code
        == 422
    )
    assert (
        builtin_client.get(
            f"/v1/workspaces/00000000-0000-0000-0000-000000000007/graphs/{graph.id}/executions",
            params={"cursor": "not-a-cursor"},
        ).status_code
        == 422
    )
    assert (
        executions.get_graph_execution(uuid4(), first.execution_id).status_code == 404
    )

    updated_response = api.workspace(WORKSPACE_ID).graphs.update(
        graph.id,
        _update_text_graph_request(
            "History r2", "remember this", expected_revision=graph.revision
        ),
    )
    assert updated_response.status_code == 200
    updated_graph = SavedGraphResponse.model_validate(updated_response.json())
    assert updated_graph.revision == 2

    second = _start_saved_text_execution(builtin_client, updated_graph)
    assert second.status == "succeeded"
    revision_one = executions.list_graph_executions_ok(graph.id, graph_revision=1)
    assert [item.execution_id for item in revision_one.items] == [first.execution_id]
    revision_two = executions.list_graph_executions_ok(graph.id, graph_revision=2)
    assert [item.execution_id for item in revision_two.items] == [second.execution_id]
    all_revisions = executions.list_graph_executions_ok(graph.id)
    assert {item.graph_revision for item in all_revisions.items} == {1, 2}
    first_page = executions.list_graph_executions_ok(graph.id, limit=1)
    assert len(first_page.items) == 1
    assert first_page.next_cursor is not None
    second_page = executions.list_graph_executions_ok(
        graph.id, limit=1, cursor=first_page.next_cursor
    )
    assert len(second_page.items) == 1
    assert second_page.items[0].execution_id != first_page.items[0].execution_id


def test_saved_graph_execution_is_not_accepted_without_its_revision(
    builtin_client: TestClient,
) -> None:
    api = GrafyApi(builtin_client)
    executions = api.workspace(WORKSPACE_ID).executions
    response = executions.start_execution(
        RunRequest(
            nodes=[],
            edges=[],
            graph_id=uuid4(),
            graph_revision=1,
        )
    )

    assert response.status_code == 404
    assert "Saved graph revision" in response.json()["detail"]


def test_duplicate_saved_node_ids_become_a_browsable_failed_execution(
    builtin_client: TestClient,
) -> None:
    api = GrafyApi(builtin_client)
    executions = api.workspace(WORKSPACE_ID).executions
    graphs = api.workspace(WORKSPACE_ID).graphs
    created_response = graphs.create(
        _create_text_graph_request("Invalid history", "duplicate")
    )
    assert created_response.status_code == 201
    graph = SavedGraphResponse.model_validate(created_response.json())
    duplicate_node = RunNodeRequest(
        id="text",
        operator_id="text.input",
        operator_version=1,
        config={"text": "duplicate"},
    )

    execution = executions.start_execution_ok(
        RunRequest(
            nodes=[duplicate_node, duplicate_node],
            edges=[],
            scope="all",
            graph_id=graph.id,
            graph_revision=graph.revision,
        )
    )
    for _ in range(100):
        execution = executions.get_execution_ok(execution.execution_id)
        if execution.status == "failed":
            break
    assert execution.status == "failed"
    assert execution.error is not None
    assert "Duplicate node ids" in execution.error

    detail = executions.get_graph_execution_ok(graph.id, execution.execution_id)
    assert detail.status == "failed"
    assert detail.requested_node_ids == ["text"]
    assert detail.node_results == []
    assert detail.error is not None
    assert "Duplicate node ids" in detail.error


async def _seed_active_execution(database_url: str) -> tuple[UUID, UUID]:
    from grafy_core.domain.collaboration import (
        CollaborativeGraphHead,
        GraphActiveExecutionSlot,
    )

    async with db(database_url) as database:
        registry = build_plugin_registry(builtin_plugins(), external_plugins=())
        saved_graphs = SavedGraphService(
            lambda: SqlAlchemySavedGraphUnitOfWork(database.sessions),
            registry,
        )
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            await unit_of_work.identity.add_user(
                User(
                    id=UUID(int=1),
                    email="owner@example.test",
                    display_name="Owner",
                )
            )
            await unit_of_work.identity.add_workspace(
                Workspace(
                    id=WORKSPACE_ID,
                    slug="local",
                    name="Local workspace",
                    kind="shared",
                )
            )
            await unit_of_work.identity.add_membership(
                WorkspaceMembership(
                    workspace_id=WORKSPACE_ID,
                    user_id=UUID(int=1),
                    role=WorkspaceRole.OWNER,
                )
            )
            await unit_of_work.commit()
        graph = await saved_graphs.create(
            workspace_id=WORKSPACE_ID,
            created_by_user_id=None,
            name="Interrupted",
            document=SavedGraphDocument(),
        )
        execution_id = uuid4()
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            await unit_of_work.collaboration.add_head(
                CollaborativeGraphHead(
                    workspace_id=WORKSPACE_ID,
                    graph_id=graph.id,
                    room_epoch=uuid4(),
                    collaboration_sequence=1,
                    checkpoint_sequence=1,
                    checkpoint_revision=graph.revision,
                    name=graph.name,
                    document=graph.document,
                )
            )
            await unit_of_work.execution_history.add(
                GraphExecution(
                    workspace_id=WORKSPACE_ID,
                    execution_id=execution_id,
                    graph_id=graph.id,
                    graph_revision=graph.revision,
                    status="running",
                    requested_node_ids=(),
                )
            )
            await unit_of_work.commit()
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            acquired = await unit_of_work.collaboration.acquire_active_execution_slot(
                GraphActiveExecutionSlot(
                    workspace_id=WORKSPACE_ID,
                    graph_id=graph.id,
                    execution_id=execution_id,
                )
            )
            assert acquired
            await unit_of_work.commit()
        return graph.id, execution_id


def test_application_startup_marks_stale_active_execution_failed(
    tmp_path: Path,
) -> None:
    database_url = create_db_url(tmp_path, "recovery.sqlite3")
    graph_id, execution_id = asyncio.run(_seed_active_execution(database_url))
    with client_with_overrides(
        settings=Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        ),
        overrides={browser_actor: browser_actor_override},
    ) as client:
        api = GrafyApi(client)
        executions = api.workspace(WORKSPACE_ID).executions
        detail = executions.get_graph_execution_ok(graph_id, execution_id)

    assert detail.status == "failed"
    assert detail.finished_at is not None
    assert detail.error is not None
    assert "API process stopped" in detail.error
