import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from notarius_core.application.saved_graphs import SavedGraphService
from notarius_core.domain.execution_history import GraphExecution
from notarius_core.domain.saved_graphs import SavedGraphDocument
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import (
    SqlAlchemySavedGraphUnitOfWork,
    SqlAlchemyUnitOfWork,
)

from notarius_api.builtins import builtin_plugins
from notarius_api.main import create_app
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.v1.routes.executions.models import (
    GraphExecutionDetailResponse,
    GraphExecutionListResponse,
    RunExecutionResponse,
)
from notarius_api.v1.routes.saved_graphs.models import SavedGraphResponse
from notarius_api.settings import Settings


def _saved_text_graph_payload(name: str, text: str) -> dict[str, object]:
    return {
        "name": name,
        "nodes": [
            {
                "id": "text",
                "operator_id": "text.input",
                "operator_version": 1,
                "config": {"text": text},
                "position": {"x": 0, "y": 0},
            }
        ],
        "edges": [],
    }


def _start_saved_text_execution(
    client: TestClient,
    graph: SavedGraphResponse,
) -> RunExecutionResponse:
    response = client.post(
        "/v1/executions",
        json={
            "nodes": [
                {
                    "id": "text",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "remember this"},
                }
            ],
            "edges": [],
            "scope": "selected-with-dependencies",
            "graph_id": str(graph.id),
            "graph_revision": graph.revision,
        },
    )
    assert response.status_code == 202
    started = RunExecutionResponse.model_validate(response.json())
    for _ in range(100):
        poll = client.get(f"/v1/executions/{started.execution_id}")
        assert poll.status_code == 200
        current = RunExecutionResponse.model_validate(poll.json())
        if current.status in {"cancelled", "succeeded", "failed"}:
            return current
    raise AssertionError("Execution did not reach a terminal status")


def test_saved_graph_execution_history_lists_filters_and_renders_artifacts(
    builtin_client: TestClient,
) -> None:
    created_response = builtin_client.post(
        "/v1/graphs",
        json=_saved_text_graph_payload("History", "remember this"),
    )
    assert created_response.status_code == 201
    graph = SavedGraphResponse.model_validate(created_response.json())

    first = _start_saved_text_execution(builtin_client, graph)
    assert first.status == "succeeded"

    listing_response = builtin_client.get(f"/v1/graphs/{graph.id}/executions")
    assert listing_response.status_code == 200
    listing = GraphExecutionListResponse.model_validate(listing_response.json())
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

    detail_response = builtin_client.get(
        f"/v1/graphs/{graph.id}/executions/{first.execution_id}"
    )
    assert detail_response.status_code == 200
    detail = GraphExecutionDetailResponse.model_validate(detail_response.json())
    assert [(result.node_id, result.status) for result in detail.node_results] == [
        ("text", "succeeded")
    ]
    output = detail.node_results[0].outputs[0]
    assert output.port == "text"
    assert output.artifacts[0].text == '"remember this"'

    succeeded = builtin_client.get(
        f"/v1/graphs/{graph.id}/executions",
        params={"status": "succeeded", "node_id": "text"},
    )
    assert succeeded.status_code == 200
    assert len(GraphExecutionListResponse.model_validate(succeeded.json()).items) == 1
    no_match = builtin_client.get(
        f"/v1/graphs/{graph.id}/executions",
        params={"status": "failed", "node_id": "missing"},
    )
    assert no_match.status_code == 200
    assert GraphExecutionListResponse.model_validate(no_match.json()).items == []
    assert (
        builtin_client.get(
            f"/v1/graphs/{graph.id}/executions",
            params={"node_id": "   "},
        ).status_code
        == 422
    )
    assert (
        builtin_client.get(
            f"/v1/graphs/{graph.id}/executions",
            params={"cursor": "not-a-cursor"},
        ).status_code
        == 422
    )
    assert (
        builtin_client.get(
            f"/v1/graphs/{uuid4()}/executions/{first.execution_id}"
        ).status_code
        == 404
    )

    update_payload = _saved_text_graph_payload("History r2", "remember this")
    update_payload["expected_revision"] = graph.revision
    updated_response = builtin_client.put(
        f"/v1/graphs/{graph.id}",
        json=update_payload,
    )
    assert updated_response.status_code == 200
    updated_graph = SavedGraphResponse.model_validate(updated_response.json())
    assert updated_graph.revision == 2

    second = _start_saved_text_execution(builtin_client, updated_graph)
    assert second.status == "succeeded"
    revision_one_response = builtin_client.get(
        f"/v1/graphs/{graph.id}/executions",
        params={"graph_revision": 1},
    )
    revision_one = GraphExecutionListResponse.model_validate(
        revision_one_response.json()
    )
    assert [item.execution_id for item in revision_one.items] == [first.execution_id]
    revision_two_response = builtin_client.get(
        f"/v1/graphs/{graph.id}/executions",
        params={"graph_revision": 2},
    )
    revision_two = GraphExecutionListResponse.model_validate(
        revision_two_response.json()
    )
    assert [item.execution_id for item in revision_two.items] == [second.execution_id]
    all_revisions_response = builtin_client.get(f"/v1/graphs/{graph.id}/executions")
    assert {
        item.graph_revision
        for item in GraphExecutionListResponse.model_validate(
            all_revisions_response.json()
        ).items
    } == {1, 2}
    first_page_response = builtin_client.get(
        f"/v1/graphs/{graph.id}/executions",
        params={"limit": 1},
    )
    first_page = GraphExecutionListResponse.model_validate(first_page_response.json())
    assert len(first_page.items) == 1
    assert first_page.next_cursor is not None
    second_page_response = builtin_client.get(
        f"/v1/graphs/{graph.id}/executions",
        params={"limit": 1, "cursor": first_page.next_cursor},
    )
    second_page = GraphExecutionListResponse.model_validate(second_page_response.json())
    assert len(second_page.items) == 1
    assert second_page.items[0].execution_id != first_page.items[0].execution_id


def test_saved_graph_execution_is_not_accepted_without_its_revision(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/executions",
        json={
            "nodes": [],
            "edges": [],
            "graph_id": str(uuid4()),
            "graph_revision": 1,
        },
    )

    assert response.status_code == 404
    assert "Saved graph revision" in response.json()["detail"]


def test_duplicate_saved_node_ids_become_a_browsable_failed_execution(
    builtin_client: TestClient,
) -> None:
    created_response = builtin_client.post(
        "/v1/graphs",
        json=_saved_text_graph_payload("Invalid history", "duplicate"),
    )
    assert created_response.status_code == 201
    graph = SavedGraphResponse.model_validate(created_response.json())
    duplicate_node = {
        "id": "text",
        "operator_id": "text.input",
        "operator_version": 1,
        "config": {"text": "duplicate"},
    }

    start_response = builtin_client.post(
        "/v1/executions",
        json={
            "nodes": [duplicate_node, duplicate_node],
            "scope": "all",
            "graph_id": str(graph.id),
            "graph_revision": graph.revision,
        },
    )
    assert start_response.status_code == 202
    execution = RunExecutionResponse.model_validate(start_response.json())
    for _ in range(100):
        poll_response = builtin_client.get(f"/v1/executions/{execution.execution_id}")
        assert poll_response.status_code == 200
        execution = RunExecutionResponse.model_validate(poll_response.json())
        if execution.status == "failed":
            break
    assert execution.status == "failed"
    assert execution.error is not None
    assert "Duplicate node ids" in execution.error

    detail_response = builtin_client.get(
        f"/v1/graphs/{graph.id}/executions/{execution.execution_id}"
    )
    assert detail_response.status_code == 200
    detail = GraphExecutionDetailResponse.model_validate(detail_response.json())
    assert detail.status == "failed"
    assert detail.requested_node_ids == ["text"]
    assert detail.node_results == []
    assert detail.error is not None
    assert "Duplicate node ids" in detail.error


async def _seed_active_execution(database_url: str) -> tuple[UUID, UUID]:
    database = create_database(database_url)
    registry = build_plugin_registry(builtin_plugins(), external_plugins=())
    saved_graphs = SavedGraphService(
        lambda: SqlAlchemySavedGraphUnitOfWork(database.sessions),
        registry,
    )
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        graph = await saved_graphs.create(
            name="Interrupted",
            document=SavedGraphDocument(),
        )
        execution_id = uuid4()
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            await unit_of_work.execution_history.add(
                GraphExecution(
                    execution_id=execution_id,
                    graph_id=graph.id,
                    graph_revision=graph.revision,
                    status="running",
                    requested_node_ids=(),
                )
            )
            await unit_of_work.commit()
        return graph.id, execution_id
    finally:
        await database.dispose()


def test_application_startup_marks_stale_active_execution_failed(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'recovery.sqlite3'}"
    graph_id, execution_id = asyncio.run(_seed_active_execution(database_url))
    application = create_app(
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
            execution_backend="inline",
        )
    )

    with TestClient(application) as client:
        response = client.get(f"/v1/graphs/{graph_id}/executions/{execution_id}")

    assert response.status_code == 200
    detail = GraphExecutionDetailResponse.model_validate(response.json())
    assert detail.status == "failed"
    assert detail.finished_at is not None
    assert detail.error is not None
    assert "API process stopped" in detail.error
