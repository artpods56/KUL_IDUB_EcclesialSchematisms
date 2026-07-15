import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import delete

from notarius_core.artifacts import ArtifactObject, ArtifactRef
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_persistence import schema
from notarius_persistence.database import create_database
from notarius_persistence.orm import metadata
from notarius_persistence.unit_of_work import SqlAlchemyUnitOfWork

from notarius_api.main import create_app
from notarius_api.schemas.saved_graphs import SavedGraphResponse
from notarius_api.schemas.workbench import (
    GraphMaterializationsResponse,
    RunPortOutputResponse,
    RunResponse,
)
from notarius_api.settings import Settings


async def _create_schema(database_url: str) -> None:
    database = create_database(database_url)
    try:
        async with database.engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
    finally:
        await database.dispose()


async def _delete_artifact(database_url: str, artifact_id: UUID) -> None:
    database = create_database(database_url)
    try:
        async with database.engine.begin() as connection:
            await connection.execute(
                delete(schema.artifact_objects).where(
                    schema.artifact_objects.c.id == artifact_id
                )
            )
    finally:
        await database.dispose()


async def _persist_partially_accessible_materialization(
    database_url: str,
    graph_id: UUID,
    graph_revision: int,
) -> None:
    database = create_database(database_url)
    accessible = ArtifactObject(
        artifact_type="scalar.integer",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"value": 13},
    )
    missing = ArtifactRef(
        artifact_id=UUID("00000000-0000-0000-0000-000000000999"),
        artifact_type="scalar.integer",
        schema_version=1,
    )
    try:
        async with SqlAlchemyUnitOfWork(database.sessions) as unit_of_work:
            await unit_of_work.artifacts.add(accessible)
            await unit_of_work.materialized_outputs.upsert(
                MaterializedNodeOutputs(
                    graph_id=graph_id,
                    graph_revision=graph_revision,
                    node_id="add-subtract",
                    workflow_run_id=UUID(
                        "00000000-0000-0000-0000-000000000998"
                    ),
                    outputs={
                        "accessible": accessible.ref(),
                        "missing": missing,
                    },
                )
            )
            await unit_of_work.commit()
    finally:
        await database.dispose()


@contextmanager
def _client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def durable_api(tmp_path: Path) -> tuple[Settings, str]:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'materializations.sqlite3'}"
    asyncio.run(_create_schema(database_url))
    return (
        Settings(
            workspace=tmp_path / "workbench",
            database_url=SecretStr(database_url),
        ),
        database_url,
    )


def _graph_payload() -> dict[str, object]:
    nodes: list[tuple[str, str, dict[str, object]]] = [
        ("nine", "arithmetic.number", {"value": 9}),
        ("four", "arithmetic.number", {"value": 4}),
        ("add-subtract", "arithmetic.add_subtract", {}),
        ("multiply", "arithmetic.multiply", {}),
    ]
    edges = _edges()
    return {
        "name": "Durable arithmetic graph",
        "nodes": [
            {
                "id": node_id,
                "operator_id": operator_id,
                "operator_version": 1,
                "config": config,
                "position": {"x": float(index * 200), "y": 20.0},
            }
            for index, (node_id, operator_id, config) in enumerate(nodes)
        ],
        "edges": [
            {"id": f"edge-{index}", **edge}
            for index, edge in enumerate(edges, start=1)
        ],
    }


def _edges() -> list[dict[str, object]]:
    return [
        {
            "from_node": "nine",
            "from_port": "value",
            "to_node": "add-subtract",
            "to_port": "left",
        },
        {
            "from_node": "four",
            "from_port": "value",
            "to_node": "add-subtract",
            "to_port": "right",
        },
        {
            "from_node": "add-subtract",
            "from_port": "result",
            "to_node": "multiply",
            "to_port": "left",
            "projection": {"path": ["addition"]},
        },
        {
            "from_node": "add-subtract",
            "from_port": "result",
            "to_node": "multiply",
            "to_port": "right",
            "projection": {"path": ["subtraction"]},
        },
    ]


def _full_run_payload(graph_id: str, graph_revision: int) -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": "nine",
                "operator_id": "arithmetic.number",
                "operator_version": 1,
                "config": {"value": 9},
            },
            {
                "id": "four",
                "operator_id": "arithmetic.number",
                "operator_version": 1,
                "config": {"value": 4},
            },
            {
                "id": "add-subtract",
                "operator_id": "arithmetic.add_subtract",
                "operator_version": 1,
                "config": {},
            },
            {
                "id": "multiply",
                "operator_id": "arithmetic.multiply",
                "operator_version": 1,
                "config": {},
            },
        ],
        "edges": _edges(),
        "graph_id": graph_id,
        "graph_revision": graph_revision,
    }


def _downstream_run_payload(
    graph_id: str,
    graph_revision: int,
    *,
    pinned_value: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "nodes": [
            {
                "id": "multiply",
                "operator_id": "arithmetic.multiply",
                "operator_version": 1,
                "config": {},
            }
        ],
        "edges": _edges()[2:],
        "graph_id": graph_id,
        "graph_revision": graph_revision,
    }
    if pinned_value is not None:
        payload["pinned_outputs"] = [
            {
                "from_node": "add-subtract",
                "from_port": "result",
                "value": pinned_value,
            }
        ]
    return payload


def _output(run: RunResponse, node_id: str) -> RunPortOutputResponse:
    node_run = next(item for item in run.node_runs if item.node_id == node_id)
    return node_run.outputs[0]


@pytest.mark.parametrize(
    "graph_context",
    [
        {"graph_id": "00000000-0000-0000-0000-000000000001"},
        {"graph_revision": 1},
    ],
)
def test_run_graph_context_requires_id_and_revision_together(
    durable_api: tuple[Settings, str],
    graph_context: dict[str, object],
) -> None:
    settings, _ = durable_api
    with _client(settings) as client:
        response = client.post(
            "/v1/runs",
            json={"nodes": [], "edges": [], **graph_context},
        )

    assert response.status_code == 422
    assert "graph_id and graph_revision must be provided together" in str(
        response.json()
    )


def test_materialization_context_validates_graph_revision_and_fragment(
    durable_api: tuple[Settings, str],
) -> None:
    settings, _ = durable_api
    missing_graph_id = "00000000-0000-0000-0000-000000000404"
    with _client(settings) as client:
        missing = client.get(
            f"/v1/graphs/{missing_graph_id}/materializations",
            params={"graph_revision": 1},
        )
        graph = SavedGraphResponse.model_validate(
            client.post("/v1/graphs", json=_graph_payload()).json()
        )
        stale = client.get(
            f"/v1/graphs/{graph.id}/materializations",
            params={"graph_revision": graph.revision + 1},
        )
        rogue = client.post(
            "/v1/runs",
            json={
                "nodes": [
                    {
                        "id": "rogue-node",
                        "operator_id": "arithmetic.number",
                        "operator_version": 1,
                        "config": {"value": 99},
                    }
                ],
                "edges": [],
                "graph_id": str(graph.id),
                "graph_revision": graph.revision,
            },
        )

    assert missing.status_code == 404
    assert stale.status_code == 409
    assert rogue.status_code == 422
    assert "does not belong to saved graph" in rogue.json()["detail"]


def test_graph_context_run_rejects_omitted_saved_incoming_edge(
    durable_api: tuple[Settings, str],
) -> None:
    settings, _ = durable_api
    with _client(settings) as client:
        graph = SavedGraphResponse.model_validate(
            client.post("/v1/graphs", json=_graph_payload()).json()
        )
        payload = _full_run_payload(str(graph.id), graph.revision)
        payload["edges"] = _edges()[:1] + _edges()[2:]

        response = client.post("/v1/runs", json=payload)
        materializations = client.get(
            f"/v1/graphs/{graph.id}/materializations",
            params={"graph_revision": graph.revision},
        )

    assert response.status_code == 422
    assert "1 missing and 0 unexpected or duplicated" in response.json()["detail"]
    assert materializations.status_code == 200
    assert GraphMaterializationsResponse.model_validate(
        materializations.json()
    ).node_runs == []


def test_graph_context_run_rejects_duplicated_saved_incoming_edge(
    durable_api: tuple[Settings, str],
) -> None:
    settings, _ = durable_api
    with _client(settings) as client:
        graph = SavedGraphResponse.model_validate(
            client.post("/v1/graphs", json=_graph_payload()).json()
        )
        payload = _full_run_payload(str(graph.id), graph.revision)
        payload["edges"] = [*_edges(), _edges()[1]]

        response = client.post("/v1/runs", json=payload)
        materializations = client.get(
            f"/v1/graphs/{graph.id}/materializations",
            params={"graph_revision": graph.revision},
        )

    assert response.status_code == 422
    assert "0 missing and 1 unexpected or duplicated" in response.json()["detail"]
    assert materializations.status_code == 200
    assert GraphMaterializationsResponse.model_validate(
        materializations.json()
    ).node_runs == []


def test_full_run_persists_outputs_and_fresh_app_reuses_them_for_downstream_run(
    durable_api: tuple[Settings, str],
) -> None:
    settings, _ = durable_api

    with _client(settings) as client:
        created = client.post("/v1/graphs", json=_graph_payload())
        assert created.status_code == 201
        graph = SavedGraphResponse.model_validate(created.json())

        full_run = client.post(
            "/v1/runs",
            json=_full_run_payload(str(graph.id), graph.revision),
        )
        assert full_run.status_code == 200
        assert RunResponse.model_validate(full_run.json()).status == "succeeded"

        materializations = client.get(
            f"/v1/graphs/{graph.id}/materializations",
            params={"graph_revision": graph.revision},
        )
        assert materializations.status_code == 200
        materialized = GraphMaterializationsResponse.model_validate(
            materializations.json()
        )
        assert {
            node_run.node_id for node_run in materialized.node_runs
        } == {"nine", "four", "add-subtract", "multiply"}

    with _client(settings) as fresh_client:
        reloaded = fresh_client.get(
            f"/v1/graphs/{graph.id}/materializations",
            params={"graph_revision": graph.revision},
        )
        assert reloaded.status_code == 200
        reloaded_result = GraphMaterializationsResponse.model_validate(reloaded.json())
        assert len(reloaded_result.node_runs) == 4
        add_subtract_run = next(
            node_run
            for node_run in reloaded_result.node_runs
            if node_run.node_id == "add-subtract"
        )
        persisted_value = add_subtract_run.outputs[0].value

        downstream = fresh_client.post(
            "/v1/runs",
            json=_downstream_run_payload(
                str(graph.id),
                graph.revision,
                pinned_value=persisted_value.model_dump(mode="json"),
            ),
        )
        assert downstream.status_code == 200
        downstream_result = RunResponse.model_validate(downstream.json())
        assert downstream_result.status == "succeeded"
        assert _output(downstream_result, "multiply").artifacts[0].text == "65"


def test_downstream_run_without_materialization_returns_dependency_guidance(
    durable_api: tuple[Settings, str],
) -> None:
    settings, _ = durable_api
    with _client(settings) as client:
        graph = SavedGraphResponse.model_validate(
            client.post("/v1/graphs", json=_graph_payload()).json()
        )
        standalone = RunResponse.model_validate(
            client.post(
                "/v1/runs",
                json={
                    "nodes": [
                        {
                            "id": "standalone",
                            "operator_id": "arithmetic.number",
                            "operator_version": 1,
                            "config": {"value": 5},
                        }
                    ],
                    "edges": [],
                },
            ).json()
        )
        unrelated_value = _output(standalone, "standalone").value

        response = client.post(
            "/v1/runs",
            json=_downstream_run_payload(
                str(graph.id),
                graph.revision,
                pinned_value=unrelated_value.model_dump(mode="json"),
            ),
        )

    assert response.status_code == 422
    assert "Run with dependencies" in response.json()["detail"]


def test_inaccessible_artifact_is_filtered_and_blocks_downstream_reuse(
    durable_api: tuple[Settings, str],
) -> None:
    settings, database_url = durable_api
    with _client(settings) as client:
        graph = SavedGraphResponse.model_validate(
            client.post("/v1/graphs", json=_graph_payload()).json()
        )
        full_run = RunResponse.model_validate(
            client.post(
                "/v1/runs",
                json=_full_run_payload(str(graph.id), graph.revision),
            ).json()
        )
        add_subtract_value = _output(full_run, "add-subtract").value
        assert isinstance(add_subtract_value, ArtifactRef)
        artifact_id = add_subtract_value.artifact_id
        pinned_value = add_subtract_value.model_dump(mode="json")

    asyncio.run(_delete_artifact(database_url, artifact_id))

    with _client(settings) as client:
        materializations = client.get(
            f"/v1/graphs/{graph.id}/materializations",
            params={"graph_revision": graph.revision},
        )
        materialized = GraphMaterializationsResponse.model_validate(
            materializations.json()
        )
        visible_nodes = {
            node_run.node_id for node_run in materialized.node_runs
        }
        assert "add-subtract" not in visible_nodes

        downstream = client.post(
            "/v1/runs",
            json=_downstream_run_payload(
                str(graph.id),
                graph.revision,
                pinned_value=pinned_value,
            ),
        )

    assert downstream.status_code == 422
    assert "no accessible materialized artifact" in downstream.json()["detail"]
    assert "Run with dependencies" in downstream.json()["detail"]


def test_materialization_response_keeps_accessible_sibling_ports(
    durable_api: tuple[Settings, str],
) -> None:
    settings, database_url = durable_api
    with _client(settings) as client:
        graph = SavedGraphResponse.model_validate(
            client.post("/v1/graphs", json=_graph_payload()).json()
        )

    asyncio.run(
        _persist_partially_accessible_materialization(
            database_url,
            graph.id,
            graph.revision,
        )
    )

    with _client(settings) as client:
        response = client.get(
            f"/v1/graphs/{graph.id}/materializations",
            params={"graph_revision": graph.revision},
        )

    assert response.status_code == 200
    materializations = GraphMaterializationsResponse.model_validate(response.json())
    node_run = next(
        item for item in materializations.node_runs if item.node_id == "add-subtract"
    )
    assert [output.port for output in node_run.outputs] == ["accessible"]


def test_saved_run_rejects_pin_that_is_not_the_latest_materialization(
    durable_api: tuple[Settings, str],
) -> None:
    settings, _ = durable_api
    with _client(settings) as client:
        graph = SavedGraphResponse.model_validate(
            client.post("/v1/graphs", json=_graph_payload()).json()
        )
        persisted = client.post(
            "/v1/runs",
            json=_full_run_payload(str(graph.id), graph.revision),
        )
        assert persisted.status_code == 200

        alternate = client.post(
            "/v1/runs",
            json={
                "nodes": [
                    {
                        "id": "eight",
                        "operator_id": "arithmetic.number",
                        "operator_version": 1,
                        "config": {"value": 8},
                    },
                    {
                        "id": "three",
                        "operator_id": "arithmetic.number",
                        "operator_version": 1,
                        "config": {"value": 3},
                    },
                    {
                        "id": "add-subtract",
                        "operator_id": "arithmetic.add_subtract",
                        "operator_version": 1,
                        "config": {},
                    },
                ],
                "edges": [
                    {
                        "from_node": "eight",
                        "from_port": "value",
                        "to_node": "add-subtract",
                        "to_port": "left",
                    },
                    {
                        "from_node": "three",
                        "from_port": "value",
                        "to_node": "add-subtract",
                        "to_port": "right",
                    },
                ],
            },
        )
        assert alternate.status_code == 200
        alternate_result = RunResponse.model_validate(alternate.json())
        pinned_value = _output(alternate_result, "add-subtract").value
        assert isinstance(pinned_value, ArtifactRef)

        downstream = client.post(
            "/v1/runs",
            json=_downstream_run_payload(
                str(graph.id),
                graph.revision,
                pinned_value=pinned_value.model_dump(mode="json"),
            ),
        )

    assert downstream.status_code == 422
    assert "is not the latest materialized output" in downstream.json()["detail"]
