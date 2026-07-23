from collections.abc import Sequence
from datetime import datetime, timezone
from functools import partial
from uuid import UUID

import httpx
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from pydantic import JsonValue

from notarius_mcp.models import (
    CreateSavedGraphRequest,
    UpdateSavedGraphRequest,
)
from notarius_mcp.server import mcp


_GRAPH_ID = UUID("12345678-1234-5678-1234-567812345678")
_NOW = datetime(2026, 7, 23, tzinfo=timezone.utc).isoformat()


def _catalog_node(
    operator_id: str,
    *,
    title: str,
    catalog_visible: bool = True,
) -> dict[str, JsonValue]:
    table_type: dict[str, JsonValue] = {"id": "table", "schema_version": 1}
    return {
        "operator_id": operator_id,
        "operator_version": 1,
        "plugin_slug": "sql",
        "title": title,
        "description": "Run SQL over artifact tables",
        "config_schema": {},
        "input_schema": {},
        "output_schema": {},
        "inputs": [
            {
                "name": "relations",
                "direction": "input",
                "artifact_type": table_type,
                "artifact_type_variable": None,
                "shape": "one",
                "accepted_shapes": ["one"],
                "instance_plugs": True,
                "variadic": True,
                "required": True,
            }
        ],
        "outputs": [
            {
                "name": "table",
                "direction": "output",
                "artifact_type": table_type,
                "artifact_type_variable": None,
                "shape": "one",
                "accepted_shapes": ["one"],
                "instance_plugs": False,
                "variadic": False,
                "required": True,
            }
        ],
        "secret_inputs": [],
        "module_graph_id": None,
        "module_graph_revision": None,
        "catalog_visible": catalog_visible,
    }


def _registry(nodes: Sequence[JsonValue]) -> dict[str, JsonValue]:
    return {
        "plugins": [{"slug": "sql", "title": "SQL", "origin": "external"}],
        "artifact_types": [
            {
                "key": {"id": "table", "schema_version": 1},
                "title": "Table",
                "payload_schema": {},
                "field_projections": [],
            }
        ],
        "artifact_conversions": [
            {
                "key": {"id": "table.identity", "version": 1},
                "source_artifact_type": {"id": "table", "schema_version": 1},
                "target_artifact_type": {"id": "table", "schema_version": 1},
                "title": "Table identity",
            }
        ],
        "nodes": list(nodes),
        "unavailable_modules": [],
    }


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: httpx.AsyncBaseTransport,
) -> None:
    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        partial(async_client, transport=handler),
    )


def _graph_response(
    request: CreateSavedGraphRequest | UpdateSavedGraphRequest,
    revision: int,
) -> dict[str, object]:
    response = request.model_dump(exclude={"expected_revision"})
    response["id"] = str(_GRAPH_ID)
    response["revision"] = revision
    response["created_at"] = _NOW
    response["updated_at"] = _NOW
    return response


@pytest.mark.asyncio
async def test_server_exposes_six_closed_world_tools_with_write_hints() -> None:
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}

    assert set(tools) == {
        "search_nodes",
        "inspect_node",
        "list_graphs",
        "get_graph",
        "create_graph",
        "replace_graph",
    }
    for name in ("search_nodes", "inspect_node", "list_graphs", "get_graph"):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is True
        assert annotations.openWorldHint is False

    create_annotations = tools["create_graph"].annotations
    assert create_annotations is not None
    assert create_annotations.readOnlyHint is False
    assert create_annotations.destructiveHint is False
    assert create_annotations.idempotentHint is False
    assert create_annotations.openWorldHint is False

    replace_annotations = tools["replace_graph"].annotations
    assert replace_annotations is not None
    assert replace_annotations.readOnlyHint is False
    assert replace_annotations.destructiveHint is True
    assert replace_annotations.idempotentHint is True
    assert replace_annotations.openWorldHint is False


@pytest.mark.asyncio
async def test_search_and_inspect_nodes_use_the_live_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(
        [
            _catalog_node("sql.artifacts.query", title="SQL artifact query"),
            _catalog_node("sql.artifacts.merge", title="SQL artifact merge"),
            _catalog_node(
                "sql.artifacts.hidden",
                title="SQL hidden node",
                catalog_visible=False,
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/nodes"
        return httpx.Response(200, json=registry)

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    async with Client(mcp) as client:
        search = await client.call_tool(
            "search_nodes",
            {
                "query": "sQl",
                "plugin_slug": "sql",
                "accepts": {"id": "table", "schema_version": 1},
                "produces": {"id": "table", "schema_version": 1},
                "limit": 1,
            },
        )
        inspection = await client.call_tool(
            "inspect_node",
            {"operator_id": "sql.artifacts.query", "operator_version": 1},
        )

    assert search.structured_content is not None
    assert search.structured_content["total_matches"] == 2
    assert search.structured_content["truncated"] is True
    assert len(search.structured_content["nodes"]) == 1
    assert inspection.structured_content is not None
    assert inspection.structured_content["node"]["operator_id"] == (
        "sql.artifacts.query"
    )
    assert inspection.structured_content["artifact_types"][0]["key"] == {
        "id": "table",
        "schema_version": 1,
    }
    assert inspection.structured_content["artifact_conversions"][0]["key"] == {
        "id": "table.identity",
        "version": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("match_count", [0, 2])
async def test_inspect_node_rejects_missing_or_duplicate_exact_matches(
    monkeypatch: pytest.MonkeyPatch,
    match_count: int,
) -> None:
    node = _catalog_node("duplicate.operator", title="Duplicate")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_registry([node] * match_count))

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="No node exists|multiple nodes"):
            await client.call_tool(
                "inspect_node",
                {"operator_id": "duplicate.operator", "operator_version": 1},
            )


@pytest.mark.asyncio
async def test_graph_read_tools_return_typed_api_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_request = CreateSavedGraphRequest(name="Draft")
    graph_response = _graph_response(graph_request, revision=3)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/graphs":
            return httpx.Response(
                200,
                json={
                    "graphs": [
                        {
                            "id": str(_GRAPH_ID),
                            "name": "Draft",
                            "revision": 3,
                            "node_count": 0,
                            "edge_count": 0,
                            "updated_at": _NOW,
                        }
                    ]
                },
            )
        assert request.url.path == f"/v1/graphs/{_GRAPH_ID}"
        return httpx.Response(200, json=graph_response)

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    async with Client(mcp) as client:
        listing = await client.call_tool("list_graphs")
        graph = await client.call_tool("get_graph", {"graph_id": str(_GRAPH_ID)})

    assert listing.structured_content is not None
    assert listing.structured_content["graphs"][0]["revision"] == 3
    assert graph.structured_content is not None
    assert graph.structured_content["id"] == str(_GRAPH_ID)


@pytest.mark.asyncio
async def test_graph_writes_normalize_transport_details_and_preserve_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_requests: list[CreateSavedGraphRequest] = []
    replaced_requests: list[UpdateSavedGraphRequest] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            parsed = CreateSavedGraphRequest.model_validate_json(request.content)
            created_requests.append(parsed)
            return httpx.Response(201, json=_graph_response(parsed, revision=1))
        parsed = UpdateSavedGraphRequest.model_validate_json(request.content)
        replaced_requests.append(parsed)
        return httpx.Response(200, json=_graph_response(parsed, revision=2))

    _install_transport(monkeypatch, httpx.MockTransport(handler))
    draft: dict[str, object] = {
        "name": "Artifact join",
        "nodes": [
            {
                "id": "source",
                "operator_id": "postgres.query",
                "operator_version": 1,
            },
            {
                "id": "query",
                "operator_id": "sql.artifacts.query",
                "operator_version": 1,
            },
        ],
        "edges": [
            {
                "from_node": "source",
                "from_port": "table",
                "to_node": "query",
                "to_port": "relations",
                "to_plug": "customers",
            }
        ],
    }

    async with Client(mcp) as client:
        created = await client.call_tool("create_graph", {"graph": draft})
        replaced = await client.call_tool(
            "replace_graph",
            {
                "graph_id": str(_GRAPH_ID),
                "expected_revision": 1,
                "graph": draft,
            },
        )

    assert created.structured_content is not None
    assert created.structured_content["revision"] == 1
    assert replaced.structured_content is not None
    assert replaced.structured_content["revision"] == 2
    assert created_requests[0].nodes[0].position.x == 0.0
    assert created_requests[0].nodes[0].position.y == 0.0
    assert created_requests[0].nodes[1].position.x == 360.0
    assert created_requests[0].edges[0].id == "edge-1"
    assert created_requests[0].nodes[1].input_plugs[0].id == "customers"
    assert created_requests[0].nodes[1].input_plugs[0].port == "relations"
    assert replaced_requests[0].expected_revision == 1


@pytest.mark.asyncio
async def test_api_error_is_safe_and_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"detail": "sensitive raw backend detail"},
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    async with Client(mcp) as client:
        with pytest.raises(ToolError) as error:
            await client.call_tool(
                "replace_graph",
                {
                    "graph_id": str(_GRAPH_ID),
                    "expected_revision": 1,
                    "graph": {"name": "Draft"},
                },
            )

    message = str(error.value)
    assert "graph revision changed" in message
    assert "Call get_graph" in message
    assert "sensitive raw backend detail" not in message


@pytest.mark.asyncio
async def test_structural_validation_error_is_actionable_without_echoing_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("The invalid draft must not reach the HTTP API")

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    async with Client(mcp) as client:
        with pytest.raises(ToolError) as error:
            await client.call_tool(
                "create_graph",
                {
                    "graph": {
                        "name": "Invalid draft",
                        "nodes": [
                            {
                                "id": "source",
                                "operator_id": "source.operator",
                                "operator_version": 1,
                            }
                        ],
                        "edges": [
                            {
                                "from_node": "source",
                                "from_port": "output",
                                "to_node": "missing-target",
                                "to_port": "input",
                            }
                        ],
                    }
                },
            )

    message = str(error.value)
    assert "structurally invalid" in message
    assert "missing target node" in message
    assert "source.operator" not in message


@pytest.mark.asyncio
async def test_api_validation_error_reports_safe_issue_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "type": "value_error",
                        "loc": ["body", "nodes", 0, "config"],
                        "msg": "Configuration is invalid",
                        "input": {"password": "must-not-be-echoed"},
                    }
                ]
            },
        )

    _install_transport(monkeypatch, httpx.MockTransport(handler))

    async with Client(mcp) as client:
        with pytest.raises(ToolError) as error:
            await client.call_tool(
                "create_graph",
                {"graph": {"name": "Rejected draft"}},
            )

    message = str(error.value)
    assert "body.nodes.0.config: Configuration is invalid" in message
    assert "must-not-be-echoed" not in message
