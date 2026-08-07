from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest

from notarius_mcp.client import NotariusApiClient, NotariusApiError
from notarius_mcp.models import (
    CreateSavedGraphRequest,
    GraphPointRequest,
    SavedGraphNodeRequest,
    UpdateSavedGraphRequest,
)


GRAPH_ID = UUID("8d570df6-188d-40f2-b62a-dedbe35637cf")
WORKSPACE_ID = UUID("11111111-2222-3333-4444-555555555555")
WORKSPACE_ROOT = f"/v1/workspaces/{WORKSPACE_ID}"
UPDATED_AT = datetime(2026, 7, 23, 10, 30, tzinfo=UTC)


def _registry_response() -> dict[str, object]:
    return {
        "plugins": [
            {
                "slug": "core",
                "title": "Core",
                "origin": "builtin",
            }
        ],
        "artifact_types": [
            {
                "key": {"id": "text.plain", "schema_version": 1},
                "title": "Plain text",
                "payload_schema": {"type": "object"},
                "field_projections": [],
            }
        ],
        "artifact_conversions": [],
        "nodes": [
            {
                "operator_id": "core.literal",
                "operator_version": 1,
                "plugin_slug": "core",
                "title": "Literal",
                "description": "Produce a literal value.",
                "config_schema": {"type": "object"},
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "inputs": [],
                "outputs": [
                    {
                        "name": "value",
                        "direction": "output",
                        "artifact_type": {
                            "id": "text.plain",
                            "schema_version": 1,
                        },
                        "shape": "one",
                        "accepted_shapes": ["one"],
                    }
                ],
            }
        ],
        "unavailable_modules": [],
        "future_registry_field": True,
    }


def _graph_response(
    *,
    name: str = "Example graph",
    revision: int = 1,
) -> dict[str, object]:
    return {
        "id": str(GRAPH_ID),
        "name": name,
        "revision": revision,
        "created_at": UPDATED_AT.isoformat(),
        "updated_at": UPDATED_AT.isoformat(),
        "nodes": [
            {
                "id": "literal",
                "operator_id": "core.literal",
                "operator_version": 1,
                "config": {"value": "hello"},
                "position": {"x": 20.0, "y": 40.0},
                "input_plugs": [],
                "artifact_type_bindings": [],
                "future_node_field": "ignored",
            }
        ],
        "edges": [],
        "future_graph_field": "ignored",
    }


def _create_request(name: str = "Example graph") -> CreateSavedGraphRequest:
    return CreateSavedGraphRequest(
        name=name,
        nodes=[
            SavedGraphNodeRequest(
                id="literal",
                operator_id="core.literal",
                operator_version=1,
                config={"value": "hello"},
                position=GraphPointRequest(x=20.0, y=40.0),
            )
        ],
    )


def _api_client(http_client: httpx.AsyncClient) -> NotariusApiClient:
    return NotariusApiClient(http_client, workspace_id=WORKSPACE_ID)


@pytest.mark.asyncio
async def test_get_registry_validates_the_typed_response() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == f"{WORKSPACE_ROOT}/nodes"
        return httpx.Response(200, json=_registry_response())

    async with httpx.AsyncClient(
        base_url="http://notarius.test",
        transport=httpx.MockTransport(respond),
    ) as http_client:
        registry = await _api_client(http_client).get_registry()

    assert registry.plugins[0].origin == "builtin"
    assert registry.nodes[0].operator_id == "core.literal"
    assert registry.nodes[0].outputs[0].artifact_type is not None
    assert registry.nodes[0].outputs[0].artifact_type.id == "text.plain"


@pytest.mark.asyncio
async def test_graph_crud_uses_expected_paths_and_statuses() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"{WORKSPACE_ROOT}/graphs":
            return httpx.Response(
                200,
                json={
                    "graphs": [
                        {
                            "id": str(GRAPH_ID),
                            "name": "Example graph",
                            "revision": 1,
                            "node_count": 1,
                            "edge_count": 0,
                            "updated_at": UPDATED_AT.isoformat(),
                        }
                    ]
                },
            )
        if request.method == "GET":
            assert request.url.path == f"{WORKSPACE_ROOT}/graphs/{GRAPH_ID}"
            return httpx.Response(200, json=_graph_response())
        if request.method == "POST":
            assert request.url.path == f"{WORKSPACE_ROOT}/graphs"
            payload = CreateSavedGraphRequest.model_validate_json(request.content)
            assert payload.name == "Example graph"
            return httpx.Response(201, json=_graph_response())
        assert request.method == "PUT"
        assert request.url.path == f"{WORKSPACE_ROOT}/graphs/{GRAPH_ID}"
        payload = UpdateSavedGraphRequest.model_validate_json(request.content)
        assert payload.expected_revision == 1
        return httpx.Response(
            200,
            json=_graph_response(name="Updated graph", revision=2),
        )

    async with httpx.AsyncClient(
        base_url="http://notarius.test",
        transport=httpx.MockTransport(respond),
    ) as http_client:
        client = _api_client(http_client)
        graph_list = await client.list_graphs()
        graph = await client.get_graph(GRAPH_ID)
        created = await client.create_graph(_create_request())
        replaced = await client.replace_graph(
            GRAPH_ID,
            UpdateSavedGraphRequest(
                **_create_request("Updated graph").model_dump(),
                expected_revision=1,
            ),
        )

    assert graph_list.graphs[0].id == GRAPH_ID
    assert graph.nodes[0].config == {"value": "hello"}
    assert created.revision == 1
    assert replaced.revision == 2


@pytest.mark.asyncio
async def test_get_graph_preserves_404_detail() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Graph was not found"})

    async with httpx.AsyncClient(
        base_url="http://notarius.test",
        transport=httpx.MockTransport(respond),
    ) as http_client:
        with pytest.raises(NotariusApiError) as raised:
            await _api_client(http_client).get_graph(GRAPH_ID)

    assert raised.value.method == "GET"
    assert raised.value.path == f"{WORKSPACE_ROOT}/graphs/{GRAPH_ID}"
    assert raised.value.status_code == 404
    assert raised.value.detail == "Graph was not found"
    assert raised.value.raw_body is None


@pytest.mark.asyncio
async def test_replace_graph_preserves_409_detail() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "Revision conflict"})

    async with httpx.AsyncClient(
        base_url="http://notarius.test",
        transport=httpx.MockTransport(respond),
    ) as http_client:
        with pytest.raises(NotariusApiError) as raised:
            await _api_client(http_client).replace_graph(
                GRAPH_ID,
                UpdateSavedGraphRequest(
                    **_create_request().model_dump(),
                    expected_revision=1,
                ),
            )

    assert raised.value.status_code == 409
    assert raised.value.detail == "Revision conflict"


@pytest.mark.asyncio
async def test_create_graph_preserves_422_validation_detail() -> None:
    validation_detail: list[dict[str, object]] = [
        {
            "type": "value_error",
            "loc": ["body", "nodes"],
            "msg": "Value error, invalid graph",
        }
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": validation_detail})

    async with httpx.AsyncClient(
        base_url="http://notarius.test",
        transport=httpx.MockTransport(respond),
    ) as http_client:
        with pytest.raises(NotariusApiError) as raised:
            await _api_client(http_client).create_graph(_create_request())

    assert raised.value.status_code == 422
    assert raised.value.detail == validation_detail


@pytest.mark.asyncio
async def test_non_json_error_preserves_raw_body() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="upstream unavailable")

    async with httpx.AsyncClient(
        base_url="http://notarius.test",
        transport=httpx.MockTransport(respond),
    ) as http_client:
        with pytest.raises(NotariusApiError) as raised:
            await _api_client(http_client).list_graphs()

    assert raised.value.status_code == 502
    assert raised.value.detail is None
    assert raised.value.raw_body == "upstream unavailable"


@pytest.mark.asyncio
async def test_transport_error_is_contextual_and_chained() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(
        base_url="http://notarius.test",
        transport=httpx.MockTransport(respond),
    ) as http_client:
        with pytest.raises(NotariusApiError) as raised:
            await _api_client(http_client).get_registry()

    assert raised.value.method == "GET"
    assert raised.value.path == f"{WORKSPACE_ROOT}/nodes"
    assert raised.value.status_code is None
    assert raised.value.detail == "connection refused"
    assert isinstance(raised.value.__cause__, httpx.ConnectError)
