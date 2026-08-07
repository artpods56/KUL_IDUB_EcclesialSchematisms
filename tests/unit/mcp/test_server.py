"""In-process MCP tool behavior with injected operations."""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from notarius_mcp.models import (
    CollaborativeHeadResponse,
    CreateSavedGraphRequest,
    NodeRegistryResponse,
    SavedGraphListResponse,
    SavedGraphResponse,
    SubmitGraphCommandResponse,
    UpdateSavedGraphRequest,
)
from notarius_mcp.operations import McpCallerContext, McpOperationError
from notarius_mcp.request_context import bind_mcp_request, reset_mcp_request
from notarius_mcp.server import mcp


_GRAPH_ID = UUID("12345678-1234-5678-1234-567812345678")
_NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


class _FakeOps:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_registry(self, caller: McpCallerContext) -> NodeRegistryResponse:
        self.calls.append(f"registry:{caller.workspace_id}")
        return NodeRegistryResponse.model_validate(
            {
                "plugins": [{"slug": "sql", "title": "SQL", "origin": "external"}],
                "artifact_types": [
                    {
                        "key": {"id": "table", "schema_version": 1},
                        "title": "Table",
                        "payload_schema": {},
                        "field_projections": [],
                    }
                ],
                "artifact_conversions": [],
                "nodes": [
                    {
                        "operator_id": "sql.query",
                        "operator_version": 1,
                        "plugin_slug": "sql",
                        "title": "SQL Query",
                        "description": "Run SQL",
                        "config_schema": {},
                        "input_schema": {},
                        "output_schema": {},
                        "inputs": [],
                        "outputs": [],
                        "secret_inputs": [],
                        "module_graph_id": None,
                        "module_graph_revision": None,
                        "catalog_visible": True,
                    }
                ],
                "unavailable_modules": [],
            }
        )

    async def list_graphs(self, caller: McpCallerContext) -> SavedGraphListResponse:
        self.calls.append("list")
        return SavedGraphListResponse(graphs=[])

    async def get_live_head(
        self,
        caller: McpCallerContext,
        graph_id: UUID,
    ) -> CollaborativeHeadResponse:
        self.calls.append(f"head:{graph_id}")
        return CollaborativeHeadResponse(
            graph_id=graph_id,
            room_epoch=UUID(int=9),
            collaboration_sequence=1,
            checkpoint_sequence=1,
            checkpoint_revision=1,
            name="Head",
            updated_at=_NOW,
            nodes=[],
            edges=[],
        )

    async def create_graph(
        self,
        caller: McpCallerContext,
        request: CreateSavedGraphRequest,
    ) -> SavedGraphResponse:
        self.calls.append(f"create:{request.name}")
        return SavedGraphResponse(
            id=_GRAPH_ID,
            name=request.name,
            revision=1,
            created_at=_NOW,
            updated_at=_NOW,
            nodes=[],
            edges=[],
        )

    async def replace_graph(
        self,
        caller: McpCallerContext,
        graph_id: UUID,
        request: UpdateSavedGraphRequest,
    ) -> SavedGraphResponse:
        raise McpOperationError(status_code=409, message="Conflict.")

    async def submit_command(
        self,
        caller: McpCallerContext,
        *,
        graph_id: UUID,
        command_id: UUID,
        room_epoch: UUID,
        observed_sequence: int,
        command: object,
    ) -> SubmitGraphCommandResponse:
        del caller, command_id, room_epoch, observed_sequence, command
        raise McpOperationError(status_code=403, message="Forbidden.")


@pytest.mark.asyncio
async def test_tools_use_request_scoped_operations() -> None:
    ops = _FakeOps()
    caller = McpCallerContext(
        user_id=UUID(int=1),
        workspace_id=UUID(int=2),
        credential_reference="pat:test",
        scopes=frozenset({"view_graph", "create_graph"}),
    )
    token = bind_mcp_request(caller, ops)
    try:
        async with Client(mcp) as client:
            search = await client.call_tool("search_nodes", {"query": "sql"})
            assert search.structured_content is not None
            assert search.structured_content["total_matches"] == 1
            head = await client.call_tool(
                "get_live_head",
                {"graph_id": str(_GRAPH_ID)},
            )
            assert head.structured_content is not None
            assert head.structured_content["graph_id"] == str(_GRAPH_ID)
            created = await client.call_tool(
                "create_graph",
                {"graph": {"name": "Draft", "nodes": [], "edges": []}},
            )
            assert created.structured_content is not None
            assert created.structured_content["name"] == "Draft"
            with pytest.raises(ToolError):
                await client.call_tool(
                    "replace_graph",
                    {
                        "graph_id": str(_GRAPH_ID),
                        "expected_revision": 1,
                        "graph": {"name": "X", "nodes": [], "edges": []},
                    },
                )
    finally:
        reset_mcp_request(token)

    assert "registry:" in ops.calls[0]
    assert f"head:{_GRAPH_ID}" in ops.calls
    assert "create:Draft" in ops.calls
