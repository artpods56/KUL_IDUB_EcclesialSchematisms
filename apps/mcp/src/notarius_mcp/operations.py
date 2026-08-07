"""Narrow MCP delivery contract for request-scoped workspace operations.

The API composition root injects a concrete implementation backed by the same
identity, catalog, and collaboration application services used by browser
transports. This module must not import FastAPI routes, SQLAlchemy, or
construct a unit of work.
"""

from typing import Protocol
from uuid import UUID

from pydantic import JsonValue

from notarius_mcp.models import (
    CollaborativeHeadResponse,
    CreateSavedGraphRequest,
    NodeRegistryResponse,
    SavedGraphListResponse,
    SavedGraphResponse,
    SubmitGraphCommandResponse,
    UpdateSavedGraphRequest,
)


class McpCallerContext:
    """Server-resolved actor and workspace for one MCP request."""

    __slots__ = ("user_id", "workspace_id", "credential_reference", "scopes")

    def __init__(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        credential_reference: str,
        scopes: frozenset[str],
    ) -> None:
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.credential_reference = credential_reference
        self.scopes = scopes


class McpOperationError(Exception):
    """Bounded tool failure mapped from application/auth outcomes."""

    def __init__(self, *, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)


class GraphWorkspaceOperations(Protocol):
    async def get_registry(self, caller: McpCallerContext) -> NodeRegistryResponse: ...

    async def list_graphs(self, caller: McpCallerContext) -> SavedGraphListResponse: ...

    async def get_live_head(
        self,
        caller: McpCallerContext,
        graph_id: UUID,
    ) -> CollaborativeHeadResponse: ...

    async def create_graph(
        self,
        caller: McpCallerContext,
        request: CreateSavedGraphRequest,
    ) -> SavedGraphResponse: ...

    async def replace_graph(
        self,
        caller: McpCallerContext,
        graph_id: UUID,
        request: UpdateSavedGraphRequest,
    ) -> SavedGraphResponse: ...

    async def submit_command(
        self,
        caller: McpCallerContext,
        *,
        graph_id: UUID,
        command_id: UUID,
        room_epoch: UUID,
        observed_sequence: int,
        command: JsonValue,
    ) -> SubmitGraphCommandResponse: ...


__all__ = [
    "GraphWorkspaceOperations",
    "McpCallerContext",
    "McpOperationError",
]
