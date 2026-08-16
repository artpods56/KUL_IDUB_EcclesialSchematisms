"""Request-scoped MCP caller and operations binding.

First-delivery Streamable HTTP is stateless: each request binds a fresh caller
context and never retains Authorization in process-global state.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass

from grafy_mcp.operations import GraphWorkspaceOperations, McpCallerContext


@dataclass(frozen=True, slots=True)
class McpRequestBinding:
    caller: McpCallerContext
    operations: GraphWorkspaceOperations


_CURRENT_BINDING: ContextVar[McpRequestBinding | None] = ContextVar(
    "grafy_mcp_request_binding",
    default=None,
)


def bind_mcp_request(
    caller: McpCallerContext,
    operations: GraphWorkspaceOperations,
) -> Token[McpRequestBinding | None]:
    return _CURRENT_BINDING.set(
        McpRequestBinding(caller=caller, operations=operations)
    )


def reset_mcp_request(token: Token[McpRequestBinding | None]) -> None:
    _CURRENT_BINDING.reset(token)


def current_mcp_binding() -> McpRequestBinding:
    binding = _CURRENT_BINDING.get()
    if binding is None:
        raise RuntimeError("MCP request binding is not available")
    return binding


__all__ = [
    "McpRequestBinding",
    "bind_mcp_request",
    "current_mcp_binding",
    "reset_mcp_request",
]
