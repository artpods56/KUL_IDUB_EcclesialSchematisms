from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, ClassVar, Literal
from uuid import UUID

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from starlette.applications import Starlette

from notarius_mcp.models import (
    ArtifactTypeBindingRequest,
    ArtifactTypeKeyRequest,
    CollaborativeHeadResponse,
    CreateSavedGraphRequest,
    GraphPointRequest,
    Identifier,
    NodeInspection,
    NodeSearchResult,
    NodeSearchSummary,
    SavedGraphConversionRequest,
    SavedGraphEdgeRequest,
    SavedGraphInputPlugRequest,
    SavedGraphListResponse,
    SavedGraphNodeLayoutRequest,
    SavedGraphNodeRequest,
    SavedGraphProjectionRequest,
    SavedGraphResponse,
    SubmitGraphCommandResponse,
    UpdateSavedGraphRequest,
)
from notarius_mcp.operations import McpOperationError
from notarius_mcp.request_context import current_mcp_binding


_READ_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
_CREATE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
_REPLACE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=True,
    openWorldHint=False,
)
_COMMAND_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class ToolInputModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class SavedGraphNodeDraft(ToolInputModel):
    id: Identifier
    operator_id: Identifier
    operator_version: int = Field(ge=1)
    config: dict[str, JsonValue] = Field(default_factory=dict)
    position: GraphPointRequest | None = None
    layout: SavedGraphNodeLayoutRequest | None = None
    input_plugs: list[SavedGraphInputPlugRequest] = Field(default_factory=list)
    artifact_type_bindings: list[ArtifactTypeBindingRequest] = Field(
        default_factory=list
    )


class SavedGraphEdgeDraft(ToolInputModel):
    id: Identifier | None = None
    enabled: bool = True
    from_node: Identifier
    from_port: Identifier
    to_node: Identifier
    to_port: Identifier
    to_plug: Identifier | None = None
    collection_mode: Literal["direct", "map"] = "direct"
    projection: SavedGraphProjectionRequest | None = None
    conversion_path: list[SavedGraphConversionRequest] = Field(
        default_factory=list,
        max_length=8,
    )
    route_offset: GraphPointRequest | None = None


class SavedGraphDraft(ToolInputModel):
    name: str = Field(min_length=1, max_length=160)
    nodes: list[SavedGraphNodeDraft] = Field(default_factory=list)
    edges: list[SavedGraphEdgeDraft] = Field(default_factory=list)


@asynccontextmanager
async def _empty_lifespan(
    _server: FastMCP[dict[str, object]],
) -> AsyncIterator[dict[str, object]]:
    # Caller identity is request-scoped via ContextVar, never lifespan state.
    yield {}


def _as_tool_error(error: McpOperationError, operation: str) -> ToolError:
    if error.status_code == 401:
        return ToolError(
            f"Authentication failed while attempting to {operation}."
        )
    if error.status_code == 403:
        return ToolError(
            f"The current personal access token lacks permission to {operation}."
        )
    if error.status_code == 404:
        return ToolError(
            f"The requested Notarius resource was not found while attempting to "
            f"{operation}."
        )
    if error.status_code == 409:
        return ToolError(
            f"The collaborative head changed before the operation could complete "
            f"while attempting to {operation}. Call get_live_head and reconcile "
            "before trying again."
        )
    if error.status_code == 422:
        return ToolError(
            f"The request was structurally invalid while attempting to {operation}."
            + (f" {error.message}" if error.message else "")
        )
    return ToolError(f"Could not {operation}.")


def _normalize_graph_draft(graph: SavedGraphDraft) -> CreateSavedGraphRequest:
    plugs_by_node = {
        node.id: list(node.input_plugs)
        for node in graph.nodes
    }
    plug_ids_by_node = {
        node_id: {plug.id for plug in plugs}
        for node_id, plugs in plugs_by_node.items()
    }
    for edge in graph.edges:
        if edge.to_plug is None or edge.to_node not in plugs_by_node:
            continue
        if edge.to_plug in plug_ids_by_node[edge.to_node]:
            continue
        plugs_by_node[edge.to_node].append(
            SavedGraphInputPlugRequest(
                id=edge.to_plug,
                port=edge.to_port,
            )
        )
        plug_ids_by_node[edge.to_node].add(edge.to_plug)

    nodes: list[SavedGraphNodeRequest] = []
    for index, node in enumerate(graph.nodes):
        position = node.position
        if position is None:
            position = GraphPointRequest(
                x=float((index % 4) * 360),
                y=float((index // 4) * 240),
            )
        nodes.append(
            SavedGraphNodeRequest(
                id=node.id,
                operator_id=node.operator_id,
                operator_version=node.operator_version,
                config=node.config,
                position=position,
                layout=node.layout,
                input_plugs=plugs_by_node[node.id],
                artifact_type_bindings=node.artifact_type_bindings,
            )
        )

    used_edge_ids = {edge.id for edge in graph.edges if edge.id is not None}
    edges: list[SavedGraphEdgeRequest] = []
    for index, edge in enumerate(graph.edges):
        edge_id = edge.id
        if edge_id is None:
            edge_id = f"edge-{index + 1}"
            suffix = 2
            while edge_id in used_edge_ids:
                edge_id = f"edge-{index + 1}-{suffix}"
                suffix += 1
            used_edge_ids.add(edge_id)
        edges.append(
            SavedGraphEdgeRequest(
                id=edge_id,
                enabled=edge.enabled,
                from_node=edge.from_node,
                from_port=edge.from_port,
                to_node=edge.to_node,
                to_port=edge.to_port,
                to_plug=edge.to_plug,
                collection_mode=edge.collection_mode,
                projection=edge.projection,
                conversion_path=edge.conversion_path,
                route_offset=edge.route_offset,
            )
        )

    return CreateSavedGraphRequest(
        name=graph.name,
        nodes=nodes,
        edges=edges,
    )


def _graph_validation_tool_error(error: ValidationError, operation: str) -> ToolError:
    issues: list[str] = []
    for issue in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:5]:
        location = ".".join(str(part) for part in issue["loc"])
        message = issue["msg"]
        issues.append(f"{location}: {message}" if location else message)
    return ToolError(
        f"Could not {operation} because the graph draft is structurally invalid: "
        + "; ".join(issues)
    )


mcp: FastMCP[dict[str, object]] = FastMCP(
    name="Notarius Graphs",
    instructions=(
        "Call search_nodes and inspect_node before authoring, and use exact operator "
        "versions, ports, artifact types, projections, and conversions. Call "
        "get_live_head immediately before submit_graph_command or replace_graph and "
        "pass the observed room epoch and collaboration sequence. Workspace identity "
        "comes from the personal access token and must not be supplied as a tool "
        "argument. Create and replace save structurally valid drafts only; they do "
        "not prove executability. Never put credentials or secrets in graph "
        "configuration or tool arguments."
    ),
    lifespan=_empty_lifespan,
    strict_input_validation=True,
    mask_error_details=True,
)


@mcp.tool(annotations=_READ_ANNOTATIONS)
async def search_nodes(
    context: Context,
    query: str = "",
    plugin_slug: str | None = None,
    accepts: ArtifactTypeKeyRequest | None = None,
    produces: ArtifactTypeKeyRequest | None = None,
    include_hidden: bool = False,
    limit: Annotated[int, Field(ge=1, le=100)] = 20,
) -> NodeSearchResult:
    """Search the live node catalog without returning its large JSON schemas."""
    del context
    binding = current_mcp_binding()
    try:
        registry = await binding.operations.get_registry(binding.caller)
    except McpOperationError as exc:
        raise _as_tool_error(exc, "search the node catalog") from exc

    normalized_query = query.strip().casefold()
    normalized_plugin_slug = (
        plugin_slug.strip().casefold() if plugin_slug is not None else None
    )
    matches: list[NodeSearchSummary] = []
    for node in registry.nodes:
        if not include_hidden and not node.catalog_visible:
            continue
        if (
            normalized_plugin_slug is not None
            and node.plugin_slug.casefold() != normalized_plugin_slug
        ):
            continue
        searchable_text = " ".join(
            (node.operator_id, node.title, node.description, node.plugin_slug)
        ).casefold()
        if normalized_query not in searchable_text:
            continue
        if accepts is not None and not any(
            port.artifact_type is not None
            and port.artifact_type.id == accepts.id
            and port.artifact_type.schema_version == accepts.schema_version
            for port in node.inputs
        ):
            continue
        if produces is not None and not any(
            port.artifact_type is not None
            and port.artifact_type.id == produces.id
            and port.artifact_type.schema_version == produces.schema_version
            for port in node.outputs
        ):
            continue
        matches.append(
            NodeSearchSummary(
                operator_id=node.operator_id,
                operator_version=node.operator_version,
                plugin_slug=node.plugin_slug,
                title=node.title,
                description=node.description,
                inputs=node.inputs,
                outputs=node.outputs,
            )
        )

    total_matches = len(matches)
    return NodeSearchResult(
        nodes=matches[:limit],
        total_matches=total_matches,
        truncated=total_matches > limit,
    )


@mcp.tool(annotations=_READ_ANNOTATIONS)
async def inspect_node(
    context: Context,
    operator_id: str,
    operator_version: Annotated[int, Field(ge=1)],
) -> NodeInspection:
    """Inspect one exact node and the artifact metadata needed to route edges."""
    del context
    binding = current_mcp_binding()
    try:
        registry = await binding.operations.get_registry(binding.caller)
    except McpOperationError as exc:
        raise _as_tool_error(exc, "inspect a catalog node") from exc

    matches = [
        node
        for node in registry.nodes
        if node.operator_id == operator_id
        and node.operator_version == operator_version
    ]
    if not matches:
        raise ToolError(
            f"No node exists for operator {operator_id!r} at version "
            f"{operator_version}."
        )
    if len(matches) > 1:
        raise ToolError(
            f"The catalog contains multiple nodes for operator {operator_id!r} at "
            f"version {operator_version}; exact inspection is ambiguous."
        )
    return NodeInspection(
        node=matches[0],
        artifact_types=registry.artifact_types,
        artifact_conversions=registry.artifact_conversions,
    )


@mcp.tool(annotations=_READ_ANNOTATIONS)
async def list_graphs(context: Context) -> SavedGraphListResponse:
    """List saved graphs in the token-bound workspace."""
    del context
    binding = current_mcp_binding()
    try:
        return await binding.operations.list_graphs(binding.caller)
    except McpOperationError as exc:
        raise _as_tool_error(exc, "list saved graphs") from exc


@mcp.tool(annotations=_READ_ANNOTATIONS)
async def get_live_head(
    context: Context,
    graph_id: UUID,
) -> CollaborativeHeadResponse:
    """Get the collaborative live head for one graph in the token-bound workspace."""
    del context
    binding = current_mcp_binding()
    try:
        return await binding.operations.get_live_head(binding.caller, graph_id)
    except McpOperationError as exc:
        raise _as_tool_error(exc, "get a collaborative live head") from exc


@mcp.tool(annotations=_CREATE_ANNOTATIONS)
async def create_graph(
    context: Context,
    graph: SavedGraphDraft,
) -> SavedGraphResponse:
    """Create a structurally valid saved graph through sequence-1 bootstrap."""
    del context
    try:
        request = _normalize_graph_draft(graph)
    except ValidationError as exc:
        raise _graph_validation_tool_error(exc, "create the graph draft") from exc
    binding = current_mcp_binding()
    try:
        return await binding.operations.create_graph(binding.caller, request)
    except McpOperationError as exc:
        raise _as_tool_error(exc, "create a saved graph draft") from exc


@mcp.tool(annotations=_REPLACE_ANNOTATIONS)
async def replace_graph(
    context: Context,
    graph_id: UUID,
    expected_revision: Annotated[int, Field(ge=1)],
    graph: SavedGraphDraft,
) -> SavedGraphResponse:
    """Replace a checkpointed graph through collaboration-aware epoch reset."""
    del context
    try:
        normalized_graph = _normalize_graph_draft(graph)
        request = UpdateSavedGraphRequest(
            name=normalized_graph.name,
            nodes=normalized_graph.nodes,
            edges=normalized_graph.edges,
            expected_revision=expected_revision,
        )
    except ValidationError as exc:
        raise _graph_validation_tool_error(exc, "replace the graph draft") from exc
    binding = current_mcp_binding()
    try:
        return await binding.operations.replace_graph(
            binding.caller,
            graph_id,
            request,
        )
    except McpOperationError as exc:
        raise _as_tool_error(exc, "replace a saved graph draft") from exc


@mcp.tool(annotations=_COMMAND_ANNOTATIONS)
async def submit_graph_command(
    context: Context,
    graph_id: UUID,
    command_id: UUID,
    room_epoch: UUID,
    observed_sequence: Annotated[int, Field(ge=0)],
    command: dict[str, JsonValue],
) -> SubmitGraphCommandResponse:
    """Submit one semantic collaboration command against the observed live head."""
    del context
    binding = current_mcp_binding()
    try:
        return await binding.operations.submit_command(
            binding.caller,
            graph_id=graph_id,
            command_id=command_id,
            room_epoch=room_epoch,
            observed_sequence=observed_sequence,
            command=command,
        )
    except McpOperationError as exc:
        raise _as_tool_error(exc, "submit a graph command") from exc


def create_streamable_http_app(*, path: str = "/") -> Starlette:
    """Build the mountable stateless Streamable HTTP ASGI application."""

    return mcp.http_app(
        path=path,
        transport="streamable-http",
        stateless_http=True,
    )


__all__ = [
    "SavedGraphDraft",
    "SavedGraphEdgeDraft",
    "SavedGraphNodeDraft",
    "create_streamable_http_app",
    "mcp",
]
