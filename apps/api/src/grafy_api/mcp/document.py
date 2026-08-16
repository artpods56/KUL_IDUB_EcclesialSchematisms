"""Convert MCP graph drafts into core saved-graph documents."""

from grafy_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraphArtifactTypeBinding,
    SavedGraphConversion,
    SavedGraphDocument,
    SavedGraphEdge,
    SavedGraphInputPlug,
    SavedGraphNode,
    SavedGraphNodeLayout,
    SavedGraphProjection,
)
from grafy_core.artifacts import ArtifactTypeKey

from grafy_mcp.models import SavedGraphWriteRequest


def document_from_mcp_request(request: SavedGraphWriteRequest) -> SavedGraphDocument:
    return SavedGraphDocument(
        nodes=tuple(
            SavedGraphNode(
                id=node.id,
                operator_id=node.operator_id,
                operator_version=node.operator_version,
                config=node.config,
                position=GraphPoint(
                    x=node.position.x,
                    y=node.position.y,
                ),
                layout=(
                    SavedGraphNodeLayout(
                        width=node.layout.width,
                        body_height=node.layout.body_height,
                        appendix_height=node.layout.appendix_height,
                    )
                    if node.layout is not None
                    else None
                ),
                input_plugs=tuple(
                    SavedGraphInputPlug(
                        id=plug.id,
                        port=plug.port,
                    )
                    for plug in node.input_plugs
                ),
                artifact_type_bindings=tuple(
                    SavedGraphArtifactTypeBinding(
                        variable=binding.variable,
                        artifact_type=ArtifactTypeKey(
                            binding.artifact_type.id,
                            binding.artifact_type.schema_version,
                        ),
                    )
                    for binding in node.artifact_type_bindings
                ),
            )
            for node in request.nodes
        ),
        edges=tuple(
            SavedGraphEdge(
                id=edge.id,
                enabled=edge.enabled,
                from_node=edge.from_node,
                from_port=edge.from_port,
                to_node=edge.to_node,
                to_port=edge.to_port,
                to_plug=edge.to_plug,
                collection_mode=edge.collection_mode,
                projection=(
                    SavedGraphProjection(path=tuple(edge.projection.path))
                    if edge.projection is not None
                    else None
                ),
                conversion_path=tuple(
                    SavedGraphConversion(
                        id=item.id,
                        version=item.version,
                    )
                    for item in edge.conversion_path
                ),
                route_offset=(
                    GraphPoint(x=edge.route_offset.x, y=edge.route_offset.y)
                    if edge.route_offset is not None
                    else None
                ),
            )
            for edge in request.edges
        ),
    )


__all__ = ["document_from_mcp_request"]
