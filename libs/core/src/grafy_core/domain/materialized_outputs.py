from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from grafy_core.domain.artifact_outputs import (
    ArtifactOutputEnvelope,
    ArtifactOutputValue,
    artifact_outputs_from_storage,
    artifact_outputs_to_storage,
    normalize_artifact_outputs,
)

if TYPE_CHECKING:
    from grafy_core.domain.saved_graphs import (
        SavedGraphDocument,
        SavedGraphEdge,
        SavedGraphNode,
    )


# Compatibility aliases for existing callers while the shared artifact-output
# vocabulary becomes the persistence boundary used by both materializations and
# invocation-cache entries.
MaterializedOutputValue = ArtifactOutputValue
MaterializedOutputEnvelope = ArtifactOutputEnvelope


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _node_execution_signature(node: "SavedGraphNode") -> tuple[object, ...]:
    return (
        node.operator_id,
        node.operator_version,
        node.config_dict(),
        tuple((plug.id, plug.port) for plug in node.input_plugs),
        tuple(
            (
                binding.variable,
                binding.artifact_type.id,
                binding.artifact_type.schema_version,
            )
            for binding in node.artifact_type_bindings
        ),
    )


def _incoming_edge_signature(edge: "SavedGraphEdge") -> tuple[object, ...]:
    projection = None
    if edge.projection is not None:
        projection = tuple(edge.projection.path)
    return (
        edge.enabled,
        edge.from_node,
        edge.from_port,
        edge.to_node,
        edge.to_port,
        edge.to_plug,
        edge.collection_mode,
        projection,
        tuple((step.id, step.version) for step in edge.conversion_path),
    )


def _incoming_edges_by_target(
    document: "SavedGraphDocument",
) -> dict[str, tuple[tuple[object, ...], ...]]:
    grouped: dict[str, list[tuple[object, ...]]] = {}
    for edge in document.edges:
        grouped.setdefault(edge.to_node, []).append(_incoming_edge_signature(edge))
    return {
        node_id: tuple(sorted(signatures))
        for node_id, signatures in grouped.items()
    }


def materializations_for_compatible_nodes(
    *,
    previous_document: "SavedGraphDocument",
    next_document: "SavedGraphDocument",
    previous_materializations: Sequence["MaterializedNodeOutputs"],
    next_revision: int,
) -> list["MaterializedNodeOutputs"]:
    """Copy pins forward for nodes whose execution identity is unchanged.

    Position, layout, and presentation may change across a saved revision without
    invalidating already materialized outputs. Node config, plugs, bindings, and
    incoming edge topology must match exactly.
    """
    if next_revision < 1:
        raise ValueError("Materialized output graph revision must be at least 1")

    previous_nodes = {node.id: node for node in previous_document.nodes}
    next_nodes = {node.id: node for node in next_document.nodes}
    previous_incoming = _incoming_edges_by_target(previous_document)
    next_incoming = _incoming_edges_by_target(next_document)

    carried: list[MaterializedNodeOutputs] = []
    for materialization in previous_materializations:
        previous_node = previous_nodes.get(materialization.node_id)
        next_node = next_nodes.get(materialization.node_id)
        if previous_node is None or next_node is None:
            continue
        if _node_execution_signature(previous_node) != _node_execution_signature(
            next_node
        ):
            continue
        if previous_incoming.get(materialization.node_id) != next_incoming.get(
            materialization.node_id
        ):
            continue
        carried.append(
            replace(
                materialization,
                graph_revision=next_revision,
            )
        )
    return carried


@dataclass
class MaterializedNodeOutputs:
    workspace_id: UUID
    graph_id: UUID
    graph_revision: int
    node_id: str
    workflow_run_id: UUID
    outputs: dict[str, MaterializedOutputValue]
    materialized_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.graph_revision < 1:
            raise ValueError("Materialized output graph revision must be at least 1")
        self.node_id = self.node_id.strip()
        if self.node_id == "":
            raise ValueError("Materialized output node id must not be blank")
        if len(self.node_id) > 255:
            raise ValueError(
                "Materialized output node id must be at most 255 characters"
            )
        if self.materialized_at.tzinfo is None:
            raise ValueError("Materialized output timestamp must be timezone-aware")

        self.outputs = normalize_artifact_outputs(self.outputs)

    def storage_envelopes(self) -> list[dict[str, object]]:
        return self.outputs_to_storage(self.outputs)

    @staticmethod
    def outputs_to_storage(
        outputs: dict[str, MaterializedOutputValue],
    ) -> list[dict[str, object]]:
        return artifact_outputs_to_storage(outputs)

    @staticmethod
    def outputs_from_storage(
        value: object,
    ) -> dict[str, MaterializedOutputValue]:
        return artifact_outputs_from_storage(value)
