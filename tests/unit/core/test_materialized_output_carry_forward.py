from datetime import UTC, datetime
from uuid import UUID

from notarius_core.artifacts import ArtifactRef
from notarius_core.domain.materialized_outputs import (
    MaterializedNodeOutputs,
    materializations_for_compatible_nodes,
)
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraphDocument,
    SavedGraphEdge,
    SavedGraphNode,
)


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000007")
GRAPH_ID = UUID("00000000-0000-0000-0000-000000000101")


def _node(
    node_id: str,
    *,
    config: dict[str, object] | None = None,
    x: float = 0,
) -> SavedGraphNode:
    return SavedGraphNode(
        id=node_id,
        operator_id="arithmetic.number",
        operator_version=1,
        config=config or {"value": 1},
        position=GraphPoint(x=x, y=0),
    )


def _edge(edge_id: str, source: str, target: str) -> SavedGraphEdge:
    return SavedGraphEdge(
        id=edge_id,
        from_node=source,
        from_port="result",
        to_node=target,
        to_port="value",
    )


def _materialization(node_id: str, revision: int = 1) -> MaterializedNodeOutputs:
    return MaterializedNodeOutputs(
        workspace_id=WORKSPACE_ID,
        graph_id=GRAPH_ID,
        graph_revision=revision,
        node_id=node_id,
        workflow_run_id=UUID("00000000-0000-0000-0000-000000000201"),
        outputs={
            "result": ArtifactRef(
                artifact_id=UUID("00000000-0000-0000-0000-000000000301"),
                artifact_type="scalar.integer",
                schema_version=1,
            )
        },
        materialized_at=datetime(2026, 8, 8, 10, 0, tzinfo=UTC),
    )


def test_carry_forward_keeps_unchanged_nodes_across_position_edits() -> None:
    previous = SavedGraphDocument(
        nodes=(_node("source"), _node("target", x=120)),
        edges=(_edge("e1", "source", "target"),),
    )
    next_document = SavedGraphDocument(
        nodes=(_node("source", x=40), _node("target", x=200)),
        edges=(_edge("e1", "source", "target"),),
    )

    carried = materializations_for_compatible_nodes(
        previous_document=previous,
        next_document=next_document,
        previous_materializations=[
            _materialization("source"),
            _materialization("target"),
        ],
        next_revision=2,
    )

    assert {item.node_id for item in carried} == {"source", "target"}
    assert all(item.graph_revision == 2 for item in carried)


def test_carry_forward_drops_nodes_with_config_or_edge_changes() -> None:
    previous = SavedGraphDocument(
        nodes=(_node("source"), _node("target")),
        edges=(_edge("e1", "source", "target"),),
    )
    next_document = SavedGraphDocument(
        nodes=(
            _node("source", config={"value": 9}),
            _node("target"),
        ),
        edges=(_edge("e1", "source", "target"),),
    )

    carried = materializations_for_compatible_nodes(
        previous_document=previous,
        next_document=next_document,
        previous_materializations=[
            _materialization("source"),
            _materialization("target"),
        ],
        next_revision=2,
    )

    assert [item.node_id for item in carried] == ["target"]
