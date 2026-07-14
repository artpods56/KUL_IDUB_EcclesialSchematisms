from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from notarius_core.domain.errors import SavedGraphRevisionConflictError
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraph,
    SavedGraphDocument,
    SavedGraphEdge,
    SavedGraphNode,
)


def _node(node_id: str) -> SavedGraphNode:
    return SavedGraphNode(
        id=node_id,
        operator_id="example.operator",
        operator_version=1,
        config={},
        position=GraphPoint(x=10.0, y=20.0),
    )


def _edge(
    edge_id: str,
    *,
    from_node: str = "source",
    to_node: str = "target",
) -> SavedGraphEdge:
    return SavedGraphEdge(
        id=edge_id,
        from_node=from_node,
        from_port="result",
        to_node=to_node,
        to_port="value",
    )


def test_saved_graph_document_allows_drafts_without_executable_connections() -> None:
    empty_draft = SavedGraphDocument()
    incomplete_draft = SavedGraphDocument(
        nodes=(
            SavedGraphNode(
                id="unconnected",
                operator_id="plugin.that-is-not-installed",
                operator_version=99,
                config={"unfinished": True},
                position=GraphPoint(x=0.0, y=0.0),
            ),
        )
    )

    assert empty_draft.nodes == ()
    assert empty_draft.edges == ()
    assert incomplete_draft.nodes[0].operator_id == "plugin.that-is-not-installed"
    assert incomplete_draft.edges == ()


def test_saved_graph_node_config_is_deeply_immutable_and_serializable() -> None:
    node = SavedGraphNode(
        id="configured",
        operator_id="example.operator",
        operator_version=1,
        config={"nested": {"enabled": True}, "values": [1, 2]},
        position=GraphPoint(x=0.0, y=0.0),
    )

    with pytest.raises(TypeError):
        cast(dict[str, object], node.config)["new"] = "value"
    with pytest.raises(TypeError):
        cast(dict[str, object], node.config["nested"])["enabled"] = False

    assert node.config_dict() == {
        "nested": {"enabled": True},
        "values": [1, 2],
    }
    assert node.model_dump(mode="json")["config"] == node.config_dict()


def test_saved_graph_document_requires_unique_node_ids() -> None:
    with pytest.raises(ValidationError, match="node ids must be unique"):
        SavedGraphDocument(nodes=(_node("duplicate"), _node("duplicate")))


def test_saved_graph_document_requires_unique_edge_ids() -> None:
    with pytest.raises(ValidationError, match="edge ids must be unique"):
        SavedGraphDocument(
            nodes=(_node("source"), _node("target")),
            edges=(_edge("duplicate"), _edge("duplicate")),
        )


@pytest.mark.parametrize(
    ("edge", "message"),
    [
        (_edge("missing-source", from_node="absent"), "missing source node absent"),
        (_edge("missing-target", to_node="absent"), "missing target node absent"),
    ],
)
def test_saved_graph_document_rejects_edges_to_missing_nodes(
    edge: SavedGraphEdge,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        SavedGraphDocument(
            nodes=(_node("source"), _node("target")),
            edges=(edge,),
        )


@pytest.mark.parametrize("name", ["", "   ", "x" * 161])
def test_saved_graph_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="Saved graph name"):
        SavedGraph(name=name, document=SavedGraphDocument())


def test_saved_graph_requires_positive_revision_and_aware_timestamps() -> None:
    aware = datetime(2026, 7, 14, 8, 30, tzinfo=UTC)
    naive = datetime(2026, 7, 14, 8, 30)

    with pytest.raises(ValueError, match="revision must be at least 1"):
        SavedGraph(
            name="Draft",
            document=SavedGraphDocument(),
            revision=0,
            created_at=aware,
            updated_at=aware,
        )

    with pytest.raises(ValueError, match="timestamps must be timezone-aware"):
        SavedGraph(
            name="Draft",
            document=SavedGraphDocument(),
            created_at=naive,
            updated_at=aware,
        )


def test_saved_graph_replace_normalizes_name_and_advances_revision() -> None:
    graph_id = UUID("00000000-0000-0000-0000-000000000001")
    created_at = datetime(2026, 7, 14, 8, 30, tzinfo=UTC)
    updated_at = datetime(2026, 7, 14, 9, 45, tzinfo=UTC)
    replacement = SavedGraphDocument(nodes=(_node("draft-node"),))
    graph = SavedGraph(
        id=graph_id,
        name="Original",
        document=SavedGraphDocument(),
        created_at=created_at,
        updated_at=created_at,
    )

    graph.replace(
        name="  Renamed draft  ",
        document=replacement,
        expected_revision=1,
        updated_at=updated_at,
    )

    assert graph.id == graph_id
    assert graph.name == "Renamed draft"
    assert graph.document == replacement
    assert graph.revision == 2
    assert graph.created_at == created_at
    assert graph.updated_at == updated_at


def test_saved_graph_replace_rejects_stale_revision_without_mutating_graph() -> None:
    graph = SavedGraph(name="Original", document=SavedGraphDocument())
    original_updated_at = graph.updated_at

    with pytest.raises(SavedGraphRevisionConflictError) as raised:
        graph.replace(
            name="Changed",
            document=SavedGraphDocument(nodes=(_node("new-node"),)),
            expected_revision=2,
        )

    assert raised.value.graph_id == graph.id
    assert raised.value.expected_revision == 2
    assert raised.value.actual_revision == 1
    assert graph.name == "Original"
    assert graph.document == SavedGraphDocument()
    assert graph.revision == 1
    assert graph.updated_at == original_updated_at


def test_saved_graph_replace_preserves_timezone_aware_timestamp_invariant() -> None:
    graph = SavedGraph(name="Draft", document=SavedGraphDocument())

    with pytest.raises(ValueError, match="timezone-aware"):
        graph.replace(
            name="Draft",
            document=SavedGraphDocument(),
            expected_revision=1,
            updated_at=datetime(2026, 7, 14, 10, 0),
        )
