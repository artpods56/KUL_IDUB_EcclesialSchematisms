from uuid import UUID, uuid4

import pytest

from notarius_core.artifacts import ArtifactTypeKey
from notarius_core.domain.collaboration import (
    AddEdgeCommand,
    AddNodeCommand,
    ClearNodeArtifactTypeBindingCommand,
    CollaborativeGraphHead,
    DuplicateNodeCommand,
    MoveNodePosition,
    MoveNodesCommand,
    RemoveEdgesCommand,
    RemoveNodesCommand,
    RenameGraphCommand,
    ReplaceDocumentCommand,
    SetNodeArtifactTypeBindingCommand,
    SetNodeInputPlugsCommand,
    UpdateEdgeCommand,
    UpdateNodeConfigurationCommand,
    UpdateNodeLayoutCommand,
    apply_graph_command,
    command_hmac_digest,
    empty_collaborative_document,
    sanitize_document_for_cross_workspace_copy,
)
from notarius_core.domain.errors import CollaborationCommandRejectedError
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraphArtifactTypeBinding,
    SavedGraphDocument,
    SavedGraphEdge,
    SavedGraphInputPlug,
    SavedGraphNode,
    SavedGraphNodeLayout,
)


def _node(
    node_id: str = "n1",
    *,
    x: float = 0,
    y: float = 0,
    config: dict[str, object] | None = None,
) -> SavedGraphNode:
    return SavedGraphNode(
        id=node_id,
        operator_id="example.operator",
        operator_version=1,
        position=GraphPoint(x=x, y=y),
        config=config or {},
    )


def test_apply_rename_and_add_node() -> None:
    document = empty_collaborative_document()
    name, document = apply_graph_command(
        name="Untitled graph",
        document=document,
        command=RenameGraphCommand(name="Named", expected_name="Untitled graph"),
    )
    assert name == "Named"
    name, document = apply_graph_command(
        name=name,
        document=document,
        command=AddNodeCommand(node=_node()),
    )
    assert name == "Named"
    assert document.nodes[0].id == "n1"


def test_apply_rejects_duplicate_node() -> None:
    node = _node()
    document = SavedGraphDocument(nodes=(node,))
    with pytest.raises(CollaborationCommandRejectedError) as exc:
        apply_graph_command(
            name="Graph",
            document=document,
            command=AddNodeCommand(node=node),
        )
    assert exc.value.error_code == "duplicate_node"


def test_rename_field_conflict() -> None:
    with pytest.raises(CollaborationCommandRejectedError) as exc:
        apply_graph_command(
            name="Current",
            document=SavedGraphDocument(),
            command=RenameGraphCommand(name="Next", expected_name="Stale"),
        )
    assert exc.value.error_code == "field_conflict"


def test_move_remove_update_edge_and_plugs() -> None:
    source = _node("source", x=1, y=2)
    target = SavedGraphNode(
        id="target",
        operator_id="example.operator",
        operator_version=1,
        position=GraphPoint(x=10, y=20),
        input_plugs=(SavedGraphInputPlug(id="plug-a", port="value"),),
        artifact_type_bindings=(
            SavedGraphArtifactTypeBinding(
                variable="T",
                artifact_type=ArtifactTypeKey(id="image.raster", schema_version=1),
            ),
        ),
    )
    edge = SavedGraphEdge(
        id="e1",
        from_node="source",
        from_port="result",
        to_node="target",
        to_port="value",
        to_plug="plug-a",
        enabled=True,
    )
    document = SavedGraphDocument(nodes=(source, target), edges=(edge,))

    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=MoveNodesCommand(
            positions=(MoveNodePosition(node_id="source", x=5, y=6),)
        ),
    )
    assert document.nodes[0].position == GraphPoint(x=5, y=6)

    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=UpdateNodeConfigurationCommand(
            node_id="source",
            field="text",
            value="hello",
            expected_value=None,
        ),
    )
    assert document.nodes[0].config_dict()["text"] == "hello"

    layout = SavedGraphNodeLayout(width=400, body_height=200)
    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=UpdateNodeLayoutCommand(
            node_id="source",
            layout=layout,
            expected_layout=None,
        ),
    )
    assert document.nodes[0].layout == layout

    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=SetNodeInputPlugsCommand(
            node_id="target",
            input_plugs=(
                SavedGraphInputPlug(id="plug-a", port="value"),
                SavedGraphInputPlug(id="plug-b", port="value"),
            ),
            expected_plug_ids=("plug-a",),
        ),
    )
    assert [plug.id for plug in document.nodes[1].input_plugs] == ["plug-a", "plug-b"]

    binding = SavedGraphArtifactTypeBinding(
        variable="T",
        artifact_type=ArtifactTypeKey(id="text.value", schema_version=1),
    )
    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=SetNodeArtifactTypeBindingCommand(
            node_id="target",
            binding=binding,
            expected_binding=SavedGraphArtifactTypeBinding(
                variable="T",
                artifact_type=ArtifactTypeKey(id="image.raster", schema_version=1),
            ),
        ),
    )
    assert document.nodes[1].artifact_type_bindings[0] == binding

    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=ClearNodeArtifactTypeBindingCommand(
            node_id="target",
            variable="T",
            expected_binding=binding,
        ),
    )
    assert document.nodes[1].artifact_type_bindings == ()

    updated_edge = edge.model_copy(update={"enabled": False})
    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=UpdateEdgeCommand(edge=updated_edge, expected_edge=edge),
    )
    assert document.edges[0].enabled is False

    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=DuplicateNodeCommand(
            source_node_id="source",
            node=_node("source-copy", x=40, y=40),
        ),
    )
    assert any(node.id == "source-copy" for node in document.nodes)

    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=RemoveEdgesCommand(edge_ids=("e1", "missing")),
    )
    assert document.edges == ()

    _, document = apply_graph_command(
        name="Graph",
        document=document,
        command=RemoveNodesCommand(node_ids=("source", "missing")),
    )
    assert [node.id for node in document.nodes] == ["target", "source-copy"]


def test_add_edge_and_sanitize_copy_document() -> None:
    node = _node(
        config={
            "uploads": [{"upload_key": "abc", "filename": "a.png", "byte_size": 1}],
            "nested": {"artifact_id": str(uuid4()), "keep": True},
        }
    )
    document = SavedGraphDocument(nodes=(node,))
    edge = SavedGraphEdge(
        id="e1",
        from_node="n1",
        from_port="result",
        to_node="n1",
        to_port="value",
    )
    _, with_edge = apply_graph_command(
        name="Graph",
        document=document,
        command=AddEdgeCommand(edge=edge),
    )
    assert with_edge.edges[0].id == "e1"

    sanitized = sanitize_document_for_cross_workspace_copy(document)
    assert "uploads" not in sanitized.nodes[0].config_dict()
    assert sanitized.nodes[0].config_dict()["nested"] == {"keep": True}

    module_document = SavedGraphDocument(
        nodes=(
            SavedGraphNode(
                id="mod",
                operator_id="graph.module." + str(uuid4()),
                operator_version=1,
                position=GraphPoint(x=0, y=0),
            ),
        )
    )
    with pytest.raises(CollaborationCommandRejectedError) as exc:
        sanitize_document_for_cross_workspace_copy(module_document)
    assert exc.value.error_code == "foreign_module_reference"


def test_command_hmac_digest_is_stable_and_keyed() -> None:
    command = ReplaceDocumentCommand(name="A", document=SavedGraphDocument())
    workspace_id = UUID("00000000-0000-0000-0000-000000000001")
    graph_id = UUID("00000000-0000-0000-0000-000000000002")
    actor_id = UUID("00000000-0000-0000-0000-000000000003")
    room_epoch = uuid4()
    first = command_hmac_digest(
        b"key-a",
        key_version=1,
        workspace_id=workspace_id,
        graph_id=graph_id,
        actor_user_id=actor_id,
        room_epoch=room_epoch,
        observed_sequence=0,
        command=command,
    )
    second = command_hmac_digest(
        b"key-a",
        key_version=1,
        workspace_id=workspace_id,
        graph_id=graph_id,
        actor_user_id=actor_id,
        room_epoch=room_epoch,
        observed_sequence=0,
        command=command,
    )
    other_key = command_hmac_digest(
        b"key-b",
        key_version=1,
        workspace_id=workspace_id,
        graph_id=graph_id,
        actor_user_id=actor_id,
        room_epoch=room_epoch,
        observed_sequence=0,
        command=command,
    )
    assert first == second
    assert first != other_key


def test_existing_graph_head_starts_at_sequence_zero() -> None:
    head = CollaborativeGraphHead.for_existing_saved_graph(
        workspace_id=uuid4(),
        graph_id=uuid4(),
        name="Legacy",
        document=SavedGraphDocument(),
        checkpoint_revision=4,
    )
    assert head.collaboration_sequence == 0
    assert head.checkpoint_sequence == 0
    assert head.checkpoint_revision == 4
    assert head.is_fully_checkpointed
