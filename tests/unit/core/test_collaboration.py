from uuid import UUID, uuid4

import pytest

from notarius_core.domain.collaboration import (
    AddNodeCommand,
    CollaborativeGraphHead,
    RenameGraphCommand,
    ReplaceDocumentCommand,
    apply_graph_command,
    command_hmac_digest,
    empty_collaborative_document,
)
from notarius_core.domain.errors import CollaborationCommandRejectedError
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraphDocument,
    SavedGraphNode,
)


def test_apply_rename_and_add_node() -> None:
    document = empty_collaborative_document()
    name, document = apply_graph_command(
        name="Untitled graph",
        document=document,
        command=RenameGraphCommand(name="Named"),
    )
    assert name == "Named"
    node = SavedGraphNode(
        id="n1",
        operator_id="example.operator",
        operator_version=1,
        position=GraphPoint(x=0, y=0),
    )
    name, document = apply_graph_command(
        name=name,
        document=document,
        command=AddNodeCommand(node=node),
    )
    assert name == "Named"
    assert document.nodes[0].id == "n1"


def test_apply_rejects_duplicate_node() -> None:
    node = SavedGraphNode(
        id="n1",
        operator_id="example.operator",
        operator_version=1,
        position=GraphPoint(x=0, y=0),
    )
    document = SavedGraphDocument(nodes=(node,))
    with pytest.raises(CollaborationCommandRejectedError):
        apply_graph_command(
            name="Graph",
            document=document,
            command=AddNodeCommand(node=node),
        )


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
