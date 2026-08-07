"""Collaborative graph head, commands, receipts, and checkpoints."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import hmac
import json
from typing import Annotated, ClassVar, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
)

from notarius_core.domain.errors import CollaborationCommandRejectedError
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraphDocument,
    SavedGraphEdge,
    SavedGraphNode,
    SavedGraphNodeLayout,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CollaborationActorKind(StrEnum):
    USER = "user"
    SYSTEM = "system"


class CommandReceiptOutcome(StrEnum):
    ACCEPTED = "accepted"
    IDEMPOTENT_REPLAY = "idempotent_replay"


class GraphCommandKind(StrEnum):
    RENAME_GRAPH = "rename_graph"
    ADD_NODE = "add_node"
    REMOVE_NODES = "remove_nodes"
    MOVE_NODES = "move_nodes"
    UPDATE_NODE_CONFIGURATION = "update_node_configuration"
    UPDATE_NODE_LAYOUT = "update_node_layout"
    ADD_EDGE = "add_edge"
    REMOVE_EDGES = "remove_edges"
    REPLACE_DOCUMENT = "replace_document"


class CollaborationValue(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class RenameGraphCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.RENAME_GRAPH] = GraphCommandKind.RENAME_GRAPH
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = value.strip()
        if name == "":
            raise ValueError("Graph name must not be blank")
        if len(name) > 160:
            raise ValueError("Graph name must be at most 160 characters")
        return name


class AddNodeCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.ADD_NODE] = GraphCommandKind.ADD_NODE
    node: SavedGraphNode


class RemoveNodesCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.REMOVE_NODES] = GraphCommandKind.REMOVE_NODES
    node_ids: tuple[str, ...] = Field(min_length=1)


class MoveNodePosition(CollaborationValue):
    node_id: str
    x: float
    y: float


class MoveNodesCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.MOVE_NODES] = GraphCommandKind.MOVE_NODES
    positions: tuple[MoveNodePosition, ...] = Field(min_length=1)


class UpdateNodeConfigurationCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.UPDATE_NODE_CONFIGURATION] = (
        GraphCommandKind.UPDATE_NODE_CONFIGURATION
    )
    node_id: str
    field: str
    value: object


class UpdateNodeLayoutCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.UPDATE_NODE_LAYOUT] = (
        GraphCommandKind.UPDATE_NODE_LAYOUT
    )
    node_id: str
    layout: SavedGraphNodeLayout | None


class AddEdgeCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.ADD_EDGE] = GraphCommandKind.ADD_EDGE
    edge: SavedGraphEdge


class RemoveEdgesCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.REMOVE_EDGES] = GraphCommandKind.REMOVE_EDGES
    edge_ids: tuple[str, ...] = Field(min_length=1)


class ReplaceDocumentCommand(CollaborationValue):
    kind: Literal[GraphCommandKind.REPLACE_DOCUMENT] = GraphCommandKind.REPLACE_DOCUMENT
    name: str
    document: SavedGraphDocument

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = value.strip()
        if name == "":
            raise ValueError("Graph name must not be blank")
        if len(name) > 160:
            raise ValueError("Graph name must be at most 160 characters")
        return name


GraphCommand = Annotated[
    RenameGraphCommand
    | AddNodeCommand
    | RemoveNodesCommand
    | MoveNodesCommand
    | UpdateNodeConfigurationCommand
    | UpdateNodeLayoutCommand
    | AddEdgeCommand
    | RemoveEdgesCommand
    | ReplaceDocumentCommand,
    Field(discriminator="kind"),
]

GRAPH_COMMAND_ADAPTER: TypeAdapter[GraphCommand] = TypeAdapter(GraphCommand)


def empty_collaborative_document() -> SavedGraphDocument:
    return SavedGraphDocument()


def canonical_command_payload(command: GraphCommand) -> bytes:
    dumped = command.model_dump(mode="json")
    return json.dumps(dumped, separators=(",", ":"), sort_keys=True).encode("utf-8")


def command_hmac_digest(
    key: bytes,
    *,
    key_version: int,
    workspace_id: UUID,
    graph_id: UUID,
    actor_user_id: UUID | None,
    room_epoch: UUID,
    observed_sequence: int,
    command: GraphCommand,
) -> bytes:
    del key_version
    envelope = {
        "actor_user_id": None if actor_user_id is None else str(actor_user_id),
        "command": command.model_dump(mode="json"),
        "graph_id": str(graph_id),
        "observed_sequence": observed_sequence,
        "room_epoch": str(room_epoch),
        "workspace_id": str(workspace_id),
    }
    payload = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hmac.new(key, payload, sha256).digest()


def _node_or_raise(document: SavedGraphDocument, node_id: str) -> SavedGraphNode:
    for node in document.nodes:
        if node.id == node_id:
            return node
    raise CollaborationCommandRejectedError(
        code="missing_node",
        message=f"Graph command targets missing node {node_id}",
    )


def apply_graph_command(
    *,
    name: str,
    document: SavedGraphDocument,
    command: GraphCommand,
) -> tuple[str, SavedGraphDocument]:
    if isinstance(command, RenameGraphCommand):
        return command.name, document

    if isinstance(command, AddNodeCommand):
        if any(node.id == command.node.id for node in document.nodes):
            raise CollaborationCommandRejectedError(
                code="duplicate_node",
                message=f"Graph command adds duplicate node {command.node.id}",
            )
        return name, SavedGraphDocument(
            nodes=(*document.nodes, command.node),
            edges=document.edges,
        )

    if isinstance(command, RemoveNodesCommand):
        removed = set(command.node_ids)
        return name, SavedGraphDocument(
            nodes=tuple(node for node in document.nodes if node.id not in removed),
            edges=tuple(
                edge
                for edge in document.edges
                if edge.from_node not in removed and edge.to_node not in removed
            ),
        )

    if isinstance(command, MoveNodesCommand):
        for position in command.positions:
            _node_or_raise(document, position.node_id)
        positions = {
            position.node_id: GraphPoint(x=position.x, y=position.y)
            for position in command.positions
        }
        return name, SavedGraphDocument(
            nodes=tuple(
                node.model_copy(update={"position": positions[node.id]})
                if node.id in positions
                else node
                for node in document.nodes
            ),
            edges=document.edges,
        )

    if isinstance(command, UpdateNodeConfigurationCommand):
        _node_or_raise(document, command.node_id)
        updated_nodes: list[SavedGraphNode] = []
        for node in document.nodes:
            if node.id != command.node_id:
                updated_nodes.append(node)
                continue
            config = dict(node.config_dict())
            config[command.field] = command.value
            updated_nodes.append(node.model_copy(update={"config": config}))
        return name, SavedGraphDocument(nodes=tuple(updated_nodes), edges=document.edges)

    if isinstance(command, UpdateNodeLayoutCommand):
        _node_or_raise(document, command.node_id)
        return name, SavedGraphDocument(
            nodes=tuple(
                node.model_copy(update={"layout": command.layout})
                if node.id == command.node_id
                else node
                for node in document.nodes
            ),
            edges=document.edges,
        )

    if isinstance(command, AddEdgeCommand):
        if any(edge.id == command.edge.id for edge in document.edges):
            raise CollaborationCommandRejectedError(
                code="duplicate_edge",
                message=f"Graph command adds duplicate edge {command.edge.id}",
            )
        return name, SavedGraphDocument(
            nodes=document.nodes,
            edges=(*document.edges, command.edge),
        )

    if isinstance(command, RemoveEdgesCommand):
        removed_edges = set(command.edge_ids)
        return name, SavedGraphDocument(
            nodes=document.nodes,
            edges=tuple(
                edge for edge in document.edges if edge.id not in removed_edges
            ),
        )

    if isinstance(command, ReplaceDocumentCommand):
        return command.name, command.document

    raise CollaborationCommandRejectedError(
        code="unsupported_command",
        message=f"Unsupported graph command kind {command.kind!r}",
    )


@dataclass
class CollaborativeGraphHead:
    workspace_id: UUID
    graph_id: UUID
    room_epoch: UUID
    collaboration_sequence: int
    checkpoint_sequence: int
    checkpoint_revision: int
    name: str
    document: SavedGraphDocument
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.collaboration_sequence < 0:
            raise ValueError("Collaboration sequence must be non-negative")
        if self.checkpoint_sequence < 0:
            raise ValueError("Checkpoint sequence must be non-negative")
        if self.checkpoint_sequence > self.collaboration_sequence:
            raise ValueError(
                "Checkpoint sequence cannot exceed collaboration sequence"
            )
        if self.checkpoint_revision < 1:
            raise ValueError("Checkpoint revision must be at least 1")
        self.name = self.name.strip()
        if self.name == "":
            raise ValueError("Collaborative head name must not be blank")
        if len(self.name) > 160:
            raise ValueError("Collaborative head name must be at most 160 characters")
        if self.updated_at.tzinfo is None:
            raise ValueError("Collaborative head timestamp must be timezone-aware")

    @property
    def is_fully_checkpointed(self) -> bool:
        return self.checkpoint_sequence == self.collaboration_sequence

    def apply_accepted_command(
        self,
        *,
        name: str,
        document: SavedGraphDocument,
        updated_at: datetime | None = None,
    ) -> None:
        stamp = updated_at or _utc_now()
        if stamp.tzinfo is None:
            raise ValueError("Collaborative head timestamp must be timezone-aware")
        validated_name = name.strip()
        if validated_name == "":
            raise ValueError("Collaborative head name must not be blank")
        self.name = validated_name
        self.document = document
        self.collaboration_sequence += 1
        self.updated_at = stamp

    def record_checkpoint(
        self,
        *,
        sequence: int,
        revision: int,
        updated_at: datetime | None = None,
    ) -> None:
        if sequence != self.collaboration_sequence:
            raise ValueError(
                "Checkpoint sequence must equal the current collaboration sequence"
            )
        if revision < 1:
            raise ValueError("Checkpoint revision must be at least 1")
        stamp = updated_at or _utc_now()
        if stamp.tzinfo is None:
            raise ValueError("Collaborative head timestamp must be timezone-aware")
        self.checkpoint_sequence = sequence
        self.checkpoint_revision = revision
        self.updated_at = stamp

    @classmethod
    def for_existing_saved_graph(
        cls,
        *,
        workspace_id: UUID,
        graph_id: UUID,
        name: str,
        document: SavedGraphDocument,
        checkpoint_revision: int,
        room_epoch: UUID | None = None,
        updated_at: datetime | None = None,
    ) -> Self:
        return cls(
            workspace_id=workspace_id,
            graph_id=graph_id,
            room_epoch=uuid4() if room_epoch is None else room_epoch,
            collaboration_sequence=0,
            checkpoint_sequence=0,
            checkpoint_revision=checkpoint_revision,
            name=name,
            document=document,
            updated_at=_utc_now() if updated_at is None else updated_at,
        )


class GraphCommandJournalEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    workspace_id: UUID
    graph_id: UUID
    room_epoch: UUID
    command_id: UUID
    command_hmac: bytes
    hmac_key_version: int
    accepted_sequence: int
    actor_kind: CollaborationActorKind
    actor_user_id: UUID | None
    graph_room_session_id: UUID | None
    authorization_version: int | None
    command_kind: GraphCommandKind
    command_payload: dict[str, object]
    accepted_at: datetime = Field(default_factory=_utc_now)

    @field_validator("accepted_at")
    @classmethod
    def require_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Journal acceptance timestamp must be timezone-aware")
        return value


class GraphCommandReceipt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    workspace_id: UUID
    graph_id: UUID
    command_id: UUID
    command_hmac: bytes
    hmac_key_version: int
    actor_kind: CollaborationActorKind
    actor_user_id: UUID | None
    room_epoch: UUID
    accepted_sequence: int
    outcome: CommandReceiptOutcome
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at")
    @classmethod
    def require_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Receipt timestamp must be timezone-aware")
        return value


class GraphCheckpointMapping(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    workspace_id: UUID
    graph_id: UUID
    room_epoch: UUID
    collaboration_sequence: int
    saved_revision: int
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at")
    @classmethod
    def require_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Checkpoint mapping timestamp must be timezone-aware")
        return value


class GraphExecutionIdempotencyRecord(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    workspace_id: UUID
    graph_id: UUID
    client_request_id: UUID
    request_hmac: bytes
    hmac_key_version: int
    actor_user_id: UUID
    room_epoch: UUID
    head_sequence: int
    execution_id: UUID
    created_at: datetime = Field(default_factory=_utc_now)


class GraphActiveExecutionSlot(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    workspace_id: UUID
    graph_id: UUID
    execution_id: UUID
    updated_at: datetime = Field(default_factory=_utc_now)


__all__ = [
    "AddEdgeCommand",
    "AddNodeCommand",
    "CollaborationActorKind",
    "CollaborativeGraphHead",
    "CommandReceiptOutcome",
    "GRAPH_COMMAND_ADAPTER",
    "GraphActiveExecutionSlot",
    "GraphCheckpointMapping",
    "GraphCommand",
    "GraphCommandJournalEntry",
    "GraphCommandKind",
    "GraphCommandReceipt",
    "GraphExecutionIdempotencyRecord",
    "MoveNodePosition",
    "MoveNodesCommand",
    "RemoveEdgesCommand",
    "RemoveNodesCommand",
    "RenameGraphCommand",
    "ReplaceDocumentCommand",
    "UpdateNodeConfigurationCommand",
    "UpdateNodeLayoutCommand",
    "apply_graph_command",
    "canonical_command_payload",
    "command_hmac_digest",
    "empty_collaborative_document",
]
