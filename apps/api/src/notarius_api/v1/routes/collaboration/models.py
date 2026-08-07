from typing import Annotated, ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from notarius_core.domain.collaboration import GraphCommand
from notarius_core.domain.identity import WorkspaceCapability

from notarius_api.v1.routes.saved_graphs.models import CollaborativeHeadResponse


PROTOCOL_VERSION = 1

ACTOR_DISPLAY_COLORS = (
    "indigo",
    "emerald",
    "amber",
    "rose",
    "sky",
    "violet",
    "teal",
    "orange",
)


class RoomProtocolModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class ActorPresentation(RoomProtocolModel):
    actor_id: UUID
    display_name: str = Field(min_length=1, max_length=160)
    color: str = Field(min_length=1, max_length=32)


class CapabilitySnapshot(RoomProtocolModel):
    capabilities: tuple[WorkspaceCapability, ...]
    authorization_version: int = Field(ge=1)


class RoomReadyMessage(RoomProtocolModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    type: Literal["room.ready"] = "room.ready"
    workspace_id: UUID
    graph_id: UUID
    graph_room_session_id: UUID
    actor: ActorPresentation
    capabilities: CapabilitySnapshot
    head: CollaborativeHeadResponse
    participants: list[ActorPresentation] = Field(default_factory=list)
    active_execution: None = None
    registry_marker: str = "builtin"


class GraphCommandSubmitMessage(RoomProtocolModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    type: Literal["graph.command.submit"] = "graph.command.submit"
    command_id: UUID
    room_epoch: UUID
    observed_sequence: int = Field(ge=0)
    command: GraphCommand


class GraphCommandAcceptedMessage(RoomProtocolModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    type: Literal["graph.command.accepted"] = "graph.command.accepted"
    command_id: UUID
    room_epoch: UUID
    sequence: int = Field(ge=0)
    actor: ActorPresentation
    graph_room_session_id: UUID | None = None
    command: GraphCommand


class GraphCommandReceiptMessage(RoomProtocolModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    type: Literal["graph.command.receipt"] = "graph.command.receipt"
    command_id: UUID
    outcome: Literal["accepted", "idempotent_replay"]
    accepted_room_epoch: UUID
    accepted_sequence: int = Field(ge=0)
    current_room_epoch: UUID
    current_sequence: int = Field(ge=0)
    deduplicated: bool
    requires_head_rehydration: bool = False


class GraphCommandRejectedMessage(RoomProtocolModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    type: Literal["graph.command.rejected"] = "graph.command.rejected"
    command_id: UUID
    error_code: str = Field(min_length=1, max_length=64)
    detail: str = Field(min_length=1, max_length=500)
    current_room_epoch: UUID | None = None
    current_sequence: int | None = Field(default=None, ge=0)


class RoomRehydrateMessage(RoomProtocolModel):
    protocol_version: Literal[1] = PROTOCOL_VERSION
    type: Literal["room.rehydrate"] = "room.rehydrate"
    reason: Literal["epoch_reset"] = "epoch_reset"
    head: CollaborativeHeadResponse


ClientRoomMessage = Annotated[
    GraphCommandSubmitMessage,
    Field(discriminator="type"),
]

CLIENT_ROOM_MESSAGE_ADAPTER: TypeAdapter[GraphCommandSubmitMessage] = TypeAdapter(
    ClientRoomMessage
)


def actor_display_color(user_id: UUID) -> str:
    return ACTOR_DISPLAY_COLORS[user_id.int % len(ACTOR_DISPLAY_COLORS)]


def bounded_display_name(display_name: str | None, email: str) -> str:
    candidate = (display_name or "").strip()
    if candidate == "":
        local, separator, _domain = email.partition("@")
        candidate = local if separator else email
    candidate = candidate.strip() or "collaborator"
    return candidate[:160]


def command_receipt_outcome(
    *,
    deduplicated: bool,
) -> Literal["accepted", "idempotent_replay"]:
    if deduplicated:
        return "idempotent_replay"
    return "accepted"
