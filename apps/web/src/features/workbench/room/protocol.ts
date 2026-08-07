import type {
  CollaborativeHead,
  SubmitGraphCommandRequest,
  WorkspaceCapability,
} from "@/lib/api";

export const ROOM_PROTOCOL_VERSION = 1;
export const ROOM_COMMAND_QUEUE_CAP = 32;

export const CLOSE_PERMISSIONS_CHANGED = 4003;
export const CLOSE_ACCESS_REVOKED = 4004;
export const CLOSE_PROTOCOL_ERROR = 4008;
export const CLOSE_SLOW_CONSUMER = 4009;
export const CLOSE_GRAPH_DELETED = 4010;

export type RoomGraphCommand = SubmitGraphCommandRequest["command"];

export type GraphRoomTerminalReason =
  | "permissions_changed"
  | "access_revoked"
  | "graph_deleted"
  | "protocol_error"
  | "slow_consumer";

export type GraphRoomStatus =
  | "idle"
  | "connecting"
  | "ready"
  | "reconnecting"
  | "unsynchronized"
  | "stopped";

export interface ActorPresentation {
  readonly actor_id: string;
  readonly display_name: string;
  readonly color: string;
}

export interface CapabilitySnapshot {
  readonly capabilities: readonly WorkspaceCapability[];
  readonly authorization_version: number;
}

export interface RoomReadyMessage {
  readonly protocol_version: 1;
  readonly type: "room.ready";
  readonly workspace_id: string;
  readonly graph_id: string;
  readonly graph_room_session_id: string;
  readonly actor: ActorPresentation;
  readonly capabilities: CapabilitySnapshot;
  readonly head: CollaborativeHead;
  readonly participants: readonly ActorPresentation[];
  readonly active_execution: null;
  readonly registry_marker: string;
}

export interface GraphCommandSubmitMessage {
  readonly protocol_version: 1;
  readonly type: "graph.command.submit";
  readonly command_id: string;
  readonly room_epoch: string;
  readonly observed_sequence: number;
  readonly command: RoomGraphCommand;
}

export interface GraphCommandAcceptedMessage {
  readonly protocol_version: 1;
  readonly type: "graph.command.accepted";
  readonly command_id: string;
  readonly room_epoch: string;
  readonly sequence: number;
  readonly actor: ActorPresentation;
  readonly graph_room_session_id: string | null;
  readonly command: RoomGraphCommand;
}

export interface GraphCommandReceiptMessage {
  readonly protocol_version: 1;
  readonly type: "graph.command.receipt";
  readonly command_id: string;
  readonly outcome: "accepted" | "idempotent_replay";
  readonly accepted_room_epoch: string;
  readonly accepted_sequence: number;
  readonly current_room_epoch: string;
  readonly current_sequence: number;
  readonly deduplicated: boolean;
  readonly requires_head_rehydration: boolean;
}

export interface GraphCommandRejectedMessage {
  readonly protocol_version: 1;
  readonly type: "graph.command.rejected";
  readonly command_id: string;
  readonly error_code: string;
  readonly detail: string;
  readonly current_room_epoch: string | null;
  readonly current_sequence: number | null;
}

export interface RoomRehydrateMessage {
  readonly protocol_version: 1;
  readonly type: "room.rehydrate";
  readonly reason: "epoch_reset";
  readonly head: CollaborativeHead;
}

export type ServerRoomMessage =
  | RoomReadyMessage
  | GraphCommandAcceptedMessage
  | GraphCommandReceiptMessage
  | GraphCommandRejectedMessage
  | RoomRehydrateMessage;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isNonNegativeInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function isPositiveInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1;
}

function parseActor(value: unknown): ActorPresentation | null {
  if (!isRecord(value)) return null;
  if (!isString(value.actor_id) || !isString(value.display_name) || !isString(value.color)) {
    return null;
  }
  return {
    actor_id: value.actor_id,
    display_name: value.display_name,
    color: value.color,
  };
}

function parseCapabilities(value: unknown): CapabilitySnapshot | null {
  if (!isRecord(value)) return null;
  if (!Array.isArray(value.capabilities) || !isPositiveInt(value.authorization_version)) {
    return null;
  }
  if (!value.capabilities.every((item) => isString(item))) return null;
  return {
    capabilities: value.capabilities as WorkspaceCapability[],
    authorization_version: value.authorization_version,
  };
}

function parseHead(value: unknown): CollaborativeHead | null {
  if (!isRecord(value)) return null;
  if (
    !isString(value.graph_id) ||
    !isString(value.room_epoch) ||
    !isNonNegativeInt(value.collaboration_sequence) ||
    !isNonNegativeInt(value.checkpoint_sequence) ||
    !isPositiveInt(value.checkpoint_revision) ||
    !isString(value.name) ||
    !isString(value.updated_at) ||
    !Array.isArray(value.nodes) ||
    !Array.isArray(value.edges)
  ) {
    return null;
  }
  return value as CollaborativeHead;
}

export function parseServerRoomMessage(raw: unknown): ServerRoomMessage | null {
  if (!isRecord(raw) || raw.protocol_version !== ROOM_PROTOCOL_VERSION) {
    return null;
  }
  const type = raw.type;
  if (type === "room.ready") {
    const actor = parseActor(raw.actor);
    const capabilities = parseCapabilities(raw.capabilities);
    const head = parseHead(raw.head);
    if (
      !actor ||
      !capabilities ||
      !head ||
      !isString(raw.workspace_id) ||
      !isString(raw.graph_id) ||
      !isString(raw.graph_room_session_id) ||
      !isString(raw.registry_marker)
    ) {
      return null;
    }
    const participants = Array.isArray(raw.participants)
      ? raw.participants.map(parseActor)
      : [];
    if (participants.some((item) => item === null)) return null;
    return {
      protocol_version: 1,
      type: "room.ready",
      workspace_id: raw.workspace_id,
      graph_id: raw.graph_id,
      graph_room_session_id: raw.graph_room_session_id,
      actor,
      capabilities,
      head,
      participants: participants as ActorPresentation[],
      active_execution: null,
      registry_marker: raw.registry_marker,
    };
  }
  if (type === "graph.command.accepted") {
    const actor = parseActor(raw.actor);
    if (
      !actor ||
      !isString(raw.command_id) ||
      !isString(raw.room_epoch) ||
      !isNonNegativeInt(raw.sequence) ||
      !isRecord(raw.command)
    ) {
      return null;
    }
    const sessionId = raw.graph_room_session_id;
    if (sessionId !== null && sessionId !== undefined && !isString(sessionId)) {
      return null;
    }
    return {
      protocol_version: 1,
      type: "graph.command.accepted",
      command_id: raw.command_id,
      room_epoch: raw.room_epoch,
      sequence: raw.sequence,
      actor,
      graph_room_session_id: isString(sessionId) ? sessionId : null,
      command: raw.command as RoomGraphCommand,
    };
  }
  if (type === "graph.command.receipt") {
    if (
      !isString(raw.command_id) ||
      (raw.outcome !== "accepted" && raw.outcome !== "idempotent_replay") ||
      !isString(raw.accepted_room_epoch) ||
      !isNonNegativeInt(raw.accepted_sequence) ||
      !isString(raw.current_room_epoch) ||
      !isNonNegativeInt(raw.current_sequence) ||
      typeof raw.deduplicated !== "boolean"
    ) {
      return null;
    }
    return {
      protocol_version: 1,
      type: "graph.command.receipt",
      command_id: raw.command_id,
      outcome: raw.outcome,
      accepted_room_epoch: raw.accepted_room_epoch,
      accepted_sequence: raw.accepted_sequence,
      current_room_epoch: raw.current_room_epoch,
      current_sequence: raw.current_sequence,
      deduplicated: raw.deduplicated,
      requires_head_rehydration: raw.requires_head_rehydration === true,
    };
  }
  if (type === "graph.command.rejected") {
    if (
      !isString(raw.command_id) ||
      !isString(raw.error_code) ||
      !isString(raw.detail)
    ) {
      return null;
    }
    const epoch = raw.current_room_epoch;
    const sequence = raw.current_sequence;
    if (epoch !== null && epoch !== undefined && !isString(epoch)) return null;
    if (sequence !== null && sequence !== undefined && !isNonNegativeInt(sequence)) {
      return null;
    }
    return {
      protocol_version: 1,
      type: "graph.command.rejected",
      command_id: raw.command_id,
      error_code: raw.error_code,
      detail: raw.detail,
      current_room_epoch: isString(epoch) ? epoch : null,
      current_sequence: isNonNegativeInt(sequence) ? sequence : null,
    };
  }
  if (type === "room.rehydrate") {
    const head = parseHead(raw.head);
    if (!head || raw.reason !== "epoch_reset") return null;
    return {
      protocol_version: 1,
      type: "room.rehydrate",
      reason: "epoch_reset",
      head,
    };
  }
  return null;
}

export function terminalReasonFromClose(
  code: number,
  reason: string,
): GraphRoomTerminalReason | null {
  const normalized = reason.trim();
  if (
    code === CLOSE_PERMISSIONS_CHANGED ||
    normalized === "permissions_changed"
  ) {
    return "permissions_changed";
  }
  if (code === CLOSE_ACCESS_REVOKED || normalized === "access_revoked") {
    return "access_revoked";
  }
  if (code === CLOSE_GRAPH_DELETED || normalized === "graph_deleted") {
    return "graph_deleted";
  }
  if (code === CLOSE_PROTOCOL_ERROR || normalized === "protocol_error") {
    return "protocol_error";
  }
  if (code === CLOSE_SLOW_CONSUMER || normalized === "slow_consumer") {
    return "slow_consumer";
  }
  return null;
}
