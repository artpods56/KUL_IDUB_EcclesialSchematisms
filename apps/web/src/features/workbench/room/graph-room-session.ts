import type { CollaborativeHead, WorkspaceCapability } from "@/lib/api";

import { graphRoomWebSocketUrl } from "./graph-room-url";
import {
  ROOM_COMMAND_QUEUE_CAP,
  ROOM_PROTOCOL_VERSION,
  parseServerRoomMessage,
  terminalReasonFromClose,
  type ActorPresentation,
  type GraphCommandAcceptedMessage,
  type GraphCommandReceiptMessage,
  type GraphCommandRejectedMessage,
  type GraphRoomStatus,
  type GraphRoomTerminalReason,
  type PresenceParticipant,
  type PresenceUpdateSubmit,
  type RoomGraphCommand,
  type RoomReadyMessage,
} from "./protocol";

export type { GraphRoomStatus, GraphRoomTerminalReason, RoomGraphCommand };
export type { PresenceParticipant, PresenceUpdateSubmit };
export { ROOM_COMMAND_QUEUE_CAP, graphRoomWebSocketUrl };

export const PRESENCE_CLIENT_MIN_INTERVAL_MS = 50;

export class GraphRoomCommandError extends Error {
  readonly errorCode: string;
  readonly commandId: string;

  constructor(commandId: string, errorCode: string, detail: string) {
    super(detail);
    this.name = "GraphRoomCommandError";
    this.commandId = commandId;
    this.errorCode = errorCode;
  }
}

export interface GraphRoomCommandResult {
  readonly receipt: GraphCommandReceiptMessage;
  readonly accepted: GraphCommandAcceptedMessage | null;
}

export interface GraphRoomSessionListeners {
  onStatusChange?: (status: GraphRoomStatus) => void;
  onReady?: (ready: RoomReadyMessage) => void;
  onRehydrate?: (head: CollaborativeHead) => void;
  onCommandAccepted?: (message: GraphCommandAcceptedMessage) => void;
  onCommandRejected?: (message: GraphCommandRejectedMessage) => void;
  onPresenceChange?: (participants: readonly PresenceParticipant[]) => void;
  onTerminalClose?: (reason: GraphRoomTerminalReason) => void;
}

export interface GraphRoomSessionOptions extends GraphRoomSessionListeners {
  workspaceId: string;
  graphId: string;
  webSocketFactory?: (url: string) => WebSocket;
  reconnectDelayMs?: number;
  createCommandId?: () => string;
}

interface QueuedCommand {
  readonly commandId: string;
  readonly command: RoomGraphCommand;
  readonly resolve: (result: GraphRoomCommandResult) => void;
  readonly reject: (error: Error) => void;
  sent: boolean;
  accepted: GraphCommandAcceptedMessage | null;
}

export class GraphRoomSession {
  readonly workspaceId: string;
  readonly graphId: string;

  private readonly webSocketFactory: (url: string) => WebSocket;
  private readonly reconnectDelayMs: number;
  private readonly createCommandId: () => string;
  private readonly listeners: GraphRoomSessionListeners;

  private socket: WebSocket | null = null;
  private status: GraphRoomStatus = "idle";
  private terminalReason: GraphRoomTerminalReason | null = null;
  private intentionallyClosed = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private generation = 0;

  private ready: RoomReadyMessage | null = null;
  private head: CollaborativeHead | null = null;
  private capabilities: readonly WorkspaceCapability[] = [];
  private authorizationVersion: number | null = null;
  private participants = new Map<string, PresenceParticipant>();
  private presenceSequence = 0;
  private lastPresenceSentAt = 0;

  private readonly queue: QueuedCommand[] = [];
  private inFlight: QueuedCommand | null = null;

  constructor(options: GraphRoomSessionOptions) {
    this.workspaceId = options.workspaceId;
    this.graphId = options.graphId;
    this.webSocketFactory =
      options.webSocketFactory ?? ((url) => new WebSocket(url));
    this.reconnectDelayMs = options.reconnectDelayMs ?? 750;
    this.createCommandId = options.createCommandId ?? (() => crypto.randomUUID());
    this.listeners = {
      onStatusChange: options.onStatusChange,
      onReady: options.onReady,
      onRehydrate: options.onRehydrate,
      onCommandAccepted: options.onCommandAccepted,
      onCommandRejected: options.onCommandRejected,
      onPresenceChange: options.onPresenceChange,
      onTerminalClose: options.onTerminalClose,
    };
  }

  getStatus(): GraphRoomStatus {
    return this.status;
  }

  getTerminalReason(): GraphRoomTerminalReason | null {
    return this.terminalReason;
  }

  getHead(): CollaborativeHead | null {
    return this.head;
  }

  getCapabilities(): readonly WorkspaceCapability[] {
    return this.capabilities;
  }

  getAuthorizationVersion(): number | null {
    return this.authorizationVersion;
  }

  getActor(): ActorPresentation | null {
    return this.ready?.actor ?? null;
  }

  getGraphRoomSessionId(): string | null {
    return this.ready?.graph_room_session_id ?? null;
  }

  getParticipants(): readonly PresenceParticipant[] {
    return [...this.participants.values()];
  }

  getRemoteParticipants(): readonly PresenceParticipant[] {
    const localId = this.getGraphRoomSessionId();
    return this.getParticipants().filter(
      (participant) => participant.graph_room_session_id !== localId,
    );
  }

  canPublishPresence(): boolean {
    return (
      this.status === "ready" &&
      this.capabilities.includes("publish_presence") &&
      this.terminalReason === null
    );
  }

  publishPresence(update: Omit<PresenceUpdateSubmit, "presence_sequence">): boolean {
    if (!this.canPublishPresence()) return false;
    const now = Date.now();
    if (
      this.lastPresenceSentAt > 0 &&
      now - this.lastPresenceSentAt < PRESENCE_CLIENT_MIN_INTERVAL_MS
    ) {
      return false;
    }
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    this.presenceSequence += 1;
    const payload = {
      protocol_version: ROOM_PROTOCOL_VERSION,
      type: "presence.update" as const,
      presence_sequence: this.presenceSequence,
      cursor: update.cursor ?? null,
      selected_node_ids: update.selected_node_ids ?? [],
      selected_edge_ids: update.selected_edge_ids ?? [],
      activity: update.activity ?? null,
      activity_target_ids: update.activity_target_ids ?? [],
      transient_node_positions: update.transient_node_positions ?? [],
    };
    try {
      socket.send(JSON.stringify(payload));
      this.lastPresenceSentAt = now;
      return true;
    } catch {
      return false;
    }
  }

  canSubmitCommands(): boolean {
    return (
      this.status === "ready" &&
      this.head !== null &&
      this.capabilities.includes("edit_graph") &&
      this.terminalReason === null
    );
  }

  connect(): void {
    if (this.terminalReason !== null) return;
    this.intentionallyClosed = false;
    this.clearReconnectTimer();
    this.openSocket();
  }

  disconnect(): void {
    this.intentionallyClosed = true;
    this.clearReconnectTimer();
    this.generation += 1;
    const socket = this.socket;
    this.socket = null;
    if (socket && socket.readyState < WebSocket.CLOSING) {
      socket.close(1000, "client_disconnect");
    }
    this.rejectAllQueued(new GraphRoomCommandError(
      "",
      "disconnected",
      "Graph room disconnected before the command completed.",
    ));
    this.setStatus("idle");
    this.ready = null;
    this.clearPresence();
  }

  submitCommand(command: RoomGraphCommand): Promise<GraphRoomCommandResult> {
    if (!this.canSubmitCommands() || this.head === null) {
      return Promise.reject(
        new GraphRoomCommandError(
          "",
          "not_ready",
          "Graph room is not ready to accept durable commands.",
        ),
      );
    }
    if (this.queue.length + (this.inFlight ? 1 : 0) >= ROOM_COMMAND_QUEUE_CAP) {
      return Promise.reject(
        new GraphRoomCommandError(
          "",
          "queue_full",
          "Waiting to synchronize. Durable authoring is paused until the queue drains.",
        ),
      );
    }

    const commandId = this.createCommandId();
    return new Promise<GraphRoomCommandResult>((resolve, reject) => {
      this.queue.push({
        commandId,
        command,
        resolve,
        reject,
        sent: false,
        accepted: null,
      });
      this.drainQueue();
    });
  }

  private openSocket(): void {
    if (this.intentionallyClosed || this.terminalReason !== null) return;

    this.generation += 1;
    const generation = this.generation;
    const url = graphRoomWebSocketUrl(this.workspaceId, this.graphId);
    this.setStatus(this.head ? "reconnecting" : "connecting");

    let socket: WebSocket;
    try {
      socket = this.webSocketFactory(url);
    } catch {
      this.setStatus("unsynchronized");
      this.scheduleReconnect();
      return;
    }

    this.socket = socket;
    socket.addEventListener("open", () => {
      if (generation !== this.generation || this.socket !== socket) return;
      // Cookies are sent automatically for same-origin WebSockets.
      // Wait for room.ready before accepting commands.
    });
    socket.addEventListener("message", (event) => {
      if (generation !== this.generation || this.socket !== socket) return;
      this.handleMessage(event.data);
    });
    socket.addEventListener("close", (event) => {
      if (generation !== this.generation || this.socket !== socket) return;
      this.socket = null;
      this.handleClose(event.code, event.reason ?? "");
    });
    socket.addEventListener("error", () => {
      if (generation !== this.generation || this.socket !== socket) return;
      // close follows; mark unsynchronized until then
      if (this.status === "ready") {
        this.setStatus("unsynchronized");
      }
    });
  }

  private handleMessage(data: unknown): void {
    let raw: unknown = data;
    if (typeof data === "string") {
      try {
        raw = JSON.parse(data) as unknown;
      } catch {
        this.stopTraffic("protocol_error");
        return;
      }
    }
    if (
      typeof raw === "object" &&
      raw !== null &&
      "type" in raw &&
      (raw as { type?: unknown }).type === "room.heartbeat"
    ) {
      return;
    }
    const message = parseServerRoomMessage(raw);
    if (message === null) {
      this.stopTraffic("protocol_error");
      return;
    }

    if (message.type === "room.ready") {
      this.applyReady(message);
      return;
    }
    if (message.type === "room.rehydrate") {
      this.head = message.head;
      this.listeners.onRehydrate?.(message.head);
      return;
    }
    if (message.type === "presence.join" || message.type === "presence.update") {
      this.upsertParticipant(message.participant);
      return;
    }
    if (message.type === "presence.leave") {
      this.removeParticipant(message.graph_room_session_id);
      return;
    }
    if (message.type === "graph.command.accepted") {
      this.applyAccepted(message);
      return;
    }
    if (message.type === "graph.command.receipt") {
      this.applyReceipt(message);
      return;
    }
    if (message.type === "graph.command.rejected") {
      this.applyRejected(message);
    }
  }

  private applyReady(message: RoomReadyMessage): void {
    if (
      message.workspace_id !== this.workspaceId ||
      message.graph_id !== this.graphId
    ) {
      this.stopTraffic("protocol_error");
      return;
    }
    this.ready = message;
    this.head = message.head;
    this.capabilities = message.capabilities.capabilities;
    this.authorizationVersion = message.capabilities.authorization_version;
    this.participants = new Map(
      message.participants.map((participant) => [
        participant.graph_room_session_id,
        participant,
      ]),
    );
    this.presenceSequence = 0;
    this.lastPresenceSentAt = 0;
    this.setStatus("ready");
    this.listeners.onReady?.(message);
    this.emitPresenceChange();
    this.drainQueue();
  }

  private upsertParticipant(participant: PresenceParticipant): void {
    this.participants.set(participant.graph_room_session_id, participant);
    this.emitPresenceChange();
  }

  private removeParticipant(sessionId: string): void {
    if (!this.participants.delete(sessionId)) return;
    this.emitPresenceChange();
  }

  private clearPresence(): void {
    if (this.participants.size === 0) return;
    this.participants.clear();
    this.emitPresenceChange();
  }

  private emitPresenceChange(): void {
    this.listeners.onPresenceChange?.(this.getParticipants());
  }

  private applyAccepted(message: GraphCommandAcceptedMessage): void {
    if (this.head && message.room_epoch === this.head.room_epoch) {
      if (message.sequence > this.head.collaboration_sequence) {
        this.head = {
          ...this.head,
          collaboration_sequence: message.sequence,
        };
      }
    }
    if (this.inFlight?.commandId === message.command_id) {
      this.inFlight.accepted = message;
    }
    this.listeners.onCommandAccepted?.(message);
  }

  private applyReceipt(message: GraphCommandReceiptMessage): void {
    const pending = this.inFlight;
    if (!pending || pending.commandId !== message.command_id) {
      return;
    }
    if (this.head) {
      this.head = {
        ...this.head,
        room_epoch: message.current_room_epoch,
        collaboration_sequence: message.current_sequence,
      };
    }
    this.inFlight = null;
    pending.resolve({
      receipt: message,
      accepted: pending.accepted,
    });
    this.drainQueue();
  }

  private applyRejected(message: GraphCommandRejectedMessage): void {
    this.listeners.onCommandRejected?.(message);
    const pending = this.inFlight;
    if (!pending || pending.commandId !== message.command_id) {
      return;
    }
    if (
      message.current_room_epoch !== null &&
      message.current_sequence !== null &&
      this.head
    ) {
      this.head = {
        ...this.head,
        room_epoch: message.current_room_epoch,
        collaboration_sequence: message.current_sequence,
      };
    }
    this.inFlight = null;
    pending.reject(
      new GraphRoomCommandError(
        message.command_id,
        message.error_code,
        message.detail,
      ),
    );
    this.drainQueue();
  }

  private drainQueue(): void {
    if (this.inFlight || !this.canSubmitCommands() || this.head === null) {
      return;
    }
    const next = this.queue.shift();
    if (!next) return;

    const payload = {
      protocol_version: ROOM_PROTOCOL_VERSION,
      type: "graph.command.submit" as const,
      command_id: next.commandId,
      room_epoch: this.head.room_epoch,
      observed_sequence: this.head.collaboration_sequence,
      command: next.command,
    };
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      this.queue.unshift(next);
      return;
    }
    this.inFlight = next;
    next.sent = true;
    try {
      socket.send(JSON.stringify(payload));
    } catch (error) {
      this.inFlight = null;
      next.reject(
        error instanceof Error
          ? error
          : new GraphRoomCommandError(
              next.commandId,
              "send_failed",
              "Failed to send graph command.",
            ),
      );
    }
  }

  private handleClose(code: number, reason: string): void {
    const terminal = terminalReasonFromClose(code, reason);
    if (terminal !== null) {
      this.stopTraffic(terminal);
      return;
    }
    this.ready = null;
    this.clearPresence();
    this.rejectInFlight(
      new GraphRoomCommandError(
        this.inFlight?.commandId ?? "",
        "disconnected",
        "Graph room disconnected before the command completed.",
      ),
    );
    if (this.intentionallyClosed) {
      this.setStatus("idle");
      return;
    }
    this.setStatus("unsynchronized");
    this.scheduleReconnect();
  }

  private stopTraffic(reason: GraphRoomTerminalReason): void {
    this.intentionallyClosed = true;
    this.clearReconnectTimer();
    this.terminalReason = reason;
    this.ready = null;
    this.capabilities = [];
    this.authorizationVersion = null;
    this.clearPresence();
    this.generation += 1;
    const socket = this.socket;
    this.socket = null;
    if (socket && socket.readyState < WebSocket.CLOSING) {
      socket.close();
    }
    this.rejectAllQueued(
      new GraphRoomCommandError(
        "",
        reason,
        `Graph room closed (${reason}). Protected traffic has stopped.`,
      ),
    );
    this.setStatus("stopped");
    this.listeners.onTerminalClose?.(reason);
  }

  private rejectInFlight(error: Error): void {
    if (!this.inFlight) return;
    const pending = this.inFlight;
    this.inFlight = null;
    pending.reject(error);
  }

  private rejectAllQueued(error: Error): void {
    this.rejectInFlight(error);
    const queued = this.queue.splice(0, this.queue.length);
    for (const item of queued) {
      item.reject(error);
    }
  }

  private scheduleReconnect(): void {
    if (this.intentionallyClosed || this.terminalReason !== null) return;
    this.clearReconnectTimer();
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (this.intentionallyClosed || this.terminalReason !== null) return;
      this.openSocket();
    }, this.reconnectDelayMs);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer === null) return;
    clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
  }

  private setStatus(status: GraphRoomStatus): void {
    if (this.status === status) return;
    this.status = status;
    this.listeners.onStatusChange?.(status);
  }
}
