import type { CollaborativeHead, WorkspaceCapability } from "@/lib/api";

import { graphRoomWebSocketUrl } from "./graph-room-url";
import {
  ROOM_COMMAND_QUEUE_CAP,
  ROOM_PROTOCOL_VERSION,
  parseServerRoomMessage,
  terminalReasonFromClose,
  type ActiveExecutionSummary,
  type ActorPresentation,
  type ExecutionClearedMessage,
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
import { applyRoomCommandToHead } from "./room-command-bridge";

export type { GraphRoomStatus, GraphRoomTerminalReason, RoomGraphCommand };
export type { ActiveExecutionSummary, PresenceParticipant, PresenceUpdateSubmit };
export { ROOM_COMMAND_QUEUE_CAP, graphRoomWebSocketUrl };

export const PRESENCE_CLIENT_MIN_INTERVAL_MS = 50;

/** True when `incoming` should replace the session's confirmed head snapshot. */
export function shouldReplaceCollaborativeHead(
  current: CollaborativeHead | null | undefined,
  incoming: CollaborativeHead,
): boolean {
  if (!current) return true;
  if (current.room_epoch !== incoming.room_epoch) return true;
  return incoming.collaboration_sequence >= current.collaboration_sequence;
}

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

export interface GraphRoomAcceptedMeta {
  readonly local: boolean;
}

export interface GraphRoomSessionListeners {
  onStatusChange?: (status: GraphRoomStatus) => void;
  onReady?: (ready: RoomReadyMessage) => void;
  onRehydrate?: (head: CollaborativeHead) => void;
  /** Session paused command drain until replaceHead supplies a full snapshot. */
  onHeadRefreshRequired?: () => void;
  onCommandAccepted?: (
    message: GraphCommandAcceptedMessage,
    meta: GraphRoomAcceptedMeta,
  ) => void;
  onCommandRejected?: (message: GraphCommandRejectedMessage) => void;
  onPresenceChange?: (participants: readonly PresenceParticipant[]) => void;
  onActiveExecution?: (execution: ActiveExecutionSummary | null) => void;
  onExecutionCleared?: (message: ExecutionClearedMessage) => void;
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
  roomEpoch: string | null;
  observedSequence: number | null;
  accepted: GraphCommandAcceptedMessage | null;
  receipt: GraphCommandReceiptMessage | null;
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
  private activeExecution: ActiveExecutionSummary | null = null;

  private readonly queue: QueuedCommand[] = [];
  private inFlight: QueuedCommand | null = null;
  private readonly localCommandIds = new Set<string>();
  /** When true, do not send queued commands until replaceHead restores a full snapshot. */
  private awaitingHeadRehydration = false;

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
      onHeadRefreshRequired: options.onHeadRefreshRequired,
      onCommandAccepted: options.onCommandAccepted,
      onCommandRejected: options.onCommandRejected,
      onPresenceChange: options.onPresenceChange,
      onActiveExecution: options.onActiveExecution,
      onExecutionCleared: options.onExecutionCleared,
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

  getActiveExecution(): ActiveExecutionSummary | null {
    return this.activeExecution;
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
    this.localCommandIds.clear();
    this.setStatus("idle");
    this.ready = null;
    this.clearPresence();
  }

  /**
   * Replace the cached collaborative head (checkpoint or conflict recovery).
   * Ignores stale same-epoch snapshots so a late HTTP fetch cannot wipe a newer
   * WebSocket-applied head (e.g. a just-accepted presentation).
   */
  replaceHead(head: CollaborativeHead): void {
    if (!shouldReplaceCollaborativeHead(this.head, head)) {
      const pendingReceipt = this.inFlight?.receipt;
      if (
        pendingReceipt === null ||
        pendingReceipt === undefined ||
        !this.headCoversReceipt(pendingReceipt)
      ) {
        return;
      }
      this.finishHeadRehydration();
      return;
    }
    this.head = head;
    this.finishHeadRehydration();
  }

  /**
   * Reconcile a checkpoint response against the epoch of its submitted command.
   * A later epoch reset is authoritative and must not be reversed by the old
   * epoch's HTTP continuation.
   */
  reconcileCheckpointHead(
    checkpointHead: CollaborativeHead,
    expectedRoomEpoch: string,
  ): CollaborativeHead {
    const currentHead = this.head;
    if (!currentHead) {
      throw new Error(
        `Graph room head is unavailable while reconciling checkpoint epoch ${expectedRoomEpoch}.`,
      );
    }
    if (currentHead.room_epoch !== expectedRoomEpoch) {
      return currentHead;
    }
    if (checkpointHead.room_epoch !== expectedRoomEpoch) {
      throw new Error(
        `Checkpoint response epoch ${checkpointHead.room_epoch} does not match submitted command epoch ${expectedRoomEpoch}.`,
      );
    }

    this.replaceHead(checkpointHead);
    let effectiveHead = this.head;
    if (!effectiveHead) {
      throw new Error(
        `Graph room head is unavailable after reconciling checkpoint epoch ${expectedRoomEpoch}.`,
      );
    }
    if (
      effectiveHead.collaboration_sequence >
        checkpointHead.collaboration_sequence &&
      checkpointHead.checkpoint_sequence > effectiveHead.checkpoint_sequence
    ) {
      effectiveHead = {
        ...effectiveHead,
        checkpoint_sequence: checkpointHead.checkpoint_sequence,
        checkpoint_revision: checkpointHead.checkpoint_revision,
      };
      this.replaceHead(effectiveHead);
    }
    return effectiveHead;
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
    this.supersedeQueuedSnapshotCommand(command);
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
    this.localCommandIds.add(commandId);
    return new Promise<GraphRoomCommandResult>((resolve, reject) => {
      this.queue.push({
        commandId,
        command,
        resolve,
        reject,
        sent: false,
        roomEpoch: null,
        observedSequence: null,
        accepted: null,
        receipt: null,
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
      const keptCurrentHead = Boolean(
        this.head && !shouldReplaceCollaborativeHead(this.head, message.head),
      );
      const rehydratedHead = keptCurrentHead ? this.head! : message.head;
      this.head = rehydratedHead;
      this.listeners.onRehydrate?.(rehydratedHead);
      if (keptCurrentHead) return;
      this.finishHeadRehydration();
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
    if (message.type === "execution.active") {
      this.activeExecution = message.execution;
      this.listeners.onActiveExecution?.(message.execution);
      return;
    }
    if (message.type === "execution.cleared") {
      if (
        this.activeExecution?.execution_id === message.execution_id ||
        this.activeExecution === null
      ) {
        this.activeExecution = null;
        this.listeners.onActiveExecution?.(null);
      }
      this.listeners.onExecutionCleared?.(message);
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
    let ready = message;
    if (
      this.head &&
      !shouldReplaceCollaborativeHead(this.head, message.head)
    ) {
      ready = { ...message, head: this.head };
    }
    this.ready = ready;
    this.head = ready.head;
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
    this.activeExecution = message.active_execution;
    this.setStatus("ready");
    this.listeners.onReady?.(ready);
    this.listeners.onActiveExecution?.(message.active_execution);
    this.emitPresenceChange();
    this.finishHeadRehydration();
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
    const local = this.localCommandIds.delete(message.command_id);
    if (this.inFlight?.commandId === message.command_id) {
      this.inFlight.accepted = message;
    }

    const head = this.head;
    if (!head || message.room_epoch !== head.room_epoch) {
      this.requestHeadRefresh();
      return;
    }
    if (message.sequence <= head.collaboration_sequence) {
      return;
    }
    if (message.sequence !== head.collaboration_sequence + 1) {
      this.requestHeadRefresh();
      return;
    }

    try {
      this.head = applyRoomCommandToHead(
        head,
        message.command,
        message.sequence,
      );
    } catch {
      this.requestHeadRefresh();
      return;
    }
    this.listeners.onCommandAccepted?.(message, { local });
    const pendingReceipt = this.inFlight?.receipt;
    if (
      this.awaitingHeadRehydration &&
      pendingReceipt !== null &&
      pendingReceipt !== undefined &&
      this.headCoversReceipt(pendingReceipt)
    ) {
      this.finishHeadRehydration();
    }
  }

  private applyReceipt(message: GraphCommandReceiptMessage): void {
    const pending = this.inFlight;
    if (!pending || pending.commandId !== message.command_id) {
      return;
    }
    this.localCommandIds.delete(message.command_id);
    pending.receipt = message;
    if (
      message.requires_head_rehydration ||
      !this.headCoversReceipt(message)
    ) {
      this.requestHeadRefresh();
      return;
    }
    this.resolveInFlightReceipt(pending);
    this.drainQueue();
  }

  private headCoversReceipt(message: GraphCommandReceiptMessage): boolean {
    return (
      this.head !== null &&
      this.head.room_epoch === message.current_room_epoch &&
      this.head.collaboration_sequence >= message.current_sequence
    );
  }

  private resolveInFlightReceipt(pending: QueuedCommand): void {
    const receipt = pending.receipt;
    if (this.inFlight !== pending || receipt === null) return;
    this.inFlight = null;
    pending.resolve({
      receipt,
      accepted: pending.accepted,
    });
  }

  private finishHeadRehydration(): void {
    const pending = this.inFlight;
    if (pending?.receipt !== null && pending?.receipt !== undefined) {
      if (!this.headCoversReceipt(pending.receipt)) {
        this.awaitingHeadRehydration = true;
        this.listeners.onHeadRefreshRequired?.();
        return;
      }
      this.resolveInFlightReceipt(pending);
    }
    this.awaitingHeadRehydration = false;
    this.drainQueue();
  }

  private applyRejected(message: GraphCommandRejectedMessage): void {
    this.localCommandIds.delete(message.command_id);
    const pending = this.inFlight;
    if (!pending || pending.commandId !== message.command_id) {
      this.listeners.onCommandRejected?.(message);
      return;
    }
    // Do not advance collaboration_sequence from the rejection alone — that
    // skips applying a peer's accepted command at the same sequence, and lets
    // the queue drain with a document that never received the winner.
    // Pause drain until Workbench replaceHead() restores a full snapshot.
    this.awaitingHeadRehydration = true;
    this.inFlight = null;
    this.listeners.onCommandRejected?.(message);
    pending.reject(
      new GraphRoomCommandError(
        message.command_id,
        message.error_code,
        message.detail,
      ),
    );
  }

  private requestHeadRefresh(): void {
    if (this.awaitingHeadRehydration) {
      this.listeners.onHeadRefreshRequired?.();
      return;
    }
    this.awaitingHeadRehydration = true;
    this.listeners.onHeadRefreshRequired?.();
  }

  private supersedeQueuedSnapshotCommand(command: RoomGraphCommand): void {
    if (
      command.kind !== "replace_presentation" &&
      command.kind !== "replace_document"
    ) {
      return;
    }
    for (let index = this.queue.length - 1; index >= 0; index -= 1) {
      const item = this.queue[index];
      if (!item || item.sent || item.command.kind !== command.kind) continue;
      this.queue.splice(index, 1);
      this.localCommandIds.delete(item.commandId);
      item.reject(
        new GraphRoomCommandError(
          item.commandId,
          "superseded",
          "A newer snapshot command replaced this queued command.",
        ),
      );
    }
  }

  private drainQueue(): void {
    if (
      this.inFlight ||
      this.awaitingHeadRehydration ||
      !this.canSubmitCommands() ||
      this.head === null
    ) {
      return;
    }
    const next = this.queue.shift();
    if (!next) return;

    const roomEpoch = next.roomEpoch ?? this.head.room_epoch;
    const observedSequence =
      next.observedSequence ?? this.head.collaboration_sequence;
    const payload = {
      protocol_version: ROOM_PROTOCOL_VERSION,
      type: "graph.command.submit" as const,
      command_id: next.commandId,
      room_epoch: roomEpoch,
      observed_sequence: observedSequence,
      command: next.command,
    };
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      this.queue.unshift(next);
      return;
    }
    this.inFlight = next;
    next.sent = true;
    next.roomEpoch = roomEpoch;
    next.observedSequence = observedSequence;
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
    if (this.inFlight) {
      const interrupted = this.inFlight;
      this.inFlight = null;
      this.queue.unshift(interrupted);
    }
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
