// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  CLOSE_ACCESS_REVOKED,
  CLOSE_GRAPH_DELETED,
  CLOSE_PERMISSIONS_CHANGED,
  ROOM_COMMAND_QUEUE_CAP,
} from "./protocol";
import {
  GraphRoomCommandError,
  GraphRoomSession,
  graphRoomWebSocketUrl,
} from "./graph-room-session";

const WORKSPACE_ID = "00000000-0000-4000-8000-000000000007";
const GRAPH_ID = "11111111-1111-4111-8111-111111111111";
const SESSION_ID = "22222222-2222-4222-8222-222222222222";
const ACTOR_ID = "00000000-0000-4000-8000-000000000001";
const ROOM_EPOCH = "33333333-3333-4333-8333-333333333333";

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  readonly sent: string[] = [];
  readyState = FakeWebSocket.CONNECTING;
  close = vi.fn((code?: number, reason?: string) => {
    this.readyState = FakeWebSocket.CLOSED;
    this.dispatch("close", { code: code ?? 1000, reason: reason ?? "" });
  });

  private listeners = new Map<string, Set<(event: unknown) => void>>();

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, listener: (event: unknown) => void): void {
    const bucket = this.listeners.get(type) ?? new Set();
    bucket.add(listener);
    this.listeners.set(type, bucket);
  }

  send(data: string): void {
    this.sent.push(data);
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.dispatch("open", {});
  }

  emitMessage(payload: unknown): void {
    this.dispatch("message", { data: JSON.stringify(payload) });
  }

  emitClose(code: number, reason: string): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.dispatch("close", { code, reason });
  }

  private dispatch(type: string, event: unknown): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }
}

function roomReady(overrides: Record<string, unknown> = {}) {
  return {
    protocol_version: 1,
    type: "room.ready",
    workspace_id: WORKSPACE_ID,
    graph_id: GRAPH_ID,
    graph_room_session_id: SESSION_ID,
    actor: {
      actor_id: ACTOR_ID,
      display_name: "Owner",
      color: "emerald",
    },
    capabilities: {
      capabilities: ["view_graph", "edit_graph", "join_graph_room"],
      authorization_version: 2,
    },
    head: {
      graph_id: GRAPH_ID,
      room_epoch: ROOM_EPOCH,
      collaboration_sequence: 4,
      checkpoint_sequence: 1,
      checkpoint_revision: 1,
      name: "Room graph",
      updated_at: "2026-08-07T10:00:00Z",
      nodes: [],
      edges: [],
    },
    participants: [],
    active_execution: null,
    registry_marker: "builtin",
    ...overrides,
  };
}

function connectReadySession(
  options: ConstructorParameters<typeof GraphRoomSession>[0] = {
    workspaceId: WORKSPACE_ID,
    graphId: GRAPH_ID,
  },
): { session: GraphRoomSession; socket: FakeWebSocket } {
  const session = new GraphRoomSession({
    ...options,
    workspaceId: options.workspaceId ?? WORKSPACE_ID,
    graphId: options.graphId ?? GRAPH_ID,
    webSocketFactory: (url) => new FakeWebSocket(url) as unknown as WebSocket,
    reconnectDelayMs: 10_000,
  });
  session.connect();
  const socket = FakeWebSocket.instances.at(-1);
  if (!socket) throw new Error("expected websocket");
  socket.open();
  socket.emitMessage(roomReady());
  return { session, socket };
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("graphRoomWebSocketUrl", () => {
  it("builds a same-origin ws URL under API_BASE", () => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    expect(graphRoomWebSocketUrl(WORKSPACE_ID, GRAPH_ID)).toBe(
      `${protocol}//${window.location.host}/api/v1/workspaces/${WORKSPACE_ID}/graphs/${GRAPH_ID}/room`,
    );
  });
});

describe("GraphRoomSession", () => {
  it("connects with credentials path, parses room.ready, and tracks capabilities", () => {
    const onReady = vi.fn();
    const onStatusChange = vi.fn();
    const { session, socket } = connectReadySession({
      workspaceId: WORKSPACE_ID,
      graphId: GRAPH_ID,
      onReady,
      onStatusChange,
    });

    expect(socket.url).toBe(graphRoomWebSocketUrl(WORKSPACE_ID, GRAPH_ID));
    expect(session.getStatus()).toBe("ready");
    expect(session.getAuthorizationVersion()).toBe(2);
    expect(session.getCapabilities()).toEqual([
      "view_graph",
      "edit_graph",
      "join_graph_room",
    ]);
    expect(session.getHead()?.collaboration_sequence).toBe(4);
    expect(session.canSubmitCommands()).toBe(true);
    expect(onReady).toHaveBeenCalledOnce();
    expect(onStatusChange).toHaveBeenCalledWith("connecting");
    expect(onStatusChange).toHaveBeenCalledWith("ready");
  });

  it("submits a command and resolves on receipt after accepted", async () => {
    const commandIds = ["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"];
    const { session, socket } = connectReadySession({
      workspaceId: WORKSPACE_ID,
      graphId: GRAPH_ID,
      createCommandId: () => commandIds.shift() ?? crypto.randomUUID(),
    });

    const pending = session.submitCommand({
      kind: "rename_graph",
      name: "Renamed",
      expected_name: "Room graph",
    });

    expect(socket.sent).toHaveLength(1);
    expect(JSON.parse(socket.sent[0]!)).toEqual({
      protocol_version: 1,
      type: "graph.command.submit",
      command_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      room_epoch: ROOM_EPOCH,
      observed_sequence: 4,
      command: {
        kind: "rename_graph",
        name: "Renamed",
        expected_name: "Room graph",
      },
    });

    socket.emitMessage({
      protocol_version: 1,
      type: "graph.command.accepted",
      command_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      room_epoch: ROOM_EPOCH,
      sequence: 5,
      actor: {
        actor_id: ACTOR_ID,
        display_name: "Owner",
        color: "emerald",
      },
      graph_room_session_id: SESSION_ID,
      command: {
        kind: "rename_graph",
        name: "Renamed",
        expected_name: "Room graph",
      },
    });
    socket.emitMessage({
      protocol_version: 1,
      type: "graph.command.receipt",
      command_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
      outcome: "accepted",
      accepted_room_epoch: ROOM_EPOCH,
      accepted_sequence: 5,
      current_room_epoch: ROOM_EPOCH,
      current_sequence: 5,
      deduplicated: false,
      requires_head_rehydration: false,
    });

    const result = await pending;
    expect(result.receipt.accepted_sequence).toBe(5);
    expect(result.accepted?.sequence).toBe(5);
    expect(session.getHead()?.collaboration_sequence).toBe(5);
  });

  it("rejects a command when the server rejects it", async () => {
    const { session, socket } = connectReadySession({
      workspaceId: WORKSPACE_ID,
      graphId: GRAPH_ID,
      createCommandId: () => "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    });

    const pending = session.submitCommand({
      kind: "rename_graph",
      name: "Nope",
      expected_name: "Wrong",
    });
    socket.emitMessage({
      protocol_version: 1,
      type: "graph.command.rejected",
      command_id: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
      error_code: "field_conflict",
      detail: "Name changed elsewhere.",
      current_room_epoch: ROOM_EPOCH,
      current_sequence: 4,
    });

    await expect(pending).rejects.toMatchObject({
      name: "GraphRoomCommandError",
      errorCode: "field_conflict",
    });
  });

  it("replaces collaborative head state on room.rehydrate", () => {
    const onRehydrate = vi.fn();
    const { session, socket } = connectReadySession({
      workspaceId: WORKSPACE_ID,
      graphId: GRAPH_ID,
      onRehydrate,
    });
    const nextEpoch = "44444444-4444-4444-8444-444444444444";
    const head = {
      graph_id: GRAPH_ID,
      room_epoch: nextEpoch,
      collaboration_sequence: 1,
      checkpoint_sequence: 2,
      checkpoint_revision: 2,
      name: "Reset",
      updated_at: "2026-08-07T11:00:00Z",
      nodes: [],
      edges: [],
    };

    socket.emitMessage({
      protocol_version: 1,
      type: "room.rehydrate",
      reason: "epoch_reset",
      head,
    });

    expect(session.getHead()).toEqual(head);
    expect(onRehydrate).toHaveBeenCalledWith(head);
  });

  it.each([
    {
      code: CLOSE_ACCESS_REVOKED,
      reason: "access_revoked",
      label: "access_revoked",
    },
    {
      code: CLOSE_PERMISSIONS_CHANGED,
      reason: "permissions_changed",
      label: "permissions_changed",
    },
    {
      code: CLOSE_GRAPH_DELETED,
      reason: "graph_deleted",
      label: "graph_deleted",
    },
  ] as const)(
    "stops traffic and exposes $label without reconnecting",
    async ({ code, reason, label }) => {
      vi.useFakeTimers();
      const onTerminalClose = vi.fn();
      const { session, socket } = connectReadySession({
        workspaceId: WORKSPACE_ID,
        graphId: GRAPH_ID,
        onTerminalClose,
        createCommandId: () => "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
      });

      const pending = session.submitCommand({
        kind: "rename_graph",
        name: "Pending",
        expected_name: "Room graph",
      });
      const socketsBeforeClose = FakeWebSocket.instances.length;
      socket.emitClose(code, reason);

      expect(session.getStatus()).toBe("stopped");
      expect(session.getTerminalReason()).toBe(label);
      expect(session.canSubmitCommands()).toBe(false);
      expect(onTerminalClose).toHaveBeenCalledWith(label);
      await expect(pending).rejects.toBeInstanceOf(GraphRoomCommandError);

      await vi.advanceTimersByTimeAsync(20_000);
      expect(FakeWebSocket.instances.length).toBe(socketsBeforeClose);
      await expect(
        session.submitCommand({
          kind: "rename_graph",
          name: "Again",
          expected_name: "Room graph",
        }),
      ).rejects.toMatchObject({ errorCode: "not_ready" });
    },
  );

  it("caps the in-memory command queue", async () => {
    const { session, socket } = connectReadySession({
      workspaceId: WORKSPACE_ID,
      graphId: GRAPH_ID,
      createCommandId: () => crypto.randomUUID(),
    });

    const first = session.submitCommand({
      kind: "rename_graph",
      name: "First",
      expected_name: "Room graph",
    });
    expect(socket.sent).toHaveLength(1);

    const queued = Array.from({ length: ROOM_COMMAND_QUEUE_CAP - 1 }, (_, index) =>
      session.submitCommand({
        kind: "rename_graph",
        name: `Queued ${index}`,
        expected_name: "Room graph",
      }),
    );
    await expect(
      session.submitCommand({
        kind: "rename_graph",
        name: "Overflow",
        expected_name: "Room graph",
      }),
    ).rejects.toMatchObject({ errorCode: "queue_full" });

    socket.emitMessage({
      protocol_version: 1,
      type: "graph.command.rejected",
      command_id: JSON.parse(socket.sent[0]!).command_id,
      error_code: "command_rejected",
      detail: "stop first",
      current_room_epoch: ROOM_EPOCH,
      current_sequence: 4,
    });
    await expect(first).rejects.toBeInstanceOf(GraphRoomCommandError);
    for (const item of queued) {
      item.catch(() => undefined);
    }
    session.disconnect();
  });
});
