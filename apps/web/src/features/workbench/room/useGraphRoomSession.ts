"use client";

import * as React from "react";

import type { CollaborativeHead } from "@/lib/api";

import {
  GraphRoomSession,
  type GraphRoomStatus,
  type GraphRoomTerminalReason,
  type PresenceParticipant,
  type PresenceUpdateSubmit,
  type RoomGraphCommand,
} from "./graph-room-session";
import type { RoomReadyMessage } from "./protocol";

export interface GraphRoomSubmitResult {
  readonly receipt: Awaited<
    ReturnType<GraphRoomSession["submitCommand"]>
  >["receipt"];
  readonly accepted: Awaited<
    ReturnType<GraphRoomSession["submitCommand"]>
  >["accepted"];
  readonly head: CollaborativeHead;
}

export interface UseGraphRoomSessionResult {
  readonly status: GraphRoomStatus;
  readonly terminalReason: GraphRoomTerminalReason | null;
  readonly head: CollaborativeHead | null;
  readonly capabilities: readonly string[];
  readonly authorizationVersion: number | null;
  readonly canSubmitCommands: boolean;
  readonly canPublishPresence: boolean;
  readonly localSessionId: string | null;
  readonly participants: readonly PresenceParticipant[];
  readonly submitCommand: (
    command: RoomGraphCommand,
  ) => Promise<GraphRoomSubmitResult>;
  readonly publishPresence: (
    update: Omit<PresenceUpdateSubmit, "presence_sequence">,
  ) => boolean;
}

interface UseGraphRoomSessionOptions {
  workspaceId: string;
  graphId: string | null;
  onReady?: (ready: RoomReadyMessage) => void;
  onRehydrate?: (head: CollaborativeHead) => void;
  onTerminalClose?: (reason: GraphRoomTerminalReason) => void;
}

export function useGraphRoomSession({
  workspaceId,
  graphId,
  onReady,
  onRehydrate,
  onTerminalClose,
}: UseGraphRoomSessionOptions): UseGraphRoomSessionResult {
  const [status, setStatus] = React.useState<GraphRoomStatus>("idle");
  const [terminalReason, setTerminalReason] =
    React.useState<GraphRoomTerminalReason | null>(null);
  const [head, setHead] = React.useState<CollaborativeHead | null>(null);
  const [capabilities, setCapabilities] = React.useState<readonly string[]>([]);
  const [authorizationVersion, setAuthorizationVersion] = React.useState<
    number | null
  >(null);
  const [participants, setParticipants] = React.useState<
    readonly PresenceParticipant[]
  >([]);
  const [localSessionId, setLocalSessionId] = React.useState<string | null>(
    null,
  );
  const sessionRef = React.useRef<GraphRoomSession | null>(null);

  const onReadyRef = React.useRef(onReady);
  const onRehydrateRef = React.useRef(onRehydrate);
  const onTerminalCloseRef = React.useRef(onTerminalClose);
  onReadyRef.current = onReady;
  onRehydrateRef.current = onRehydrate;
  onTerminalCloseRef.current = onTerminalClose;

  React.useEffect(() => {
    if (!graphId) {
      sessionRef.current?.disconnect();
      sessionRef.current = null;
      setStatus("idle");
      setTerminalReason(null);
      setHead(null);
      setCapabilities([]);
      setAuthorizationVersion(null);
      setParticipants([]);
      setLocalSessionId(null);
      return;
    }

    const session = new GraphRoomSession({
      workspaceId,
      graphId,
      onStatusChange: setStatus,
      onReady: (ready) => {
        setHead(ready.head);
        setCapabilities(ready.capabilities.capabilities);
        setAuthorizationVersion(ready.capabilities.authorization_version);
        setTerminalReason(null);
        setLocalSessionId(ready.graph_room_session_id);
        setParticipants(ready.participants);
        onReadyRef.current?.(ready);
      },
      onRehydrate: (nextHead) => {
        setHead(nextHead);
        onRehydrateRef.current?.(nextHead);
      },
      onPresenceChange: setParticipants,
      onTerminalClose: (reason) => {
        setTerminalReason(reason);
        setCapabilities([]);
        setAuthorizationVersion(null);
        setParticipants([]);
        setLocalSessionId(null);
        onTerminalCloseRef.current?.(reason);
      },
    });
    sessionRef.current = session;
    session.connect();

    return () => {
      session.disconnect();
      if (sessionRef.current === session) {
        sessionRef.current = null;
      }
    };
  }, [workspaceId, graphId]);

  const submitCommand = React.useCallback(
    async (command: RoomGraphCommand): Promise<GraphRoomSubmitResult> => {
      const session = sessionRef.current;
      if (!session) {
        throw new Error("Graph room is not connected.");
      }
      const result = await session.submitCommand(command);
      const nextHead = session.getHead();
      if (!nextHead) {
        throw new Error("Graph room head is unavailable after command submit.");
      }
      setHead(nextHead);
      return { ...result, head: nextHead };
    },
    [],
  );

  const publishPresence = React.useCallback(
    (update: Omit<PresenceUpdateSubmit, "presence_sequence">): boolean => {
      return sessionRef.current?.publishPresence(update) ?? false;
    },
    [],
  );

  return {
    status,
    terminalReason,
    head,
    capabilities,
    authorizationVersion,
    canSubmitCommands:
      status === "ready" &&
      head !== null &&
      capabilities.includes("edit_graph") &&
      terminalReason === null,
    canPublishPresence:
      status === "ready" &&
      capabilities.includes("publish_presence") &&
      terminalReason === null,
    localSessionId,
    participants,
    submitCommand,
    publishPresence,
  };
}
