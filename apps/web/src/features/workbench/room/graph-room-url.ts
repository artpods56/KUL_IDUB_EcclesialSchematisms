import { API_BASE } from "@/lib/api/client";

/** Same-origin WebSocket URL for a workspace-scoped graph room. */
export function graphRoomWebSocketUrl(
  workspaceId: string,
  graphId: string,
): string {
  const path =
    `${API_BASE}/v1/workspaces/${encodeURIComponent(workspaceId)}` +
    `/graphs/${encodeURIComponent(graphId)}/room`;
  if (typeof window === "undefined") {
    return path;
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}
