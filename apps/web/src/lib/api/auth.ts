import { API_BASE, request } from "./client";
import type { Session } from "./contract";

const DEFAULT_RETURN_PATH = "/workspaces";

export function getSession(signal?: AbortSignal) {
  return request<Session>("GET", "/v1/auth/session", { signal });
}

export function deleteSession() {
  return request<undefined>("DELETE", "/v1/auth/session");
}

export function safeReturnPath(value: string | undefined): string {
  if (!value || !value.startsWith("/") || value.startsWith("//")) {
    return DEFAULT_RETURN_PATH;
  }

  try {
    const parsed = new URL(value, "https://grafy.invalid");
    if (parsed.origin !== "https://grafy.invalid") return DEFAULT_RETURN_PATH;
    return `${parsed.pathname}${parsed.search}${parsed.hash}`;
  } catch {
    return DEFAULT_RETURN_PATH;
  }
}

export function oidcLoginUrl(returnPath?: string): string {
  const path = safeReturnPath(returnPath);
  return `${API_BASE}/v1/auth/oidc/login?return_path=${encodeURIComponent(path)}`;
}
