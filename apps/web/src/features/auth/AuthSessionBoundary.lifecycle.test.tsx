// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const authMocks = vi.hoisted(() => ({
  getSession: vi.fn(),
  deleteSession: vi.fn(),
}));

vi.mock("@/lib/api/auth", () => ({
  getSession: authMocks.getSession,
  deleteSession: authMocks.deleteSession,
  oidcLoginUrl: () => "/api/v1/auth/oidc/login?return_path=%2Fworkspaces",
  safeReturnPath: (path: string) => path,
}));

import {
  AuthSessionBoundary,
  createProtectedSWRCache,
  useAuthSession,
} from "./AuthSessionBoundary";
import { request } from "@/lib/api/client";

const session = {
  id: "session-1",
  user_id: "user-1",
  created_at: "2026-01-01T00:00:00Z",
  expires_at: "2026-01-02T00:00:00Z",
  last_used_at: null,
  revoked_at: null,
  current: true,
};

function ProtectedSurface() {
  const { logout } = useAuthSession();
  return (
    <div>
      <div data-protected="true">protected</div>
      <button type="button" onClick={() => void logout()}>Log out</button>
    </div>
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
  authMocks.getSession.mockReset();
  authMocks.deleteSession.mockReset();
});

describe("AuthSessionBoundary lifecycle", () => {
  it("drops protected children on post-auth 401 before the request body finishes", async () => {
    authMocks.getSession.mockResolvedValue(session);
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => root.render(<AuthSessionBoundary><ProtectedSurface /></AuthSessionBoundary>));
    expect(container.querySelector("[data-protected]")).not.toBeNull();

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 401 })));
    await act(async () => {
      await expect(request("GET", "/v1/protected")).rejects.toMatchObject({ status: 401 });
    });

    expect(container.querySelector("[data-protected]")).toBeNull();
    expect(container.textContent).toContain("Your session has expired");
    await act(async () => root.unmount());
  });

  it("unmounts protected children on logout failure and signs out only after retry succeeds", async () => {
    authMocks.getSession.mockResolvedValue(session);
    authMocks.deleteSession
      .mockRejectedValueOnce(new Error("private upstream detail"))
      .mockResolvedValueOnce(undefined);
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => root.render(<AuthSessionBoundary><ProtectedSurface /></AuthSessionBoundary>));

    await act(async () => {
      container.querySelector("button")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(container.querySelector("[data-protected]")).toBeNull();
    expect(container.textContent).toContain("Sign out could not be completed");
    expect(container.textContent).not.toContain("private upstream detail");

    await act(async () => {
      container.querySelector("button")?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(container.textContent).toContain("Sign in to Notarius");
    await act(async () => root.unmount());
  });

  it("keeps a deferred write in the discarded cache generation", async () => {
    const oldCache = createProtectedSWRCache();
    const nextCache = createProtectedSWRCache();
    let resolveRequest: ((value: string) => void) | undefined;
    const deferredRequest = new Promise<string>((resolve) => { resolveRequest = resolve; });
    void deferredRequest.then((value) => oldCache.set("protected", { data: value }));

    resolveRequest?.("stale response");
    await deferredRequest;

    expect(oldCache.get("protected")?.data).toBe("stale response");
    expect(nextCache.get("protected")).toBeUndefined();
  });
});
