import { afterEach, describe, expect, it, vi } from "vitest";

import { deleteSession, getSession, oidcLoginUrl, safeReturnPath } from "./auth";

afterEach(() => vi.unstubAllGlobals());

describe("auth API client", () => {
  it("uses generated session endpoints through same-origin request", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ user_id: "user" }), { status: 200 }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await getSession();
    await deleteSession();

    expect(fetchMock.mock.calls[0]?.[0]).toBe("/api/v1/auth/session");
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({ method: "GET", credentials: "same-origin" });
    expect(fetchMock.mock.calls[1]?.[1]).toMatchObject({ method: "DELETE", credentials: "same-origin" });
  });

  it("only permits same-origin relative OIDC return paths", () => {
    expect(safeReturnPath("/workspaces/acme?tab=members#top")).toBe("/workspaces/acme?tab=members#top");
    expect(safeReturnPath("//evil.example/login")).toBe("/workspaces");
    expect(safeReturnPath("https://evil.example/login")).toBe("/workspaces");
    expect(oidcLoginUrl("/workspaces/local")).toBe(
      "/api/v1/auth/oidc/login?return_path=%2Fworkspaces%2Flocal",
    );
  });
});
