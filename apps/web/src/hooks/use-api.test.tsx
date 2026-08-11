// @vitest-environment jsdom

import * as React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { SWRConfig } from "swr";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Workspace } from "@/lib/api/contract";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const apiMocks = vi.hoisted(() => ({
  request: vi.fn(),
  listWorkspaces: vi.fn(),
  listWorkspaceMembers: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  listWorkspaces: apiMocks.listWorkspaces,
  listWorkspaceMembers: apiMocks.listWorkspaceMembers,
}));

vi.mock("@/lib/api/client", () => ({
  request: apiMocks.request,
}));

import {
  type AllWorkspacesGraphsResult,
  useAllWorkspacesGraphs,
} from "./use-api";

const personal: Workspace = {
  id: "workspace-personal",
  name: "Personal workspace",
  slug: "personal",
  kind: "personal",
  role: "owner",
  capabilities: ["view_graph", "create_graph"],
};

const team: Workspace = {
  id: "workspace-team",
  name: "Atlas",
  slug: "atlas",
  kind: "shared",
  role: "editor",
  capabilities: ["view_graph", "create_graph"],
};

let latest: AllWorkspacesGraphsResult | undefined;

function captureLatest(value: AllWorkspacesGraphsResult) {
  latest = value;
}

function Harness({ workspaces }: { workspaces: readonly Workspace[] }) {
  const value = useAllWorkspacesGraphs(workspaces);
  React.useEffect(() => captureLatest(value), [value]);
  return <span>{value.graphs?.length ?? "loading"}</span>;
}

beforeEach(() => {
  latest = undefined;
  apiMocks.request.mockReset();
});

afterEach(() => {
  document.body.replaceChildren();
});

describe("useAllWorkspacesGraphs", () => {
  it("fans out one typed request per workspace and preserves partial failures", async () => {
    apiMocks.request.mockImplementation(
      async (_method: string, path: string) => {
        if (path.includes("workspace-personal")) {
          return {
            graphs: [
              {
                id: "graph-1",
                name: "Invoice intake",
                node_count: 2,
                edge_count: 1,
                revision: 3,
                updated_at: "2026-08-10T12:00:00Z",
              },
            ],
          };
        }
        throw new Error("Team location unavailable");
      },
    );

    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => {
      root.render(
        <SWRConfig
          value={{
            provider: () => new Map(),
            dedupingInterval: 0,
            revalidateOnFocus: false,
          }}
        >
          <Harness workspaces={[personal, team]} />
        </SWRConfig>,
      );
    });

    await vi.waitFor(() => expect(latest?.graphs).toHaveLength(1));
    expect(apiMocks.request.mock.calls).toEqual([
      ["GET", "/v1/workspaces/workspace-personal/graphs"],
      ["GET", "/v1/workspaces/workspace-team/graphs"],
    ]);
    expect(latest?.graphs?.[0]?.location).toEqual({
      id: personal.id,
      slug: personal.slug,
      name: personal.name,
      kind: personal.kind,
    });
    expect(latest?.failures).toHaveLength(1);
    expect(latest?.failures[0]?.location.name).toBe("Atlas");

    await act(async () => latest?.retry());
    await vi.waitFor(() => expect(apiMocks.request).toHaveBeenCalledTimes(4));

    await act(async () => root.unmount());
  });

  it("returns a settled true-empty result without issuing a malformed request", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => {
      root.render(
        <SWRConfig value={{ provider: () => new Map() }}>
          <Harness workspaces={[]} />
        </SWRConfig>,
      );
    });

    expect(latest?.graphs).toEqual([]);
    expect(latest?.isLoading).toBe(false);
    expect(apiMocks.request).not.toHaveBeenCalled();

    await act(async () => root.unmount());
  });
});
