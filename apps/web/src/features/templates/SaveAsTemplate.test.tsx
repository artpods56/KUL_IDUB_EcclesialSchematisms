// @vitest-environment jsdom

import * as React from "react";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Workspace } from "@/lib/api";


Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const testState = vi.hoisted(() => ({
  push: vi.fn(),
  create: vi.fn(),
}));

const location: Workspace = {
  id: "personal-location",
  slug: "personal-user",
  name: "Personal",
  kind: "personal",
  role: "owner",
  capabilities: ["view_graph", "create_graph", "create_template"],
};

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: testState.push }),
}));

vi.mock("next/link", () => ({
  default: ({ children, ...props }: React.ComponentProps<"a">) => (
    <a {...props}>{children}</a>
  ),
}));

vi.mock("@/features/auth/AuthSessionBoundary", () => ({
  useAuthSession: () => ({
    session: { user_id: "user-1" },
    logout: vi.fn(),
  }),
}));

vi.mock("@/features/workspaces/WorkspaceLayout", () => ({
  WorkspaceRail: () => null,
}));

vi.mock("@/hooks/use-api", () => ({
  useWorkspaces: () => ({ data: [location] }),
  useSavedGraphs: () => ({
    data: {
      graphs: [
        {
          id: "source-graph",
          name: "Quarterly analysis",
          revision: 9,
          node_count: 3,
          edge_count: 2,
          updated_at: "2026-08-11T08:00:00Z",
        },
      ],
    },
    error: undefined,
    isLoading: false,
    mutate: vi.fn(),
  }),
}));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    createWorkspaceTemplate: testState.create,
  };
});

import { SaveAsTemplate } from "./SaveAsTemplate";


async function renderSaveFlow(): Promise<{
  container: HTMLDivElement;
  root: Root;
}> {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  await act(async () =>
    root.render(
      <SaveAsTemplate
        source={{
          workspaceId: location.id,
          graphId: "source-graph",
          revision: 7,
        }}
      />,
    ),
  );
  return { container, root };
}


beforeEach(() => {
  testState.push.mockReset();
  testState.create.mockReset();
});

afterEach(() => {
  document.body.replaceChildren();
});


describe("save as template flow", () => {
  it("retries the exact-revision snapshot and returns to the template library", async () => {
    testState.create
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ id: "created-template" });
    const { container, root } = await renderSaveFlow();
    expect(container.textContent).toContain("revision 7 · My graphs");

    const form = container.querySelector("form");
    await act(async () => {
      form?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(container.textContent).toContain(
      "Your template is unchanged; try again.",
    );

    await act(async () => {
      form?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(testState.create).toHaveBeenLastCalledWith("personal-location", {
      source_graph_id: "source-graph",
      source_revision: 7,
      name: "Quarterly analysis",
      description: null,
    });
    expect(testState.push).toHaveBeenCalledWith(
      "/templates?created=created-template",
    );
    await act(async () => root.unmount());
  });

  it("fails closed when the route has no exact source", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () => root.render(<SaveAsTemplate source={null} />));
    expect(container.textContent).toContain("Source graph is missing");
    expect(container.querySelector("form")).toBeNull();
    await act(async () => root.unmount());
  });
});
