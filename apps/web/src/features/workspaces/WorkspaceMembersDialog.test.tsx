// @vitest-environment jsdom

import * as React from "react";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

const mocks = vi.hoisted(() => ({
  addWorkspaceMember: vi.fn(),
  changeWorkspaceMemberRole: vi.fn(),
  removeWorkspaceMember: vi.fn(),
  mutateMembers: vi.fn(),
  refreshWorkspaces: vi.fn(),
  members: [
    {
      user: { id: "user-2", display_name: "Second User", email: "second@example.com" },
      role: "viewer",
    },
  ],
  workspace: {
    id: "workspace-1",
    name: "Operations",
    slug: "operations",
    kind: "shared",
    role: "owner",
    capabilities: ["manage_members"],
  },
  session: { user_id: "user-1" },
}));

vi.mock("@/lib/api", () => ({
  addWorkspaceMember: mocks.addWorkspaceMember,
  changeWorkspaceMemberRole: mocks.changeWorkspaceMemberRole,
  removeWorkspaceMember: mocks.removeWorkspaceMember,
}));

vi.mock("@/hooks/use-api", () => ({
  useWorkspaceMembers: () => ({
    data: mocks.members,
    error: undefined,
    mutate: mocks.mutateMembers,
  }),
}));

vi.mock("@/features/auth/AuthSessionBoundary", () => ({
  useAuthSession: () => ({ session: mocks.session }),
}));

vi.mock("./WorkspaceLayout", () => ({
  useWorkspaceContext: () => ({
    workspace: mocks.workspace,
    refreshWorkspaces: mocks.refreshWorkspaces,
  }),
}));

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ open, children }: { open: boolean; children: React.ReactNode }) =>
    open ? <div role="dialog">{children}</div> : null,
  DialogBody: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogDescription: ({ children }: { children: React.ReactNode }) => <p>{children}</p>,
  DialogHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogTitle: ({ children }: { children: React.ReactNode }) => <h2>{children}</h2>,
}));

import { ApiError } from "@/lib/api/client";
import { WorkspaceMembersDialog } from "./WorkspaceMembersDialog";

function renderDialog() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  return { container, root };
}

async function openDialog(container: HTMLElement) {
  const manageButton = container.querySelector("button");
  expect(manageButton).not.toBeNull();
  await act(async () => {
    manageButton?.click();
    await Promise.resolve();
  });
  expect(document.body.querySelector('[role="dialog"]')).not.toBeNull();
}

function fillUserId(value: string) {
  const input = document.body.querySelector('input[placeholder="xxxxxxxx-xxxx-…"]');
  expect(input).toBeInstanceOf(HTMLInputElement);
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  setter?.call(input, value);
  input?.dispatchEvent(new Event("input", { bubbles: true }));
}

async function submitAdd() {
  const form = document.body.querySelector("form");
  expect(form).not.toBeNull();
  await act(async () => {
    form?.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await Promise.resolve();
  });
}

afterEach(() => {
  document.body.innerHTML = "";
  vi.clearAllMocks();
});

beforeEach(() => {
  mocks.addWorkspaceMember.mockResolvedValue(undefined);
  mocks.changeWorkspaceMemberRole.mockResolvedValue(undefined);
  mocks.removeWorkspaceMember.mockResolvedValue(undefined);
  mocks.mutateMembers.mockResolvedValue(mocks.members);
  mocks.refreshWorkspaces.mockResolvedValue([mocks.workspace]);
});

describe("WorkspaceMembersDialog rendered mutation outcomes", () => {
  it("keeps a saved-but-unrefreshed outcome visible and fails closed", async () => {
    mocks.mutateMembers.mockRejectedValue(new Error("member list unavailable"));
    const { container, root } = renderDialog();

    await act(async () => root.render(<WorkspaceMembersDialog />));
    await openDialog(container);
    fillUserId("user-3");
    await submitAdd();

    expect(mocks.addWorkspaceMember).toHaveBeenCalledOnce();
    expect(mocks.refreshWorkspaces).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain("Change saved, but member list refresh failed.");
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(container.querySelector("button")).toHaveProperty("disabled", true);

    await act(async () => root.unmount());
  });

  it("preserves a denied mutation and fails closed when capability refresh fails", async () => {
    mocks.addWorkspaceMember.mockRejectedValue(new ApiError(403, "private denied detail"));
    mocks.refreshWorkspaces.mockRejectedValue(new Error("private capability detail"));
    const { container, root } = renderDialog();

    await act(async () => root.render(<WorkspaceMembersDialog />));
    await openDialog(container);
    fillUserId("user-3");
    await submitAdd();

    expect(mocks.addWorkspaceMember).toHaveBeenCalledOnce();
    expect(mocks.refreshWorkspaces).toHaveBeenCalledOnce();
    expect(mocks.mutateMembers).not.toHaveBeenCalled();
    expect(document.body.textContent).toContain("Permission changed. This dialog was closed; the denied change was not retried.");
    expect(document.body.textContent).not.toContain("private denied detail");
    expect(document.body.textContent).not.toContain("private capability detail");
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(container.querySelector("button")).toHaveProperty("disabled", true);

    await act(async () => root.unmount());
  });

  it("keeps the rendered dialog usable after a successful mutation and refresh", async () => {
    const { container, root } = renderDialog();

    await act(async () => root.render(<WorkspaceMembersDialog />));
    await openDialog(container);
    fillUserId("user-3");
    await submitAdd();

    expect(mocks.addWorkspaceMember).toHaveBeenCalledOnce();
    expect(mocks.mutateMembers).toHaveBeenCalledOnce();
    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull();
    expect(container.querySelector("button")).toHaveProperty("disabled", false);
    expect(document.body.textContent).not.toContain("could not be completed");

    await act(async () => root.unmount());
  });
});
