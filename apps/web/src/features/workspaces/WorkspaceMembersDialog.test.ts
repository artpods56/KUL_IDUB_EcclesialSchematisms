import { describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api/client";
import {
  executeMemberMutation,
  MemberListRefreshError,
} from "./workspace-member-mutation";

describe("workspace member mutation authorization recovery", () => {
  it("refetches capabilities once without replaying a denied mutation", async () => {
    const operation = vi.fn().mockRejectedValue(new ApiError(403, "forbidden"));
    const refreshMembers = vi.fn().mockResolvedValue(undefined);
    const refreshWorkspaceCapabilities = vi.fn().mockResolvedValue(undefined);

    await expect(executeMemberMutation(
      operation,
      refreshMembers,
      refreshWorkspaceCapabilities,
    )).rejects.toMatchObject({ status: 403 });

    expect(operation).toHaveBeenCalledOnce();
    expect(refreshMembers).not.toHaveBeenCalled();
    expect(refreshWorkspaceCapabilities).toHaveBeenCalledOnce();
  });

  it("preserves the original 403 when the capability refetch also fails", async () => {
    const denied = new ApiError(403, "forbidden");
    const operation = vi.fn().mockRejectedValue(denied);
    const refreshMembers = vi.fn().mockResolvedValue(undefined);
    const refreshWorkspaceCapabilities = vi.fn().mockRejectedValue(new Error("capability service unavailable"));

    await expect(executeMemberMutation(
      operation,
      refreshMembers,
      refreshWorkspaceCapabilities,
    )).rejects.toBe(denied);

    expect(operation).toHaveBeenCalledOnce();
    expect(refreshMembers).not.toHaveBeenCalled();
    expect(refreshWorkspaceCapabilities).toHaveBeenCalledOnce();
  });

  it("reports a saved change separately when the member refresh fails", async () => {
    const operation = vi.fn().mockResolvedValue(undefined);
    const refreshMembers = vi.fn().mockRejectedValue(new Error("member list unavailable"));
    const refreshWorkspaceCapabilities = vi.fn().mockResolvedValue(undefined);

    await expect(executeMemberMutation(
      operation,
      refreshMembers,
      refreshWorkspaceCapabilities,
    )).rejects.toBeInstanceOf(MemberListRefreshError);

    expect(operation).toHaveBeenCalledOnce();
    expect(refreshMembers).toHaveBeenCalledOnce();
    expect(refreshWorkspaceCapabilities).not.toHaveBeenCalled();
  });
});
