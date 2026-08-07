import { describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api/client";
import { executeMemberMutation } from "./workspace-member-mutation";

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
});
