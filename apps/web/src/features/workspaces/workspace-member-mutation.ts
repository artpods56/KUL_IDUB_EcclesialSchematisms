import { ApiError } from "@/lib/api/client";

export async function executeMemberMutation(
  operation: () => Promise<unknown>,
  refreshMembers: () => Promise<unknown>,
  refreshWorkspaceCapabilities: () => Promise<unknown>,
): Promise<void> {
  try {
    await operation();
    await refreshMembers();
  } catch (error) {
    if (error instanceof ApiError && error.status === 403) {
      await refreshWorkspaceCapabilities();
    }
    throw error;
  }
}
