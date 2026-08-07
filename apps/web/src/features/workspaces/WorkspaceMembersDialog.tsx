"use client";

import * as React from "react";
import { UserPlus } from "lucide-react";

import { ApiError } from "@/lib/api/client";
import {
  addWorkspaceMember,
  changeWorkspaceMemberRole,
  removeWorkspaceMember,
  type WorkspaceRole,
} from "@/lib/api";
import { useWorkspaceMembers } from "@/hooks/use-api";
import { useAuthSession } from "@/features/auth/AuthSessionBoundary";
import { useWorkspaceContext } from "./WorkspaceLayout";
import { executeMemberMutation } from "./workspace-member-mutation";
import { Dialog, DialogBody, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const roles: readonly WorkspaceRole[] = ["viewer", "editor", "owner"];

function operationError(error: unknown): string {
  if (!(error instanceof ApiError)) return "The member change could not be completed.";
  if (error.status === 403) return "Permission changed. Refreshing workspace capabilities; the denied change was not retried.";
  if (error.status === 404) return "Workspace or user UUID was not found.";
  if (error.status === 409) return "The member state changed elsewhere. Refresh the list and try again.";
  return `Member change failed (${error.status}).`;
}

export function WorkspaceMembersDialog() {
  const { session } = useAuthSession();
  const { workspace, refreshWorkspaces } = useWorkspaceContext();
  const [open, setOpen] = React.useState(false);
  const { data: members, error, mutate } = useWorkspaceMembers(session.user_id, open ? workspace.id : undefined);
  const [userId, setUserId] = React.useState("");
  const [role, setRole] = React.useState<WorkspaceRole>("viewer");
  const [busyKey, setBusyKey] = React.useState<string | null>(null);
  const [message, setMessage] = React.useState<string | null>(null);

  const runMutation = async (key: string, operation: () => Promise<unknown>) => {
    setBusyKey(key);
    setMessage(null);
    try {
      await executeMemberMutation(operation, mutate, refreshWorkspaces);
    } catch (caught) {
      setMessage(operationError(caught));
    } finally {
      setBusyKey(null);
    }
  };

  const addMember = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const normalizedUserId = userId.trim();
    if (!normalizedUserId) return;
    await runMutation("add", () => addWorkspaceMember(workspace.id, { user_id: normalizedUserId, role }));
    setUserId("");
  };

  return (
    <>
      <button type="button" className="ns-workspace-button" onClick={() => setOpen(true)}>
        <UserPlus size={14} /> Manage members
      </button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Members · {workspace.name}</DialogTitle>
            <DialogDescription>Member changes use user UUIDs. Server capabilities remain authoritative.</DialogDescription>
          </DialogHeader>
          <DialogBody>
            <form className="ns-member-form" onSubmit={(event) => void addMember(event)}>
              <label>User UUID<input value={userId} onChange={(event) => setUserId(event.target.value)} placeholder="xxxxxxxx-xxxx-…" /></label>
              <label>Role<select value={role} onChange={(event) => setRole(event.target.value as WorkspaceRole)}>{roles.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>
              <button className="ns-workspace-button ns-workspace-button--primary" type="submit" disabled={busyKey !== null}>Add</button>
            </form>
            {message ? <p className="ns-member-message" role="status">{message}</p> : null}
            {error ? <p className="ns-member-message" role="status">Members could not be loaded.</p> : null}
            {!members ? <p className="ns-member-empty">Loading members…</p> : members.length === 0 ? <p className="ns-member-empty">No members returned.</p> : (
              <div className="ns-member-list">
                {members.map((member) => (
                  <div className="ns-member-row" key={member.user.id}>
                    <div><strong>{member.user.display_name ?? member.user.email ?? member.user.id}</strong><span>{member.user.id}</span></div>
                    <select aria-label={`Role for ${member.user.id}`} value={member.role} disabled={busyKey === member.user.id} onChange={(event) => void runMutation(member.user.id, () => changeWorkspaceMemberRole(workspace.id, member.user.id, { role: event.target.value as WorkspaceRole }))}>{roles.map((option) => <option key={option} value={option}>{option}</option>)}</select>
                    <button type="button" className="ns-member-remove" disabled={busyKey === member.user.id} onClick={() => void runMutation(member.user.id, () => removeWorkspaceMember(workspace.id, member.user.id))}>Remove</button>
                  </div>
                ))}
              </div>
            )}
          </DialogBody>
        </DialogContent>
      </Dialog>
    </>
  );
}
