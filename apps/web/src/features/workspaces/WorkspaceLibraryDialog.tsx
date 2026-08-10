"use client";

import * as React from "react";
import { Library } from "lucide-react";
import { useRouter } from "next/navigation";
import useSWR from "swr";

import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  deprecateModule,
  importModuleRelease,
  listWorkspaceModules,
  withdrawModule,
  type ModuleLibraryEntry,
  type Workspace,
} from "@/lib/api";
import { useWorkspaces } from "@/hooks/use-api";
import { useAuthSession } from "@/features/auth/AuthSessionBoundary";
import { workbenchGraphPath } from "@/features/workbench/routes";

function canPublish(workspace: Workspace): boolean {
  return workspace.capabilities.includes("publish_module");
}

function canManageLibrary(workspace: Workspace): boolean {
  return workspace.capabilities.includes("manage_module_library");
}

function canCreateGraph(workspace: Workspace): boolean {
  return workspace.capabilities.includes("create_graph");
}

export function WorkspaceLibraryDialog({
  workspace,
  open: controlledOpen,
  onOpenChange,
  triggerLabel = "Workspace library",
  showTrigger = true,
  onOpenSourceGraph,
  onLibraryChanged,
}: {
  workspace: Workspace;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  triggerLabel?: string;
  showTrigger?: boolean;
  onOpenSourceGraph?: (graphId: string) => void;
  onLibraryChanged?: () => void;
}) {
  const router = useRouter();
  const { session } = useAuthSession();
  const { data: workspaces = [] } = useWorkspaces(session.user_id);
  const [uncontrolledOpen, setUncontrolledOpen] = React.useState(false);
  const open = controlledOpen ?? uncontrolledOpen;
  const setOpen = onOpenChange ?? setUncontrolledOpen;
  const [busyId, setBusyId] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [importTarget, setImportTarget] = React.useState<ModuleLibraryEntry | null>(
    null,
  );
  const [destinationId, setDestinationId] = React.useState("");

  const { data, mutate, isLoading } = useSWR(
    open ? ["workspace-modules", workspace.id] : null,
    () => listWorkspaceModules(workspace.id),
  );
  const modules = data?.modules ?? [];
  const releaseCount = (module: ModuleLibraryEntry) => module.releases?.length ?? 0;
  const importDestinations = workspaces.filter(
    (candidate) =>
      candidate.id !== workspace.id && canCreateGraph(candidate),
  );

  const runSteward = async (
    moduleId: string,
    action: "deprecate" | "withdraw",
  ) => {
    setBusyId(moduleId);
    setError(null);
    try {
      if (action === "deprecate") {
        await deprecateModule(workspace.id, moduleId);
      } else {
        const confirmed = window.confirm(
          "Withdraw this Module from the workspace library? Existing Module calls keep resolving their pinned releases. This is not a hard delete.",
        );
        if (!confirmed) return;
        await withdrawModule(workspace.id, moduleId);
      }
      await mutate();
      onLibraryChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Library update failed.");
    } finally {
      setBusyId(null);
    }
  };

  const runImport = async () => {
    if (!importTarget || !destinationId) return;
    const destination = importDestinations.find(
      (candidate) => candidate.id === destinationId,
    );
    if (!destination) return;
    setBusyId(importTarget.id);
    setError(null);
    try {
      const imported = await importModuleRelease(destinationId, {
        source_workspace_id: workspace.id,
        source_module_id: importTarget.id,
        revision: importTarget.current_library_release ?? undefined,
      });
      setImportTarget(null);
      setDestinationId("");
      setOpen(false);
      onLibraryChanged?.();
      // Land on the destination's new source graph (copy-by-value import).
      router.push(workbenchGraphPath(destination.slug, imported.graph_id));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Couldn't import into the destination workspace.",
      );
    } finally {
      setBusyId(null);
    }
  };

  return (
    <>
      {showTrigger ? (
        <button
          type="button"
          className="ns-workspace-button"
          onClick={() => setOpen(true)}
        >
          <Library size={14} /> {triggerLabel}
        </button>
      ) : null}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent style={{ width: "min(720px, 92vw)" }}>
          <DialogHeader>
            <DialogTitle>Workspace library</DialogTitle>
            <DialogDescription>
              Published and deprecated Modules hosted by this workspace. Withdraw
              hides a Module from insert surfaces; pinned calls keep working.
            </DialogDescription>
          </DialogHeader>
          <DialogBody>
            {error ? (
              <p role="alert" className="text-sm text-red-700">
                {error}
              </p>
            ) : null}

            {isLoading ? (
              <p className="text-sm text-muted-foreground">Loading library…</p>
            ) : modules.length === 0 ? (
              <div className="space-y-2 text-sm">
                <p>No published modules in this workspace.</p>
                <ol className="list-decimal pl-5 space-y-1 text-muted-foreground">
                  <li>
                    Open a saved graph and add Module Input / Module Output
                    boundaries.
                  </li>
                  <li>When the contract validates, use Publish release.</li>
                  {canPublish(workspace) ? null : (
                    <li>Publishing requires Editor or Owner access.</li>
                  )}
                </ol>
              </div>
            ) : (
              <ul className="space-y-3 max-h-[50vh] overflow-auto">
                {modules.map((module) => (
                  <li
                    key={module.id}
                    className="rounded-md border border-border px-3 py-2 space-y-2"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="font-medium">{module.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {module.publication_state}
                          {module.current_library_release
                            ? ` · release ${module.current_library_release}`
                            : ""}
                          {" · "}
                          {releaseCount(module)} release
                          {releaseCount(module) === 1 ? "" : "s"}
                        </div>
                        {module.description ? (
                          <p className="text-sm text-muted-foreground mt-1">
                            {module.description}
                          </p>
                        ) : null}
                      </div>
                      <div className="flex flex-wrap gap-2 justify-end">
                        {onOpenSourceGraph ? (
                          <button
                            type="button"
                            className="ns-workspace-button"
                            onClick={() =>
                              onOpenSourceGraph(module.source_graph_id)
                            }
                          >
                            Open source graph
                          </button>
                        ) : null}
                        {canManageLibrary(workspace) &&
                        module.publication_state === "published" ? (
                          <button
                            type="button"
                            className="ns-workspace-button"
                            disabled={busyId === module.id}
                            onClick={() =>
                              void runSteward(module.id, "deprecate")
                            }
                          >
                            Deprecate
                          </button>
                        ) : null}
                        {canManageLibrary(workspace) ? (
                          <button
                            type="button"
                            className="ns-workspace-button"
                            disabled={busyId === module.id}
                            onClick={() =>
                              void runSteward(module.id, "withdraw")
                            }
                          >
                            Withdraw from library
                          </button>
                        ) : null}
                        <button
                          type="button"
                          className="ns-workspace-button"
                          disabled={
                            busyId === module.id ||
                            importDestinations.length === 0
                          }
                          onClick={() => {
                            setImportTarget(module);
                            setDestinationId(importDestinations[0]?.id ?? "");
                          }}
                        >
                          Import into workspace
                        </button>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}

            {importTarget ? (
              <div className="rounded-md border border-border p-3 space-y-2 mt-3">
                <p className="text-sm font-medium">
                  Import “{importTarget.name}” (copy-by-value, not a live link)
                </p>
                <label className="text-sm block space-y-1">
                  Destination workspace
                  <select
                    className="w-full border rounded px-2 py-1"
                    value={destinationId}
                    onChange={(event) =>
                      setDestinationId(event.currentTarget.value)
                    }
                  >
                    {importDestinations.map((candidate) => (
                      <option key={candidate.id} value={candidate.id}>
                        {candidate.name}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="flex gap-2 justify-end">
                  <button
                    type="button"
                    className="ns-workspace-button"
                    onClick={() => setImportTarget(null)}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="ns-workspace-button ns-workspace-button--primary"
                    disabled={!destinationId || busyId === importTarget.id}
                    onClick={() => void runImport()}
                  >
                    Confirm import
                  </button>
                </div>
              </div>
            ) : null}
          </DialogBody>
        </DialogContent>
      </Dialog>
    </>
  );
}
