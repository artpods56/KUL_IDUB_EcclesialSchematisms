"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";

import {
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { publishModuleRelease } from "@/lib/api";
import { tokens } from "@/lib/stylex/tokens.stylex";

const s = stylex.create({
  form: {
    display: "flex",
    flexDirection: "column",
    gap: tokens.space3,
  },
  field: {
    display: "grid",
    gap: "5px",
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    fontWeight: 700,
    textTransform: "uppercase",
  },
  control: {
    width: "100%",
    minHeight: "32px",
    padding: `6px ${tokens.space2}`,
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: tokens.radiusSm,
    outline: "none",
    backgroundColor: tokens.colorSurface,
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    fontWeight: 400,
    textTransform: "none",
    boxSizing: "border-box",
  },
  textarea: {
    minHeight: "72px",
    resize: "vertical",
  },
  error: {
    margin: 0,
    color: "light-dark(#b42318, #f97066)",
    fontSize: tokens.fontSizeSm,
    fontWeight: 400,
    textTransform: "none",
  },
  actions: {
    display: "flex",
    justifyContent: "flex-end",
    gap: tokens.space2,
    marginTop: tokens.space1,
  },
});

export function PublishModuleDialog({
  open,
  onOpenChange,
  workspaceId,
  sourceGraphId,
  graphName,
  revision,
  onPublished,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workspaceId: string;
  sourceGraphId: string;
  graphName: string;
  revision: number;
  onPublished?: () => void;
}) {
  const [name, setName] = React.useState(graphName);
  const [description, setDescription] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (open) {
      setName(graphName);
      setDescription("");
      setError(null);
    }
  }, [graphName, open]);

  const confirm = async () => {
    setBusy(true);
    setError(null);
    try {
      const module = await publishModuleRelease(workspaceId, {
        source_graph_id: sourceGraphId,
        revision,
        name: name.trim() || graphName,
        description: description.trim() || null,
      });
      onOpenChange(false);
      onPublished?.();
      window.alert(
        `Published release ${module.current_library_release} to workspace library.`,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Couldn't publish this release. Fix the contract and retry.",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Publish release</DialogTitle>
          <DialogDescription>
            Offer revision {revision} of this source graph as an immutable Module
            release. Callers pin the release and do not auto-track tip changes.
          </DialogDescription>
        </DialogHeader>
        <DialogBody>
          <div {...stylex.props(s.form)}>
            <label {...stylex.props(s.field)}>
              Module name
              <input
                {...stylex.props(s.control)}
                value={name}
                onChange={(event) => setName(event.currentTarget.value)}
              />
            </label>
            <label {...stylex.props(s.field)}>
              Description (optional)
              <textarea
                {...stylex.props(s.control, s.textarea)}
                rows={3}
                value={description}
                onChange={(event) => setDescription(event.currentTarget.value)}
              />
            </label>
            {error ? (
              <p role="alert" {...stylex.props(s.error)}>
                {error}
              </p>
            ) : null}
            <div {...stylex.props(s.actions)}>
              <button
                type="button"
                className="ns-workspace-button"
                onClick={() => onOpenChange(false)}
                disabled={busy}
              >
                Cancel
              </button>
              <button
                type="button"
                className="ns-workspace-button ns-workspace-button--primary"
                disabled={busy || name.trim() === ""}
                onClick={() => void confirm()}
              >
                {busy ? "Publishing…" : "Confirm publish"}
              </button>
            </div>
          </div>
        </DialogBody>
      </DialogContent>
    </Dialog>
  );
}
