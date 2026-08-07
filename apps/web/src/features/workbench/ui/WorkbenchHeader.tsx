"use client";

import * as stylex from "@stylexjs/stylex";
import {
  ChevronDown,
  LoaderCircle,
  Monitor,
  Moon,
  Save,
  Sun,
} from "lucide-react";

import type { ThemePreference } from "@/components/theme";
import { tokens } from "@/lib/stylex/tokens.stylex";

export type WorkbenchHeaderGraphStatus =
  | "ready"
  | "incomplete"
  | "error"
  | "running";

export interface WorkbenchHeaderProps {
  graphName: string;
  activeGraphRevision: number | null;
  isDirty: boolean;
  saving: boolean;
  saveDisabled: boolean;
  nodeCount: number;
  edgeCount: number;
  graphStatus: WorkbenchHeaderGraphStatus;
  canvasStatusMessage: string;
  themePreference: ThemePreference;
  onToggleGraphBrowser: () => void;
  onGraphNameChange: (graphName: string) => void;
  onSaveGraph: () => void;
  onCycleTheme: () => void;
}

const s = stylex.create({
  topBar: {
    position: "absolute",
    zIndex: 20,
    top: "13px",
    left: "13px",
    right: "13px",
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: "12px",
    pointerEvents: "none",
  },
  chrome: {
    display: "flex",
    alignItems: "center",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorder,
    borderRadius: "8px",
    backgroundColor: tokens.colorChrome,
    pointerEvents: "auto",
  },
  identity: {
    minHeight: "43px",
    gap: "9px",
    padding: "6px 9px 6px 11px",
    borderRadius: "12px",
    boxShadow: tokens.shadowNode,
  },
  identityCopy: {
    width: "min(230px, 42vw)",
    minWidth: 0,
    display: "grid",
    gap: "1px",
  },
  identityMenu: {
    minWidth: 0,
    display: "flex",
    alignItems: "center",
    gap: "5px",
    padding: 0,
    borderWidth: 0,
    backgroundColor: "transparent",
    color: tokens.colorMuted,
    cursor: "pointer",
    textAlign: "left",
  },
  brand: {
    color: tokens.colorMuted,
    fontSize: tokens.fontSizeXs,
    fontWeight: 800,
    letterSpacing: "0.16em",
    lineHeight: 1.1,
  },
  saveState: {
    overflow: "hidden",
    color: tokens.colorSubtle,
    fontSize: "10px",
    lineHeight: 1.1,
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  saveStateDirty: { color: tokens.colorWarning },
  workflowName: {
    width: "100%",
    minWidth: 0,
    padding: 0,
    overflow: "hidden",
    borderWidth: 0,
    outline: "none",
    backgroundColor: "transparent",
    color: tokens.colorTextEmphasis,
    fontFamily: "inherit",
    fontSize: tokens.fontSizeMd,
    fontWeight: 700,
    lineHeight: 1.2,
    textOverflow: "ellipsis",
  },
  identityDivider: {
    width: "1px",
    height: "28px",
    flexShrink: 0,
    backgroundColor: tokens.colorDivider,
  },
  identityActions: {
    display: "flex",
    alignItems: "center",
    gap: "2px",
  },
  identityAction: {
    width: "29px",
    paddingInline: 0,
  },
  identityActionActive: {
    backgroundColor: {
      default: tokens.colorAccentSoft,
      ":hover": tokens.colorAccentSoft,
    },
    color: tokens.colorAccent,
  },
  identityStats: {
    minWidth: {
      default: "116px",
      "@media (max-width: 520px)": "52px",
    },
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "5px",
    flexShrink: 0,
    color: tokens.colorSubtle,
    fontSize: "10px",
    fontVariantNumeric: "tabular-nums",
    whiteSpace: "nowrap",
  },
  identityStatValue: {
    color: tokens.colorTextEmphasis,
    fontWeight: 750,
  },
  identityStatLabel: {
    display: {
      default: "inline",
      "@media (max-width: 520px)": "none",
    },
  },
  identityStatSeparator: { color: tokens.colorDivider },
  graphStatusDot: {
    width: "5px",
    height: "5px",
    flexShrink: 0,
    borderRadius: "99px",
    backgroundColor: tokens.colorSuccess,
  },
  graphStatusDotIncomplete: { backgroundColor: tokens.colorWarning },
  graphStatusDotError: { backgroundColor: tokens.colorDanger },
  graphStatusDotRunning: { backgroundColor: tokens.colorInfo },
  toolButton: {
    height: "31px",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "6px",
    paddingInline: "9px",
    borderWidth: 0,
    borderRadius: "5px",
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorHover,
      ":disabled": "transparent",
    },
    color: {
      default: tokens.colorMuted,
      ":disabled": tokens.colorTextDisabled,
    },
    cursor: { default: "pointer", ":disabled": "not-allowed" },
    fontSize: tokens.fontSizeSm,
  },
  spinner: {
    animationName: "ns-spin",
    animationDuration: "900ms",
    animationIterationCount: "infinite",
    animationTimingFunction: "linear",
  },
});

export function WorkbenchHeader({
  graphName,
  activeGraphRevision,
  isDirty,
  saving,
  saveDisabled,
  nodeCount,
  edgeCount,
  graphStatus,
  canvasStatusMessage,
  themePreference,
  onToggleGraphBrowser,
  onGraphNameChange,
  onSaveGraph,
  onCycleTheme,
}: WorkbenchHeaderProps) {
  return (
    <div {...stylex.props(s.topBar)}>
      <div {...stylex.props(s.chrome, s.identity)}>
        <span {...stylex.props(s.identityCopy)}>
          <button
            type="button"
            {...stylex.props(s.identityMenu)}
            onClick={onToggleGraphBrowser}
          >
            <span {...stylex.props(s.brand)}>NOTARIUS</span>
            <span
              {...stylex.props(
                s.saveState,
                isDirty ? s.saveStateDirty : null,
              )}
            >
              {saving
                ? "saving…"
                : activeGraphRevision !== null
                  ? isDirty
                    ? "unsaved"
                    : `saved · r${activeGraphRevision}`
                  : "not saved"}
            </span>
            <ChevronDown size={11} />
          </button>
          <input
            aria-label="Graph name"
            value={graphName}
            maxLength={160}
            {...stylex.props(s.workflowName)}
            onChange={(event) => onGraphNameChange(event.currentTarget.value)}
          />
        </span>
        <span {...stylex.props(s.identityDivider)} />
        <span
          aria-label={`${nodeCount} node${nodeCount === 1 ? "" : "s"}, ${edgeCount} connection${edgeCount === 1 ? "" : "s"}. ${canvasStatusMessage}`}
          title={canvasStatusMessage}
          {...stylex.props(s.identityStats)}
        >
          <span
            aria-hidden="true"
            {...stylex.props(
              s.graphStatusDot,
              graphStatus === "error" ? s.graphStatusDotError : null,
              graphStatus === "running" ? s.graphStatusDotRunning : null,
              graphStatus === "incomplete"
                ? s.graphStatusDotIncomplete
                : null,
            )}
          />
          <span>
            <span {...stylex.props(s.identityStatValue)}>{nodeCount}</span>{" "}
            <span {...stylex.props(s.identityStatLabel)}>
              node{nodeCount === 1 ? "" : "s"}
            </span>
          </span>
          <span
            aria-hidden="true"
            {...stylex.props(s.identityStatSeparator)}
          >
            ·
          </span>
          <span>
            <span {...stylex.props(s.identityStatValue)}>{edgeCount}</span>{" "}
            <span {...stylex.props(s.identityStatLabel)}>
              connection{edgeCount === 1 ? "" : "s"}
            </span>
          </span>
        </span>
        <span {...stylex.props(s.identityDivider)} />
        <span {...stylex.props(s.identityActions)}>
          <button
            type="button"
            aria-label={
              saving
                ? "Saving graph"
                : activeGraphRevision !== null && !isDirty
                  ? "Graph saved"
                  : "Save graph"
            }
            disabled={saveDisabled}
            title={
              activeGraphRevision !== null && !isDirty
                ? "All changes are saved"
                : "Save graph"
            }
            {...stylex.props(
              s.toolButton,
              s.identityAction,
              isDirty ? s.identityActionActive : null,
            )}
            onClick={onSaveGraph}
          >
            {saving ? (
              <LoaderCircle size={13} {...stylex.props(s.spinner)} />
            ) : (
              <Save size={13} />
            )}
          </button>
          <button
            type="button"
            aria-label={
              themePreference === "light"
                ? "Switch to dark theme"
                : themePreference === "dark"
                  ? "Switch to system theme"
                  : "Switch to light theme"
            }
            title={
              themePreference === "light"
                ? "Light theme"
                : themePreference === "dark"
                  ? "Dark theme"
                  : "System theme"
            }
            {...stylex.props(s.toolButton, s.identityAction)}
            onClick={onCycleTheme}
          >
            {themePreference === "light" ? (
              <Sun size={13} />
            ) : themePreference === "dark" ? (
              <Moon size={13} />
            ) : (
              <Monitor size={13} />
            )}
          </button>
        </span>
      </div>
    </div>
  );
}
