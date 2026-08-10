"use client";

import * as React from "react";
import * as stylex from "@stylexjs/stylex";
import { Popover } from "@base-ui/react/popover";
import { ChevronDown, X } from "lucide-react";

import { tokens } from "@/lib/stylex/tokens.stylex";
import { useOptionalCanvasGridSettings } from "../canvas-grid-settings";
import {
  EDGE_SELECTOR_HEIGHT_CELLS,
  EDGE_SELECTOR_WIDTH_CELLS,
  GRID_CELL_SIZE_DEFAULT,
  edgeSelectorBlockSize,
} from "../grid-layout";

/** Visual height of the pill — matches workflow port tabs. */
const EDGE_SELECTOR_PILL_HEIGHT = 24;

const s = stylex.create({
  positioner: {
    position: "absolute",
    display: "grid",
    placeItems: "center",
    pointerEvents: "all",
    zIndex: 10,
  },
  /**
   * Full 2×1 footprint is the bend grab — much larger than the painted pill.
   * Menu / remove sit above with their own hit targets.
   */
  bendHandle: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    padding: 0,
    borderWidth: 0,
    borderRadius: "8px",
    backgroundColor: "transparent",
    cursor: "grab",
    touchAction: "none",
    zIndex: 1,
  },
  bendHandleDragging: {
    cursor: "grabbing",
  },
  block: {
    position: "relative",
    boxSizing: "border-box",
    display: "grid",
    alignItems: "stretch",
    overflow: "visible",
    width: "100%",
    height: `${EDGE_SELECTOR_PILL_HEIGHT}px`,
    borderWidth: 0,
    borderRadius: "9999px",
    backgroundColor: tokens.colorSurfaceMuted,
    boxShadow: tokens.shadowNode,
    pointerEvents: "none",
    zIndex: 2,
  },
  blockSelected: {
    boxShadow: tokens.shadowNodeSelected,
    outlineWidth: 1,
    outlineStyle: "solid",
    outlineColor: tokens.colorAccentBorder,
  },
  blockDragging: {
    backgroundColor: tokens.colorAccentSoft,
  },
  blockDisabled: {
    opacity: 0.78,
  },
  labelFace: {
    minWidth: 0,
    width: "100%",
    height: "100%",
    display: "inline-flex",
    alignItems: "center",
    gap: "4px",
    paddingLeft: "14px",
    paddingRight: "44px",
    borderWidth: 0,
    color: tokens.colorTextEmphasis,
    fontSize: tokens.fontSizeXs,
    fontWeight: 600,
    pointerEvents: "none",
  },
  editLabel: {
    minWidth: 0,
    flex: "1 1 auto",
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
    textAlign: "left",
  },
  menuButton: {
    position: "absolute",
    top: "50%",
    right: "22px",
    width: "20px",
    height: "20px",
    display: "grid",
    placeItems: "center",
    padding: 0,
    borderWidth: 0,
    borderRadius: "9999px",
    transform: "translateY(-50%)",
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorHoverStrong,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorTextEmphasis },
    cursor: "pointer",
    pointerEvents: "all",
    zIndex: 3,
  },
  removeButton: {
    position: "absolute",
    top: "50%",
    right: "2px",
    width: "20px",
    height: "20px",
    display: "grid",
    placeItems: "center",
    padding: 0,
    borderWidth: 0,
    borderRadius: "9999px",
    transform: "translateY(-50%)",
    backgroundColor: {
      default: "transparent",
      ":hover": tokens.colorDangerHover,
    },
    color: { default: tokens.colorSubtle, ":hover": tokens.colorDanger },
    cursor: "pointer",
    pointerEvents: "all",
    zIndex: 3,
  },
  popup: {
    width: "300px",
    overflow: "hidden",
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: tokens.colorBorderStrong,
    borderRadius: "7px",
    backgroundColor: tokens.colorSurface,
    boxShadow: tokens.shadowNodeSelected,
    color: tokens.colorText,
    zIndex: 50,
  },
});

export type EdgeSelectorBendHandlers = Pick<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  | "onClick"
  | "onDoubleClick"
  | "onKeyDown"
  | "onPointerDown"
  | "onPointerMove"
  | "onPointerUp"
  | "onPointerCancel"
  | "onLostPointerCapture"
>;

export interface EdgeSelectorBlockProps {
  anchor: { x: number; y: number };
  selected?: boolean;
  disabled?: boolean;
  label: string;
  bendAriaLabel: string;
  bendDragging?: boolean;
  bendHandlers: EdgeSelectorBendHandlers;
  editAriaLabel: string;
  editTitle: string;
  editDisabled?: boolean;
  removeAriaLabel: string;
  onRemove: () => void;
  children: React.ReactNode;
}

/** Midpoint feed selector — 2 cells wide × 1 cell tall in flow coordinates. */
export function EdgeSelectorBlock({
  anchor,
  selected = false,
  disabled = false,
  label,
  bendAriaLabel,
  bendDragging = false,
  bendHandlers,
  editAriaLabel,
  editTitle,
  editDisabled = false,
  removeAriaLabel,
  onRemove,
  children,
}: EdgeSelectorBlockProps) {
  const grid = useOptionalCanvasGridSettings();
  const cellSize = grid?.settings.cellSize ?? GRID_CELL_SIZE_DEFAULT;
  const { width, height } = edgeSelectorBlockSize(cellSize);

  return (
    <div
      className="nodrag nopan nowheel"
      data-testid="edge-selector-block"
      data-width-cells={EDGE_SELECTOR_WIDTH_CELLS}
      data-height-cells={EDGE_SELECTOR_HEIGHT_CELLS}
      data-cell-size={cellSize}
      style={{
        width,
        height,
        transform: `translate(-50%, -50%) translate(${anchor.x}px, ${anchor.y}px)`,
      }}
      {...stylex.props(s.positioner)}
    >
      <button
        type="button"
        aria-label={bendAriaLabel}
        aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown Home"
        title="Drag to bend · arrow keys nudge · double-click or Home to reset"
        disabled={disabled}
        {...stylex.props(
          s.bendHandle,
          bendDragging ? s.bendHandleDragging : null,
        )}
        {...bendHandlers}
      />
      <div
        {...stylex.props(
          s.block,
          selected ? s.blockSelected : null,
          bendDragging ? s.blockDragging : null,
          disabled ? s.blockDisabled : null,
        )}
      >
        <span {...stylex.props(s.labelFace)}>
          <span title={label} {...stylex.props(s.editLabel)}>
            {label}
          </span>
        </span>
        <Popover.Root>
          <Popover.Trigger
            type="button"
            disabled={editDisabled}
            aria-label={editAriaLabel}
            title={editTitle}
            {...stylex.props(s.menuButton)}
            onPointerDown={(event) => event.stopPropagation()}
          >
            <ChevronDown size={11} />
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Positioner side="bottom" align="center" sideOffset={7}>
              <Popover.Popup
                className="nodrag nopan nowheel"
                {...stylex.props(s.popup)}
              >
                {children}
              </Popover.Popup>
            </Popover.Positioner>
          </Popover.Portal>
        </Popover.Root>
        <button
          type="button"
          aria-label={removeAriaLabel}
          title="Remove connection"
          {...stylex.props(s.removeButton)}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            onRemove();
          }}
        >
          <X size={12} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}
