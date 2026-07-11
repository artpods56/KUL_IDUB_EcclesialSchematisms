import * as stylex from "@stylexjs/stylex";
import { tokens } from "@/lib/stylex/tokens.stylex";

/** Shared canvas-node styles (StyleX). */

export const nodeShell = stylex.create({
  shell: {
    width: "240px",
    borderRadius: tokens.radiusLg,
    backgroundColor: tokens.colorSurface,
    border: `1px solid ${tokens.colorBorder}`,
    boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
    fontFamily: "inherit",
    color: tokens.colorText,
    fontSize: tokens.fontSizeSm,
    overflow: "visible",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: tokens.space2,
    padding: `${tokens.space2} ${tokens.space3}`,
    borderRadius: `${tokens.radiusLg} ${tokens.radiusLg} 0 0`,
    backgroundColor: tokens.colorSurfaceRaised,
    borderBottom: `1px solid ${tokens.colorBorder}`,
  },
  title: {
    fontSize: tokens.fontSizeSm,
    fontWeight: 700,
    color: tokens.colorText,
    display: "flex",
    alignItems: "center",
    gap: tokens.space2,
  },
  subtitle: {
    fontSize: tokens.fontSizeXs,
    color: tokens.colorSubtle,
  },
  body: {
    padding: tokens.space3,
    display: "flex",
    flexDirection: "column",
    gap: tokens.space2,
  },
  row: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: tokens.space2,
    fontSize: tokens.fontSizeXs,
  },
  port: {
    fontSize: tokens.fontSizeXs,
    color: tokens.colorMuted,
    paddingBlock: "2px",
  },
  thumb: {
    height: "44px",
    width: "100%",
    borderRadius: tokens.radiusSm,
    objectFit: "cover",
    backgroundColor: "#0a0e1c",
    border: `1px solid ${tokens.colorBorder}`,
  },
  badge: { display: "inline-flex", gap: tokens.space1 },
  empty: {
    fontSize: tokens.fontSizeXs,
    color: tokens.colorSubtle,
    fontStyle: "italic",
    textAlign: "center",
    padding: tokens.space2,
  },
  selected: { borderColor: tokens.colorAccent },
});

export const handle = stylex.create({
  base: {
    width: "12px",
    height: "12px",
    borderRadius: "9999px",
    backgroundColor: tokens.colorAccent,
    border: `2px solid ${tokens.colorSurface}`,
  },
});

/** Color per artifact-type family, used for port dots/labels. */
export const ARTIFACT_TYPE_COLOR: Record<string, string> = {
  "source.page_image": "#43c59e",
  "source.document": "#43c59e",
  "ocr.page_result": "#9a7cf2",
  "ocr.mistral_response": "#9a7cf2",
  "ocr.document_result": "#9a7cf2",
  "ocr.request_trace": "#fbbf24",
  "ocr.response_trace": "#fbbf24",
  "extraction.record_result": "#f472b6",
  "extraction.document_result": "#f472b6",
  "evaluation.metrics": "#fbbf24",
  "scalar.integer": "#57a5ef",
  "arithmetic.result": "#a78bfa",
  "table.fragment": "#57a5ef",
  "table.page": "#4bc0c8",
  "tabular.csv_bundle": "#f0a65a",
  "export.dataset": "#a78bfa",
};
