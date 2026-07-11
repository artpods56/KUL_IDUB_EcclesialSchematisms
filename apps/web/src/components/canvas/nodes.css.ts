import { tokens } from "@/lib/stylex/tokens.stylex";

/** Color per artifact-type family, used for port dots and edge strokes. */
export const ARTIFACT_TYPE_COLOR: Record<string, string> = {
  "source.page_image": "light-dark(#2a9d7c, #43c59e)",
  "source.document": "light-dark(#2a9d7c, #43c59e)",
  "ocr.page_result": "light-dark(#7b63c9, #9a7cf2)",
  "ocr.mistral_response": "light-dark(#7b63c9, #9a7cf2)",
  "ocr.document_result": "light-dark(#7b63c9, #9a7cf2)",
  "ocr.request_trace": "light-dark(#c9920f, #fbbf24)",
  "ocr.response_trace": "light-dark(#c9920f, #fbbf24)",
  "extraction.record_result": "light-dark(#c35a91, #f472b6)",
  "extraction.document_result": "light-dark(#c35a91, #f472b6)",
  "evaluation.metrics": "light-dark(#c9920f, #fbbf24)",
  "scalar.integer": "light-dark(#4590c7, #57a5ef)",
  "arithmetic.result": "light-dark(#8663c9, #a78bfa)",
  "table.fragment": "light-dark(#4590c7, #57a5ef)",
  "table.page": "light-dark(#3c9aa1, #4bc0c8)",
  "tabular.csv_bundle": "light-dark(#c08549, #f0a65a)",
  "export.dataset": "light-dark(#8663c9, #a78bfa)",
};

/** Resolved projection-edge label presentation using theme tokens. */
export function projectionEdgePresentation() {
  return {
    labelStyle: {
      fill: tokens.colorTextEmphasis,
      fontSize: 11,
      fontWeight: 700,
    },
    labelShowBg: true,
    labelBgStyle: {
      fill: tokens.colorSurfaceRaised,
      fillOpacity: 1,
      stroke: tokens.colorAccentBorder,
      strokeWidth: 1,
    },
    labelBgPadding: [5, 3] as [number, number],
    labelBgBorderRadius: 4,
  };
}
