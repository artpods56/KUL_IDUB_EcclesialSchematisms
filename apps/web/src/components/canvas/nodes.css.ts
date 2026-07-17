/** Color per artifact-type family, used for port dots and edge strokes. */
export const ARTIFACT_TYPE_COLOR: Record<string, string> = {
  "image.raster": "light-dark(#2a9d7c, #43c59e)",
  "ocr.page_result": "light-dark(#7b63c9, #9a7cf2)",
  "ocr.mistral_response": "light-dark(#7b63c9, #9a7cf2)",
  "ocr.document_result": "light-dark(#7b63c9, #9a7cf2)",
  "ocr.request_trace": "light-dark(#c9920f, #fbbf24)",
  "ocr.response_trace": "light-dark(#c9920f, #fbbf24)",
  "extraction.record_result": "light-dark(#c35a91, #f472b6)",
  "extraction.document_result": "light-dark(#c35a91, #f472b6)",
  "evaluation.metrics": "light-dark(#c9920f, #fbbf24)",
  "scalar.integer": "light-dark(#4590c7, #57a5ef)",
  "scalar.text": "light-dark(#b46735, #e88a50)",
  "table.fragment": "light-dark(#4590c7, #57a5ef)",
  "table.page": "light-dark(#3c9aa1, #4bc0c8)",
  "tabular.csv_bundle": "light-dark(#c08549, #f0a65a)",
  "export.dataset": "light-dark(#8663c9, #a78bfa)",
};
