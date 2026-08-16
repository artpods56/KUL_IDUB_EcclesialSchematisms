from grafy_core.artifacts import ArtifactTypeKey, ArtifactTypeSpec


OCR_PAGE_RESULT = ArtifactTypeSpec(
    key=ArtifactTypeKey("ocr.page_result", 1),
    title="OCR page result",
)

MISTRAL_OCR_RESPONSE = ArtifactTypeSpec(
    key=ArtifactTypeKey("ocr.mistral_response", 1),
    title="Mistral OCR response",
)

TABLE_FRAGMENT = ArtifactTypeSpec(
    key=ArtifactTypeKey("table.fragment", 1),
    title="OCR table fragment",
)
