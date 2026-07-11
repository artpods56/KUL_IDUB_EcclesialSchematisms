# Operator Catalog

This catalog describes initial operator families. The backend should use these as typed contracts over artifacts, not as UI-only nodes.

## Operator Contract

An operator spec should declare:

```text
operator_id
operator_version
input ports
output ports
config schema
execution mode
runtime handler binding
display metadata
```

Ports should use artifact types and sequence types:

```text
source.page_image
ArtifactSequence[source.page_image]
ocr.page_result
ArtifactSequence[ocr.page_result]
evaluation.metrics
```

## Source Operators

### PDF Import

Purpose: store an uploaded PDF as a source document.

Inputs:

```text
upload bytes
```

Outputs:

```text
source.document
```

Execution mode:

```text
single
```

Current implementation:

```text
context.static.define -> context.bundle
```

The payload stores `name`, a serializable `context` object, `applies_to`, and
an optional description. `extraction.contextual_structured` accepts this artifact
as an optional `context` input and exposes it in each model input as
`STATIC_CONTEXT` and `CONTEXT_BUNDLE`.

### Page Splitter

Purpose: render or extract pages from a source document.

Inputs:

```text
source.document
```

Outputs:

```text
ArtifactSequence[source.page_image]
```

Execution mode:

```text
map or single
```

Use `single` if the implementation renders the whole document in one process and emits a sequence. Use `map` if page extraction is already represented as independent page jobs.

## OCR Operators

### Mistral OCR

Purpose: run Mistral OCR over ordered page images.

Inputs:

```text
pages: ArtifactSequence[source.page_image]
```

Outputs:

```text
ocr_pages: ArtifactSequence[ocr.page_result]
ocr_document: ocr.document_result
request_traces: ArtifactSequence[ocr.request_trace]
response_traces: ArtifactSequence[ocr.response_trace]
```

Execution mode:

```text
map
```

Example config:

```json
{
  "provider": "mistral",
  "model": "mistral-ocr-latest",
  "include_images": true,
  "include_bounding_boxes": true,
  "language_hints": ["pl", "la"]
}
```

Expected page result payload:

```json
{
  "page_number": 1,
  "text": "...",
  "markdown": "...",
  "blocks": [],
  "tokens": [],
  "confidence": null,
  "image_artifact_id": "artifact-id",
  "provider": "mistral",
  "model": "mistral-ocr-latest"
}
```

The operator should preserve provider metadata and raw responses through invocation traces or payload refs. API keys and credentials must not be serialized into artifacts or traces.

### Local Tesseract OCR

Purpose: run a local OCR engine for baseline comparison.

Inputs:

```text
pages: ArtifactSequence[source.page_image]
```

Outputs:

```text
ocr_pages: ArtifactSequence[ocr.page_result]
ocr_document: ocr.document_result
```

Execution mode:

```text
map
```

Useful config:

```json
{
  "language": "pol+lat",
  "psm": 6,
  "oem": 1
}
```

## Vision And Layout Operators

### Layout Detection

Purpose: identify regions, blocks, tables, headings, or other visual structures.

Inputs:

```text
pages: ArtifactSequence[source.page_image]
```

Outputs:

```text
layout_pages: ArtifactSequence[layout.page_blocks]
```

Execution mode:

```text
map
```

Payloads should include bounding boxes, labels, scores, model metadata, and image artifact refs.

### Vision Classification

Purpose: classify pages, blocks, or regions.

Inputs:

```text
source.page_image
layout.page_blocks
```

Outputs:

```text
vision.classification
```

Execution mode:

```text
single or map
```

## Input Assembly Operators

### Input Policy Definition

Purpose: define how a downstream operator selects, carries, prunes, or bundles inputs.

Outputs:

```text
input.policy
```

Execution mode:

```text
single
```

Example policies:

```text
current-page-only
previous-page-output
sliding-window-3
summary-memory
strip-images-after-current-page
preserve-failed-items
```

### Static Context Provider

Purpose: provide user-authored background material to downstream operators.

Outputs:

```text
context.bundle
```

Execution mode:

```text
single
```

### Retrieval Context Provider

Purpose: retrieve relevant context from indexed artifacts or external corpora.

Inputs:

```text
query artifact
index artifact or corpus ref
```

Outputs:

```text
context.bundle
input.assembly_trace
```

Execution mode:

```text
single or map
```

## Model Invocation Operators

### Generic Model Invocation

Purpose: call a local or remote model and store its output as a typed artifact.

Inputs:

```text
model.binding
model.input
```

Outputs:

```text
model.response
model.invocation_trace
```

Execution mode:

```text
single
```

This is useful for model families that do not yet deserve a specialized operator.

### Structured Extraction

Purpose: produce validated structured results from text, images, layout, or assembled model inputs.

Inputs:

```text
source pages
text or OCR results
schema
model binding
input policy
optional context bundle
optional prompt or input template
```

Outputs:

```text
ArtifactSequence[extraction.record_result]
extraction.document_result
ArtifactSequence[input.assembly_trace]
ArtifactSequence[model.invocation_trace]
```

Execution mode:

```text
stateful_sequence
```

This operator may use an LLM, a local sequence model, a rules engine, or a hybrid pipeline. The operator contract should describe artifacts, not the implementation family.

Current implementation:

- `extraction.contextual_structured` consumes ordered `ocr.page_result`
  sequences, `extraction.schema`, `prompt.template`, `model.binding`,
  `input.policy`, optional `context.bundle`, and optional source page images.
- It emits ordered `extraction.record_result`, `model.input`, and
  `model.response` sequences plus an `extraction.document_result`.
- It supports deterministic local echo extraction and OpenAI-compatible
  chat-completions structured extraction via JSON Schema.

OpenAI-compatible settings live in `model.binding.parameters`:

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key_env_var": "OPENAI_API_KEY",
  "schema_name": "notarius_extraction_result",
  "timeout_seconds": 60,
  "temperature": 0,
  "max_tokens": 2048
}
```

The API key value is read from the worker environment at runtime and must not be
stored in workflow configs, artifacts, traces, or model parameters.

## Evaluation Operators

### OCR Comparison

Purpose: compare two OCR output sequences.

Inputs:

```text
candidate_a: ArtifactSequence[ocr.page_result]
candidate_b: ArtifactSequence[ocr.page_result]
```

Outputs:

```text
evaluation.metrics
```

Execution mode:

```text
reduce
```

Metrics can include similarity ratio, edit distance, page-level differences, missing pages, and confidence summaries.

### Schema Validation

Purpose: validate structured outputs against a schema artifact.

Inputs:

```text
extraction.result
extraction.schema
```

Outputs:

```text
validation.result
evaluation.metrics
```

Execution mode:

```text
single or map
```

## Export Operators

### JSON Export

Inputs:

```text
extraction.document_result
```

Outputs:

```text
export.dataset
```

Execution mode:

```text
single
```

Current implementation:

- `export.dataset` consumes one `extraction.document_result` artifact.
- It emits one `export.dataset` artifact.
- Supported `workflow_node_config.format` values:
  - `json`
  - `jsonl`
  - `csv`
- JSON exports wrap source metadata and records in a typed payload.
- JSONL and CSV exports are stored as downloadable text payloads with
  `content_type` metadata.

### CSV Export

Inputs:

```text
extraction.document_result
```

Outputs:

```text
export.dataset
```

Execution mode:

```text
single
```

## Registration Guidance

Register operators in a catalog that the compiler and API can query.

The first version can be a boring in-process registry:

```text
operator_id -> OperatorSpec
operator_id -> handler
```

Move to dynamic discovery only after there are real external operator packages. The early platform needs stable contracts more than a plugin system.
