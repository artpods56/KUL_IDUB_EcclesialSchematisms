# Example Workflows

This document describes canonical workflows the platform should support. These examples are implementation anchors for API, worker, and UI development.

## Four-Page OCR Comparison

Goal: run multiple OCR methods over four page images and compare their outputs.

```mermaid
flowchart LR
  Images[4 page images] --> Sequence[Page image sequence]
  Sequence --> Mistral[Mistral OCR]
  Sequence --> Tesseract[Tesseract OCR]
  Mistral --> Compare[OCR page comparison]
  Tesseract --> Compare
  Compare --> Metrics[Evaluation metrics]
```

Expected artifacts:

```text
source.page_image x 4
source.page_sequence x 1
ocr.page_result x 8
ocr.document_result x 2
evaluation.metrics x 1
```

The page sequence preserves order:

```text
page 1 -> page 2 -> page 3 -> page 4
```

The OCR operators run in `map` mode over the sequence. The comparison operator runs in `reduce` mode over both OCR result sequences and emits page-level and aggregate metrics.

The user should be able to inspect, for each page:

- source image
- Mistral OCR text and metadata
- Tesseract OCR text and metadata
- similarity score
- confidence and runtime hints where available

## OCR Selection Before Extraction

Goal: compare OCR streams, select one, and pass the selected text into a downstream model.

```mermaid
flowchart LR
  Pages[Page image sequence] --> OCRA[OCR A]
  Pages --> OCRB[OCR B]
  OCRA --> Compare[Compare]
  OCRB --> Compare
  Compare --> Select[Selector]
  OCRA --> Select
  OCRB --> Select
  Select --> Selected[Selected OCR sequence]
```

Selection can be manual or policy-driven:

- always choose provider A
- choose highest confidence
- choose highest similarity to reference
- choose per-page manual override
- keep multiple streams for later comparison

The selector should emit a new artifact sequence rather than mutating the original OCR outputs.

## Contextual Structured Extraction

Goal: extract records from ordered pages while carrying controlled state across the sequence.

```mermaid
flowchart LR
  Pages[Page image sequence] --> Extract[Structured extraction]
  OCR[OCR result sequence] --> Extract
  Schema[Extraction schema] --> Extract
  Template[Prompt or input template] --> Extract
  Policy[Input policy] --> Extract
  Model[Model binding] --> Extract
  Static[Static context] --> Bundle[Input bundle]
  Retrieval[Retrieval node] --> Bundle
  Bundle --> Extract
  Extract --> Records[Record results]
  Extract --> Trace[Input and invocation traces]
```

The important design point is that context providers, prompt/input templates, schemas, model bindings, and policies should be graph-visible inputs. Some are material artifacts. Some may later be compiler-resolved spec nodes.

The extraction operator runs in `stateful_sequence` mode. It consumes an ordered sequence and emits page-level records, a document-level result, input assembly traces, and invocation traces.

This workflow should preserve:

- rendered model input
- attached artifacts
- previous state used for each page
- pruned images or text
- raw model/tool response
- parsed output
- validation result
- retry attempts

## Experiment Matrix

Goal: compare several workflow variants over the same source corpus.

```mermaid
flowchart TD
  Template[Workflow template] --> Grid[Parameter grid]
  Corpus[Input corpus] --> Grid
  Grid --> RunA[Run variant A]
  Grid --> RunB[Run variant B]
  Grid --> RunC[Run variant C]
  RunA --> Compare[Experiment comparison]
  RunB --> Compare
  RunC --> Compare
```

Example dimensions:

```text
ocr.provider:
  - tesseract
  - mistral

layout.model:
  - layoutlmv3
  - provider-layout-api

model.binding:
  - local-model
  - remote-provider

input.policy:
  - previous-page-only
  - sliding-window-3
  - summary-memory

schema:
  - strict
  - relaxed
```

The comparison view should derive:

- run status
- node-run status counts
- artifact counts
- validation errors
- evaluation metrics
- duration
- cost
- selected output artifacts

## DAG Resolution

The API should accept workflow definitions, but the compiler should resolve the DAG before work is queued.

```mermaid
flowchart LR
  Draft[Editable workflow] --> Compile[Compile]
  Compile --> Plan[Concrete node runs]
  Plan --> Queue[Queue executable jobs]
  Queue --> Worker[Worker executes node_run_id]
```

The worker should execute persisted `node_run_id` jobs. It should not re-derive the graph from scratch.

The compiler should reject:

- cycles
- missing required inputs
- incompatible artifact types
- unresolved configuration nodes
- unsupported execution modes
- missing runtime handlers

## JetStream Execution Flow

Goal: use NATS JetStream for durable work and event delivery without making it the source of truth.

```mermaid
flowchart LR
  API[API transaction] --> DB[(Postgres)]
  API --> Outbox[Outbox row]
  Publisher[Outbox publisher] --> JS[JetStream TASKS]
  JS --> Worker[Worker]
  Worker --> DB
  Worker --> EventOutbox[Outbox event rows]
  EventPublisher[Outbox publisher] --> Events[JetStream EVENTS]
```

Postgres stores workflow, run, node-run, artifact, and outbox state. Object storage stores large payloads. JetStream carries tasks and events.

Recommended task subjects:

```text
jobs.workflow.compile.requested
jobs.workflow.run.requested
jobs.node_run.execute.requested
```

Recommended event subjects:

```text
events.workflow_run.*
events.node_run.*
events.artifact.*
```

Implemented lifecycle subjects use the concrete suffixes:

```text
queued
running
succeeded
failed_retryable
failed_permanent
cancelled
created
```

Use dead-letter subjects for permanent failures and retry exhaustion.
