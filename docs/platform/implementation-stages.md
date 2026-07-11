# Implementation Stages

This plan is intended for a coding agent working on a separate branch or worktree.

## Stage 0: Repo Safety And Inventory

Goal: understand the current state without moving code.

Tasks:

1. Create a dedicated branch or worktree.
2. Record `git status --short`.
3. Inventory current Notarius extraction modules:
   - dataset processors
   - item processors
   - sequence state
   - context providers
   - context strategies
   - message builder
   - LLM engine ports
   - OCR and ML model adapters
4. Inventory Stacking backend modules that can be adapted.
5. Run the current focused tests and record baseline failures.

Gate:

- No behavior changes.
- Dirty user changes are preserved.
- Baseline status is documented.

## Stage 1: Workspace Skeleton

Goal: establish the target package layout.

Tasks:

1. Create or normalize workspace packages:
   - `apps/api`
   - `apps/worker`
   - `apps/dagster`
   - `libs/core`
   - `libs/storage`
   - `libs/persistence`
   - `libs/messaging`
   - `libs/shared`
   - `libs/llm`
   - `libs/schematisms`
2. Ensure each package imports.
3. Add architecture boundary tests.
4. Keep compatibility imports in old paths if needed.

Gate:

- `python -c "import notarius_core"` works.
- Boundary tests exist.
- No domain behavior has been moved yet.

## Stage 2: Generic Core Domain

Goal: define the platform vocabulary before moving runtime behavior.

Tasks:

1. Add core models:
   - `WorkflowDefinition`
   - `WorkflowVersion`
   - `WorkflowRun`
   - `OperatorSpec` or `NodeSpec`
   - `NodeRun`
   - `Artifact`
   - `ArtifactSequence`
   - `ArtifactRef`
   - `InputAssemblyTrace`
   - `InvocationTrace`
   - `ExecutionMode`
2. Add ports for repositories, storage, invocation, and node handlers.
3. Add focused model tests.

Gate:

- Core has no FastAPI, SQLAlchemy, NATS, Dagster, or provider-specific imports.
- Models are generic and not LLM-centered.

## Stage 3: Storage And Persistence

Goal: persist artifacts, runs, and payload refs.

Tasks:

1. Adapt generic object-store code from Stacking.
2. Add local and S3/MinIO storage adapters.
3. Add SQLAlchemy mappings for:
   - workflows
   - workflow versions
   - workflow runs
   - node runs
   - artifacts
   - artifact sequences
   - outbox messages
4. Add repository implementations.
5. Add `SqlAlchemyUnitOfWork`.

Gate:

- Repository tests cover create/read/update paths.
- Artifact payloads are stored out of row.
- Unit of work rolls back on exceptions.

## Stage 4: Messaging And Worker Runtime

Goal: run concrete node jobs through JetStream.

Tasks:

1. Add message contracts:
   - workflow compile requested
   - workflow run requested
   - node run execute requested
   - dead-letter message
2. Add subject constants.
3. Add JetStream client/provider.
4. Add worker streams:
   - `TASKS`
   - `EVENTS`
   - `LIVE_DELTAS`
   - `DLQ`
5. Add outbox publisher.
6. Add a fake node-run handler and worker subscriber.

Gate:

- Worker consumes a fake node-run job.
- NodeRun status transitions are persisted.
- Retryable and permanent failures are handled differently.

## Stage 5: FastAPI Backend

Goal: expose the platform over HTTP.

Tasks:

1. Add `create_app()`.
2. Add health endpoints:
   - `/health/live`
   - `/health/ready`
3. Add source upload endpoint.
4. Add workflow definition endpoints.
5. Add workflow version endpoint.
6. Add workflow run endpoint.
7. Add node run and artifact inspection endpoints.
8. Add object payload or preview access endpoints.

Gate:

- API starts.
- Readiness checks DB and NATS.
- OpenAPI schema is valid.
- A smoke test can create a workflow run request.

## Stage 6: Workflow Compiler

Goal: turn editable graphs into concrete execution plans.

Current implementation status:

- Backend-owned workflow templates are available through:
  - `GET /v1/workflow-templates`
  - `POST /v1/workflow-templates/{template_id}/materialize`
  - `POST /v1/workflow-templates/{template_id}/launch`
- Draft workflow graphs can be checked without persistence through
  `POST /v1/workflows/validate`. The response reports compile validity,
  compile errors, graph size, compiler execution order, and a typed execution
  plan for script clients.
  `scripts/platform/validate_workflow_definition.py` wraps the same endpoint
  for local JSON workflow files and can exit nonzero with `--fail-on-invalid`.
- Operator and artifact contracts are inspectable through:
  - `GET /v1/node-specs`
  - `GET /v1/artifact-types`
  - `GET /v1/artifact-payload-schemas`
  - `GET /v1/artifact-payload-schemas/{artifact_type}/{schema_version}`
- Artifact lineage is inspectable through:
  - `GET /v1/workflow-runs/{workflow_run_id}/artifact-graph`
  - `GET /v1/artifacts/{artifact_id}/lineage`
- Implemented template ids:
  - `ocr-pages`
  - `contextual-extraction`
  - `ocr-compare-contextual-extraction`
- Template launch expands to a persisted `WorkflowDefinition`, immutable
  `WorkflowVersion`, queued `WorkflowRun`, concrete `NodeRun` records, and
  outbox messages for executable root nodes.
- Persisted workflow runs expose their concrete execution plan through
  `GET /v1/workflow-runs/{workflow_run_id}/execution-plan`, including node-run
  dependencies, root/leaf nodes, expected port contracts, and bound artifact
  refs.
- Template materialization expands to a persisted `WorkflowDefinition` and
  immutable `WorkflowVersion` without creating a run. This supports experiment
  grids that need a reusable workflow version.
- `scripts/platform/run_ocr_workflow.py` and
  `scripts/platform/run_contextual_extraction_workflow.py` use the template
  launch API instead of constructing the full DAG client-side.
- `scripts/platform/run_workflow_definition.py` can run a local workflow
  definition JSON end to end: validate, persist, version, launch, execute,
  summarize, and optionally include output bundles.
- OCR workflow runs now emit both ordered `ocr.page_result` sequences and a
  document-level `ocr.document_result`; `scripts/platform/run_ocr_workflow.py`
  returns both payload views for Python callers.
- `scripts/platform/run_contextual_extraction_workflow.py` can opt into the
  OCR compare/select template with `--comparison-ocr-engine`.
- `scripts/platform/run_ocr_experiment.py` uses template materialization before
  creating its parameter-grid experiment.
- `scripts/platform/run_contextual_extraction_experiment.py` materializes the
  contextual extraction template, or the OCR-compare contextual extraction
  template when `--comparison-ocr-engine` is provided. It creates extraction
  experiment variants through parameter presets, executes every variant, and
  returns comparison metrics plus per-variant output payloads for
  script/notebook users.
- Template configs reject literal sensitive keys. Provider credentials should
  be passed as `credential_ref` or non-secret environment-variable names such
  as `api_key_env_var`.

Tasks:

1. Validate graph structure.
2. Resolve spec/config nodes.
3. Validate port compatibility.
4. Expand execution modes:
   - `single`
   - `map`
   - `reduce`
   - `stateful_sequence`
5. Persist concrete node runs and dependencies.
6. Queue executable node runs through outbox.

Gate:

- Compiler rejects missing inputs, incompatible artifact types, unresolved handlers, and cycles.
- Worker executes concrete `node_run_id` jobs only.

## Stage 7: Port Current Sequential Extraction

Goal: move the current advanced Notarius extraction into the generic platform.

Current implementation status:

- Generic artifact-producing define nodes are implemented for:
  - `prompt.template.define -> prompt.template`
  - `extraction.schema.define -> extraction.schema`
  - `model.binding.define -> model.binding`
  - `input.policy.define -> input.policy`
  - `context.static.define -> context.bundle`
- These are executable `single` operators for the current compiler/worker path.
  They can later become compiler-resolved spec nodes without changing the
  downstream artifact contracts.
- `extraction.contextual_structured` consumes prompt, schema, model binding,
  input policy, optional static context, and OCR page-result sequences.
- Local echo and OpenAI-compatible structured extraction engines are available
  behind `model.binding` selection. The OpenAI-compatible path uses
  environment-variable API keys and does not serialize secrets into artifacts.
- `export.dataset` consumes `extraction.document_result` and emits a first-class
  `export.dataset` artifact in JSON, JSONL, or CSV format.
- `validation.schema` consumes `extraction.document_result` and
  `extraction.schema`, then emits a detailed `validation.result` artifact plus
  normalized `evaluation.metrics` for experiment comparison.
- The `contextual-extraction` workflow template now ends with `export.dataset`
  so script/API users get a typed export artifact from a normal run.

Mapping:

```text
DatasetProcessor       -> stateful sequence executor
ItemProcessor          -> per-item extraction step
ContextProvider        -> input assembly provider or context operator
ContextStrategy        -> input/history policy
MessageBuilder         -> model-input renderer
LLM engine port        -> model invocation port
Structured schema      -> extraction.schema artifact
LLM raw response       -> model.response artifact
Parsed result          -> extraction.record_result artifact
```

Tasks:

1. Move generic sequence-processing behavior into `notarius_core`.
2. Move schematism-specific types and prompts into `notarius_schematisms`.
3. Move provider-specific LLM code into `notarius_llm`.
4. Wire `prompt.template`, `extraction.schema`, `model.binding`, and
   `input.policy` artifacts into a contextual structured extraction node.
5. Wire graph-visible static context providers into contextual extraction.
6. Preserve existing context pruning, image pruning, and history behavior.
7. Emit input assembly and invocation traces.

Gate:

- Existing structured extraction tests pass through the new package path.
- A four-page sequence can run through the worker and produce page-level artifacts.

## Stage 8: OCR And Model Comparison Slice

Goal: prove the platform with multiple model families.

Current implementation status:

- Page image sequence artifacts, OCR extraction, local text OCR, Tesseract OCR,
  Mistral OCR, page-by-page OCR comparison, and OCR stream selection are
  implemented in the platform packages.
- The remaining Stage 8 work is hardening the selector UI and adding richer
  comparison/selection policies after the basic artifact flow is exercised.

Tasks:

1. Add source PDF/image import.
2. Add page image sequence artifacts.
3. Add OCR operator interface.
4. Add one local OCR operator.
5. Add Mistral OCR operator.
6. Add OCR comparison operator.
7. Add OCR selector operator.

Gate:

- Four page images can run through two OCR methods.
- The API exposes both OCR outputs and comparison artifacts.
- The result can feed structured extraction.

## Stage 9: Experiment Matrix

Goal: support scientific comparison.

Current implementation status:

- Generic `Experiment`, `ExperimentParameter`, and `ExperimentVariant` models
  are implemented in the platform domain.
- Experiments can expand a parameter grid over workflow node config paths.
- Experiment creation accepts parameter presets for common operator config
  paths, so script clients do not need to hard-code internal workflow config
  paths:
  - `ocr_engine -> engine`
  - `ocr_engine_config -> engine_config`
  - `ocr_language_hints -> language_hints`
  - `ocr_candidate_a_label -> candidate_a_label`
  - `ocr_candidate_b_label -> candidate_b_label`
  - `ocr_selected_candidate -> selected_candidate`
  - `ocr_selection_note -> decision_note`
  - `model_provider -> provider`
  - `model_name -> model`
  - `model_parameters -> parameters`
  - `prompt_template -> template`
  - `extraction_schema -> json_schema`
  - `static_context -> context`
  - `input_policy_type -> policy_type`
  - `input_policy_settings -> settings`
  - `export_format -> format`
  - `export_filename -> filename`
- Parameter presets can override their grid name. This is required when the
  same preset kind targets parallel nodes, for example `ocr_a.engine` and
  `ocr_b.engine`, because experiment parameter names must be unique.
- Creating an experiment through the API launches one workflow run per variant
  against the same workflow version and input corpus.
- Each launched workflow run records experiment metadata:
  - `experiment_id`
  - `experiment_variant_key`
  - `experiment_parameter_values`
- Experiments are persisted through in-memory and SQLAlchemy unit-of-work
  adapters.
- The API exposes an experiment comparison view that derives per-variant:
  - workflow run status
  - node-run status counts
  - artifact counts
  - invocation count
  - validation error count
  - duration and cost hints from invocation traces
  - `evaluation.metrics` artifact metadata
- The comparison response now includes normalized `metric_names` and per-variant
  `metric_values` so script users can build a table across variants without
  parsing every raw metric artifact. Summary values use names such as
  `summary.total_cost`; evaluation artifact metadata is prefixed by
  `metric_family`, for example `ocr_comparison.mean_similarity_ratio`.
- The API supports variant-scoped cancellation and rerun:
  - cancellation marks the current workflow run and non-terminal node runs as
    cancelled
  - rerun creates a fresh workflow run for the same variant parameters
  - the variant points at the latest workflow run and records previous run ids
    in metadata
- `scripts/platform/manage_experiment.py` exposes experiment inspection,
  comparison, cancellation, single-variant rerun, and bulk failed-variant
  rerun to script/notebook users. Its `comparison --output-format csv
  --output-path ...` command writes a flattened comparison table with parameter,
  artifact-count, and normalized metric columns. Its `outputs` command fetches
  variant output bundles with artifact-type, payload, text-payload, and trace
  filters. Its `events` command aggregates the current workflow-run timeline
  for every experiment variant.
- The worker rechecks node and workflow cancellation before persisting handler
  outputs, so mid-flight cancellation is not overwritten by success or failure.
- The worker validates handler output refs against registered operator output
  ports before persisting artifacts, sequences, traces, or success state. A
  handler that omits a required output, emits an undeclared port, or returns the
  wrong artifact type/schema fails the `NodeRun` permanently.
- Workflow launches, node-run retry, worker execution, and artifact persistence
  now write typed lifecycle outbox messages:
  - `events.workflow_run.queued|running|succeeded|failed_retryable|failed_permanent|cancelled`
  - `events.node_run.queued|running|succeeded|failed_retryable|failed_permanent|cancelled`
  - `events.artifact.created`
- Workflow-run timelines are inspectable through
  `GET /v1/workflow-runs/{workflow_run_id}/events`, which normalizes existing
  outbox workflow, node, artifact, DLQ, and malformed retained event payloads
  into ordered run-scoped events. `scripts/platform/watch_workflow_run_events.py`
  wraps this endpoint for one-shot or polling script use.
- Single artifacts are inspectable through `GET /v1/artifacts/{artifact_id}/inspect`,
  which can include decoded JSON/text payloads and the artifact lineage graph.
  `scripts/platform/manage_artifact.py` exposes metadata, inspection, and
  lineage commands for script/notebook users.
- Workflow definitions can opt into first-stage concrete map planning with
  `metadata.execution_planning = "concrete_map"`. Root map nodes with one bound
  sequence input expand into one node run per item, and downstream reduce nodes
  receive a collected artifact sequence ordered by `map_item_index`. Existing
  OCR/contextual templates stay on the compatibility path until their handlers
  are split into per-item execution plus aggregation.
- The local node-run outbox drainer writes malformed executable node-run
  commands to `dlq.node_run.execute` and removes the invalid source command from
  the pending queue.
- The local node-run outbox drainer now classifies execution failures after the
  executor updates node-run state:
  - `failed_retryable` keeps the executable command pending for another attempt
  - `failed_permanent` writes `dlq.node_run.execute` and marks the source command
    published
- Worker startup now reconciles the expected JetStream stream definitions before
  starting the outbox drain loop.
- The FastStream worker registers `jobs.node_run.execute.requested` as a durable
  JetStream consumer on `TASKS`, and the NATS outbox publisher targets the
  proper stream for task, event, live-delta, and DLQ subjects.
- Generic outbox publishing now terminal-fails messages after the configured
  publish-attempt cap instead of retrying broker publish errors indefinitely.
- The API exposes operational outbox inspection and failed-message requeue
  endpoints:
  - `GET /v1/outbox-messages` with status, subject-prefix, limit, and offset
    filters
  - `GET /v1/outbox-messages/dlq-summary` grouped by consumer name, error
    code, and original subject
  - `GET /v1/outbox-messages/{outbox_message_id}`
  - `POST /v1/outbox-messages/{outbox_message_id}/requeue`
  - `POST /v1/outbox-messages/cleanup` for dry-running or deleting old
    terminal `published` and `failed` messages by status, creation timestamp,
    subject prefix, and message type. Cleanup rejects `pending` messages.
- `scripts/platform/manage_outbox.py` exposes those controls to script users,
  including `cleanup --older-than ...` with dry-run default and `--execute` for
  deletion. `cleanup --archive-path ... --execute` first writes a dry-run JSON
  archive of the matched messages, then calls the destructive cleanup request.

Tasks:

1. Continue expanding metric normalization as new metric artifact families are
   added.
2. Add higher-level parameter helpers as new stable operator config fields
   emerge.
3. Add experiment comparison UI views.
4. Add deeper cancellation UX and operational controls:
   - cancelled/outdated outbox cleanup when broker delivery is introduced
   - durable server-side archive storage for old outbox and DLQ messages if
     local script archives are not sufficient

Gate:

- A user can run N variants over the same input corpus.
- Runs can be compared by cost, latency, validation failures, and evaluation metrics.
- Failed or cancelled variants can be rerun without mutating prior artifacts.

## Stage 10: Product UI Foundation

Goal: expose a usable workbench, not a blank graph editor.

Initial views:

- project/corpus view
- guided workflow templates
- run view
- artifact inspector
- page timeline
- OCR comparison viewer
- input assembly inspector
- invocation trace inspector
- extraction result inspector
- evaluation dashboard

Gate:

- The first screen lets a researcher start a concrete workflow.
- The graph is inspectable and editable after template creation.
