# Stacking Backend Reuse Map

The Stacking backend at `/Users/user/PycharmProjects/Stacking/projects/backend` is a useful donor project for infrastructure patterns. It should not be copied wholesale.

## Reuse Principles

Use Stacking for backend structure and boring infrastructure.

Do not import invoice, KSeF, JPK, or accounting-client concepts into Notarius platform vocabulary.

Adapt patterns into Notarius names:

```text
JobTrace      -> NodeRun or WorkflowRun status/history
ArtifactObject -> Artifact
OutboxMessage -> OutboxMessage
FileStoragePort -> ArtifactPayloadStoragePort
```

## High-Value Reuse Targets

### Workspace Layout

Reuse the monorepo shape:

```text
apps/api
apps/worker
libs/core
libs/storage
libs/persistence
libs/messaging
libs/shared
```

Why: Notarius needs clear separation between product apps, core domain logic, and infrastructure adapters.

### FastAPI App Factory

Reuse the pattern:

```text
create_app(settings, jetstream_factory)
lifespan startup/shutdown
DB init
JetStream connection
CORS
pagination
error handlers
health/live
health/ready
```

Adapt it to Notarius routes:

```text
sources
workflows
workflow versions
workflow runs
node runs
artifacts
experiments
exports
```

### Dependency Wiring

Reuse the pattern for:

- settings dependency
- session factory
- unit-of-work factory
- object storage dependency
- auth dependency
- organization/project scope dependency when product auth is introduced

Avoid copying invoice-specific auth and KSeF credential resolution.

### NATS JetStream And Worker Streams

Reuse the stream shape:

```text
TASKS: durable work queue
EVENTS: durable audit events
LIVE_DELTAS: short-lived UI/live updates
DLQ: dead-letter messages
```

Adapt subjects:

```text
jobs.workflow.compile.requested
jobs.workflow.run.requested
jobs.node_run.execute.requested
events.workflow_run.*
events.node_run.*
events.artifact.*
dlq.*
```

### Outbox Publisher

Reuse the database outbox pattern:

```text
write domain change and outbox message in one transaction
publisher polls pending messages
publisher sends through broker
publisher marks messages published
```

Why: this keeps API and worker state changes durable even when NATS is temporarily unavailable.

### Job Specs And Status Transitions

Reuse the idea, not the invoice types.

Notarius equivalents:

```text
WorkflowCompileJobSpec
WorkflowRunRequestedJobSpec
NodeRunExecuteJobSpec
ExperimentRunJobSpec
```

Status models should carry:

- queued
- running
- succeeded
- failed retryable
- failed permanent
- cancelled

Errors should preserve operation, identifiers, retryability, provider details, and original cause.

### Unit Of Work And Repository Ports

Reuse the UoW shape:

```text
async with uow:
    ...
    await uow.commit()
```

Notarius repositories:

```text
workflow_definitions
workflow_versions
workflow_runs
node_runs
artifacts
artifact_sequences
outbox
processed_messages
experiments
```

### Object Storage

Reuse local and S3/MinIO object-store adapters.

Rename around artifact payloads:

```text
SaveArtifactPayloadCommand
StoredArtifactPayload
ArtifactPayloadStoragePort
```

Strip invoice-specific metadata and replace it with:

```text
artifact_id
artifact_type
workflow_run_id
node_run_id
content_hash
source_filename
```

### Architecture Tests

Reuse boundary tests early.

Required Notarius checks:

```text
notarius_core does not import notarius_api
notarius_core does not import notarius_worker
notarius_core does not import notarius_dagster
notarius_core does not import fastapi
notarius_core does not import sqlalchemy
notarius_core does not import nats or faststream
notarius_core does not import concrete provider SDKs
```

## Do Not Reuse

Do not reuse these directly:

- invoice-specific domain models
- KSeF/JPK modules
- accounting-client model
- invoice discovery and promotion jobs
- OpenBao/KSeF credential flow unless a later product requirement needs secret storage
- invoice-specific API route structure
- invoice-specific artifact kinds

## Recommended Extraction Order

1. Copy/adapt workspace and package layout.
2. Copy/adapt generic settings/logging only if needed.
3. Copy/adapt storage ports and adapters.
4. Copy/adapt persistence/UoW shape.
5. Copy/adapt messaging/JetStream shape.
6. Copy/adapt outbox publisher.
7. Build Notarius workflow/artifact/node-run domain from scratch.
8. Port current Notarius extraction into the new domain.

This order prevents invoice vocabulary from leaking into the Notarius platform core.

