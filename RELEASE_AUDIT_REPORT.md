# Notarius First-Release Audit

Date: 2026-08-13  
Release target: first server deployment

## Release decision

- **Private, trusted pilot:** conditional go.
- **Public or hostile multi-user deployment:** not yet recommended.

Before the private pilot, put the Compose gateway behind a real TLS reverse
proxy, back up `/data` and the long-lived crypto keys, rehearse restoration,
and monitor API readiness, memory, disk, and Prefect.

## Reported graph corruption

The reported “changed in another session” message was misleading. Another
session was not required. Multiple same-tab and collaboration-ordering defects
could produce the conflict or silently diverge the canvas from the server.

The principal failure chain was:

1. A command could commit between the room-head read and WebSocket hub join.
2. The joining client received a stale `room.ready` head and permanently missed
   the committed event.
3. The client accepted later non-contiguous sequence numbers, advancing its
   marker without applying the missing command.
4. Saving could submit the incomplete document at the apparently current
   sequence, causing a conflict or overwriting the missing change.

Additional corruption paths included stale reconnect heads, receipts advancing
sequence state without applying commands, checkpoint responses overwriting a
newer peer presentation, epoch-reset responses arriving out of order, offline
edits that were never queued, and invalid Artifact Viewer mappings closing the
room protocol.

## Collaboration and persistence fixes

- The server now joins and buffers a room before reading its authoritative
  head, sends `room.ready` first, and only then activates fanout.
- The client applies only the exact next command sequence. Gaps pause command
  traffic until an authoritative snapshot is loaded.
- Stale `room.ready` and `room.rehydrate` messages cannot rewind a newer head.
- Receipts acknowledge completion only; they never advance materialized graph
  state.
- Interrupted commands retain their command ID, room epoch, observed sequence,
  and command body for HMAC-compatible idempotent replay.
- Delayed peer events and private replay receipts cannot skip unapplied changes.
- Checkpoint reconciliation is anchored to the epoch of the submitted command;
  late responses from an old epoch cannot reverse a room reset.
- A newer peer edit arriving during checkpoint continuation remains visible and
  leaves the graph dirty until that newer sequence is checkpointed.
- Save no longer falls back to legacy whole-document PUT while the room is
  reconnecting or unsynchronized.
- Graph authoring is disabled while saving, running, switching graphs, or while
  durable room synchronization is unavailable.
- Graph-switch navigation becomes busy immediately after confirmation, closing
  the route-transition edit-loss window.
- Blank or partial Artifact Viewer interaction mappings are omitted until both
  fields are valid instead of closing the WebSocket with a protocol error.
- HTTP 409 errors preserve the server detail instead of universally claiming
  that another session changed the graph.

Key implementation areas:

- `apps/api/src/notarius_api/v1/routes/collaboration/views.py`
- `apps/api/src/notarius_api/v1/routes/collaboration/hub.py`
- `apps/web/src/features/workbench/room/graph-room-session.ts`
- `apps/web/src/features/workbench/room/useGraphRoomSession.ts`
- `apps/web/src/features/workbench/ui/Workbench.tsx`
- `apps/web/src/features/workbench/ui/useSavedGraphLifecycle.ts`

## Other release hardening

### Network and remote content

- WFS and WMS requests are pinned to the DNS-validated public IP while
  preserving the original Host and TLS SNI, preventing DNS-rebinding TOCTOU.
- WFS imports require a total feature limit, capped at 10,000 features and
  16 MiB of cumulative response data.
- Per-page WFS responses remain independently bounded.
- Staged uploads have an exact configurable limit, hard-capped at 64 MiB;
  oversized partial files are removed and return HTTP 413.

### Artifact delivery

- Directly stored artifacts use fixed-size streaming reads with deterministic
  cleanup rather than whole-object buffering.
- Buffered inline, table, and chunked-Geo responses are capped at 64 MiB when
  their logical size is known, with a post-serialization guard as fallback.
- PMTiles full responses and byte ranges are capped at 16 MiB before storage
  reads.
- Artifact downloads no longer navigate away from the workbench page.

### Execution safety

- All top-level graph executions, including diagnostic `/runs`, share one
  process-wide admission limiter.
- The default active execution limit is two; excess requests receive a typed
  HTTP 429 response with `Retry-After`.
- Slots are released on success, failure, cancellation, and start-time errors.
- SQL result rows are bounded.
- The production API dependency set includes GIS, LLM, and OCR, but excludes
  the SQL plugin and DuckDB until untrusted SQL has a separate least-privileged,
  networkless worker.

### Deployment configuration

- Compose passes and validates required OIDC and independent crypto-key inputs.
- Forwarded-header trust covers the configured Docker subnet rather than the
  bridge gateway address alone.
- `/health` remains liveness; `/ready` checks initialized resources, database
  connectivity, and Prefect under a three-second ceiling.
- Compose uses `/ready` for the API health check.
- Raster TileJSON uses public same-origin `/api` paths.
- Deployment documentation distinguishes the plain loopback gateway from the
  required operator-provided TLS endpoint.

## Verification

- Backend: **870 tests passed**.
- Frontend: **516 tests passed across 80 files**.
- ESLint: **0 errors**, 16 warnings.
- TypeScript typecheck: passed.
- Next.js production build: passed.
- Ruff: passed.
- OpenAPI schema and checked-in TypeScript client: synchronized.
- Docker Compose configuration render: passed.
- `git diff --check`: passed.
- Real-browser smoke: OIDC login, node creation, pointer-created connections,
  dependency execution, and successful completion.

One remaining warning is the existing Authlib JOSE deprecation notice.

## Remaining risks

### Required before public multi-user exposure

- Run API, migration, Prefect, and gateway containers as non-root. This needs a
  rehearsed image and volume-ownership migration; a Compose `user:` line alone
  is not safe.
- Set evidence-based CPU, memory, PID, file-descriptor, and service limits.
- Move every untrusted query runtime into a separately isolated worker.
- Exercise backup/restore and failure recovery on the actual target server.

### Known lifecycle and compatibility debt

- Staged uploads have no safe TTL or workspace quota. Saved graphs currently
  retain upload keys, so age-only deletion would break valid reruns. A proper
  staged → promoted/referenced → reclaimable lifecycle is required.
- Legacy artifacts without logical `byte_size` cannot always be rejected before
  reconstruction. They retain compatibility and are checked after serialization.
- S3 loading still spools before downstream streaming; the new response path
  prevents a second unbounded allocation but is not true remote-to-client
  streaming.
- Base-image and service-version pinning should move to immutable digests.

## Refactoring priorities

1. **Workbench collaboration and persistence state machine**

   `Workbench.tsx` is over 4,000 lines and owns authoring, collaboration,
   persistence, execution, presentation, and rendering. Extract workflow-owned
   hooks in this order: collaboration/persistence, authoring command mapping,
   presentation, then render composition.

2. **Artifact format ownership**

   `ArtifactService` is over 1,600 lines and mixes stored content, table export,
   GeoJSON, PMTiles, raster, and WMS. Split it at real format boundaries rather
   than introducing generic helper modules.

3. **Execution and lifecycle reducers**

   `useRunExecution.ts`, `useSavedGraphLifecycle.ts`, and the room session have
   interdependent lifecycle states. Explicit reducers/state machines would make
   ordering invariants inspectable and substantially reduce race risk.

4. **Upload lifecycle model**

   Add explicit promotion/reference state and transactional quota reservation
   before implementing expiry or cleanup.

## Durable engineering rules discovered

- Receipts acknowledge command completion; they never advance materialized
  client state.
- Idempotent retries preserve every field covered by the command HMAC.
- Missing collaboration sequences require authoritative rehydration before
  dependent traffic resumes.
- A paginated remote import requires total item and total byte limits; per-page
  limits alone are not a memory bound.
- DNS validation must bind the validated address to the actual connection.

## Working-tree note

The repository was already extensively dirty when the audit began. Existing
changes were preserved. Audit fixes are currently mixed into that working tree
and have not been staged or committed.
