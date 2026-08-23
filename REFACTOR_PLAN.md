# Grafy Refactor Plan — Findings Tracker

> Tracks the 11 materially useful simplifications from the independent audit of
> the current `grafy_*` working tree (HEAD `70ad310c`). This document is the
> single source of truth for what has been done at each moment. Each finding is
> implemented one at a time in rank order. When a finding is fully implemented,
> its Status is flipped to `DONE` and the "Done" section records what changed.

- **Mode:** implementation tracker (not read-only audit)
- **Rule:** `[R25: Design Before Edits]` — read and design before editing; do not
  start a finding until its predecessor prerequisites are satisfied.
- **Current finding in progress:** 9

---

## Ranked findings (implementation order)

| # | Finding | Verdict | Status | Prerequisite |
|---|---|---|---|---|
| 1 | One post-activation WebSocket writer | Recommend | DONE | None |
| 2 | Remove synchronous `storage.exists` | Recommend | DONE | None |
| 3 | Core-owned Module definition resolution | Recommend | DONE | None |
| 4 | Phase-local execution adjacency indexes | Recommend | DONE | None |
| 5 | Co-locate presence with room membership | Recommend | DONE | Finding 1 first |
| 6 | Deepen the authored-document reducer | Recommend | DONE | None |
| 7 | Transition-owned execution lifecycle | Recommend | DONE | Explicit state design |
| 8 | One frontend execution observer | Recommend | TODO | Prefer finding 6 first |
| 9 | Execution history owns active-run uniqueness | Recommend | TODO | Design with 7 |
| 10 | Typed generated-node ports | Recommend | SKIP (subject removed) | None |
| 11 | Discriminated artifact content | Recommend | TODO | Finding 2 first |

---

## Done

### Finding 10 — Keep generated-node ports typed (SKIP — subject removed)

**Not implemented.** The generated-node feature was removed from the current
tree during the external rename: `git ls-files` and `git status` show no
`generated_node*` source files, and commit `447f7d0 "chore: preserve rename and
generated node prototype snapshot"` archived the prototype. `GeneratedNodePort`/
`GeneratedNodeRelease` (core domain), `schema.py` columns, operator/runner call
sites, and tests no longer exist as source. There is nothing to refactor; the
finding is recorded as skipped rather than fabricated against absent code.

---

### Finding 7 — Transition-owned execution lifecycle (DONE)

**What changed:**
- `libs/core/src/grafy_core/domain/execution_history.py` — `GraphExecution` now
  owns explicit transition operations: `transition_to_running` (queued only,
  stamps start time), `transition_to_cancelling` (queued/running only), and
  `transition_to_terminal` (terminal status, finish timestamp, workflow id, and
  error advance together; cancelling can only complete as cancelled). Illegal
  sequences are rejected, preventing running-without-start /
  queued-with-start / success-with-error states.
- `apps/api/src/grafy_api/v1/routes/executions/services.py` — `mark_running`,
  `mark_cancelling`, and `complete` now delegate to the transitions instead of
  mutating correlated fields directly.
- `apps/api/src/grafy_api/v1/routes/executions/runtime/manager.py` —
  `_RunExecutionRecord` now distinguishes active state (`status`, transient
  `error`) from one typed terminal outcome (`terminal: _TerminalOutcome` holding
  status/result/error together); deleted the redundant `retained_terminal` bool
  (replaced by `terminal is not None`); `_complete` sets the outcome once;
  `snapshot` and `_publish_cleared` read from the terminal outcome.

**Files touched:** `execution_history.py`, `services.py`, `manager.py`,
`tests/unit/core/test_execution_history.py` (added legal-transition-table and
single-complete-outcome tests).

**Validation:** 29 execution history + manager tests pass; ruff clean.
Cancel-versus-complete winners, failed partial results, cancellation-with-result,
journal sealing, room publication, lease release, and shutdown callback races
preserved (existing race tests still pass).

---

### Finding 6 — Deepen the authored-document reducer (DONE)

**What changed:** Two tranches applied together.

**1. Reducer is the sole overlay owner (delete legacy invalidation):**
- `Workbench.tsx` — deleted the `invalidateWorkflowResults` helper that called
  the legacy React Flow-shaped `invalidateWorkflowNodeRuns`; the three call
  sites (`onEdgesChange`, `commitEdge`, contextual discovery) now call the
  existing `clearRunError()` (or nothing when a semantic command already clears
  errors). The canonical `reduceWorkbenchAuthoringState` `apply_commands`
  already resets overlays atomically for invalidated nodes, so the redundant
  legacy traversal (which could overwrite overlays from stale renderer state)
  is gone.
- `canvas/types.ts` — deleted `invalidateWorkflowNodeRuns` and its now-unused
  `WorkflowNodeState`/`WorkflowConnectionState` interfaces.

**2. Batch normalization (avoid O(K×(V+E)) deep clones):**
- `model/graph-document.ts` — split `applyGraphCommand` into the public
  normalizing wrapper and an exported canonical
  `applyGraphCommandNormalized` that operates on an already-normalized document
  without re-normalizing/deep-cloning per command.
- `canvas/graph-document-adapter.ts` — the `apply_commands` reducer now
  normalizes the starting document once per batch, then applies each command
  through the canonical transition.

**Files touched:** `Workbench.tsx`, `canvas/types.ts`, `canvas/types.test.ts`,
`model/graph-document.ts`, `canvas/graph-document-adapter.ts`,
`canvas/graph-document-adapter.test.ts`.

**Tests:** Removed the two legacy `invalidateWorkflowNodeRuns` tests (and the
`nodeWithRun` helper); added reducer-level tests: downstream-descendant overlay
clearing (preserving upstream/unrelated), disabled-edge non-clearing, and
sequential-versus-batch equivalence. Web suite: 547 tests pass (was 546);
typecheck and eslint clean. Public `applyGraphCommand` (room bridge) and
standalone/room callers preserved.

---

### Finding 5 — Co-locate presence with registered room membership (DONE)

**What changed:** The hub now stores one room-member value that co-locates the
registered session with its optional presence, replacing the two independent
`_rooms` and `_presence` maps:

- `hub.py` — added `_RoomMember(session, presence)`; `_rooms` now maps to
  members. `join` registers a member; `register_presence` sets its presence;
  `participants_for` reads members' presence; `_expire_locked` clears presence
  (never membership) on TTL; `_close_session` removes the member (presence gone
  with it); `shutdown`/`_sessions_for`/`_sessions_for_workspace_user` map over
  `member.session`.
- `apply_presence_update` now requires the exact open registered session under
  the lock (`member.session is session and not session.closed`); a delayed
  update after close is dropped and can never recreate presence.

**Files touched:** `hub.py`, `test_graph_room.py` (added
`test_presence_update_after_close_cannot_recreate_membership` and
`test_presence_update_race_with_close_never_recreates_after_close`; moved
`PresenceUpdateSubmitMessage` to module-level import).

**Validation:** 24 graph-room tests + 1 collaboration acceptance test pass; ruff
clean. Join gating, rate limits, best-effort delivery, TTL semantics, and
idempotent close preserved.

---

### Finding 4 — Phase-local execution adjacency indexes (DONE)

**What changed:** Each execution phase now builds a phase-local adjacency index
once instead of rescanning the flat edge list per node/port (O(VE) → O(V+E)):

- `runtime/compiler.py` `_topological_order` — builds `outgoing_by_node` once
  (grouping internal edges by source node); the deque loop touches only each
  node's outgoing edges. Zero-indegree ordering, duplicate-edge multiplicity,
  and cycle detection preserved.
- `runtime/coordinator.py` — builds `incoming_by_node` once and derives the
  per-node `incoming_edges` tuples from it instead of a per-node O(E)
  comprehension. Added `CompiledEdge` import.
- `runtime/edge_values.py` `assemble_inputs` — builds `incoming_by_port` once
  (still filtering `to_node == node_request.id`, preserving the direct-call
  contract where incoming edges can target other nodes), then resolves ports
  and plugs from it.

**Files touched:** `compiler.py`, `coordinator.py`, `edge_values.py`,
`test_graph_compiler.py` (added fan-in/fan-out/duplicate-edge, cycle-rejection,
and large-sparse-DAG equivalence tests).

**Validation:** 9 compiler+edge-value tests, 11 inline/Prefect engine tests pass;
ruff clean. Ordering, duplicate edges, external pins, plug ordering, and
variadic fan-in preserved.

---

### Finding 3 — Core-owned workspace Module definition resolution (DONE)

**What changed:** The workspace-library module now owns one canonical
definition-resolution interface. A module-level `validate_optional_input_targets`
in `libs/core/src/grafy_core/application/modules.py` is the single shared
validator (the catalog's ~90-line near-copy is deleted). Added
`ModuleLibraryService.resolve_definition` as the canonical pinned-execution
resolver.

- `modules.py` — extracted the near-duplicate validation into
  `validate_optional_input_targets` (shared by service + catalog); the service's
  private `_validate_optional_input_targets` delegates to it; added
  `resolve_definition`; added `_resolve_revision_or_none` helper that tolerates
  both return-None (port) and raise-NotFoundError (service) revision providers.
- `catalog/services.py` — `get_definition` delegates to
  `_module_library.resolve_definition` when a module library is present,
  otherwise uses the saved-graphs fallback through the shared validator;
  `list` no longer re-validates (catalog_definitions already validates + skips
  invalid); deleted the catalog's `_validate_optional_input_targets` copy;
  removed now-unused `GraphModuleReferenceError` and `UnknownOperatorError`
  imports.

**Files touched:** `modules.py`, `catalog/services.py`, `test_modules.py`
(added `test_module_library_resolve_definition_is_the_core_contract` proving
resolve+validate success and NotFoundError on a missing revision).

**Validation:** 17 module+compiler tests pass; ruff clean. Pinned
unpublished/deprecated/withdrawn release behavior, list-time skipping, and
lightweight no-module-library configuration preserved.

---

### Finding 2 — Remove synchronous `storage.exists` (DONE)

**What changed:** The synchronous `exists` presence operation is removed from
`FileStoragePort` and both real adapters. `await storage.stat(...) is not None`
is now the sole public presence operation. All four caller modules now use
the async `stat` presence check:
- `artifact_collections.py` (accessible + intact checks)
- `operators/tables.py` (table accessibility)
- `executions/runtime/invocation_cache.py` (cache staleness)
- `routes/artifacts/services.py` (ref validation)

Private synchronous `_file_exists` remains inside S3 `_move_sync`, which already
runs under `asyncio.to_thread`, so synchronous remote I/O never blocks the event
loop.

**Files touched:**
- `libs/core/src/grafy_core/ports/storage.py` — removed `exists` from protocol.
- `libs/storage/src/grafy_storage/adapters/s3.py` — removed public `exists`;
  kept private `_file_exists` for thread-dispatched `_move_sync`.
- `libs/storage/src/grafy_storage/adapters/local.py` — removed public `exists`.
- Four caller modules listed above — replaced sync `exists` with `await stat`.
  Generator expressions containing `await` were rewritten as explicit loops
  (an `all(...)` over an async generator is not iterable).
- Test fakes: removed dead `exists` methods from four plugin `EmptyStorage`
  fakes and the GIS `TrackingStorage` fake.
- `tests/unit/storage/test_s3.py` — `exists` assertions converted to async
  `stat`; first test made `@pytest.mark.asyncio`.
- `tests/unit/api/test_invocation_cache.py` — outage test now patches async
  `stat` instead of sync `exists`.
- `tests/unit/core/test_tables.py` — added `AsyncStatOnlyStorage` fake (no sync
  `exists` attribute) and `test_table_accessibility_uses_only_async_stat`
  proving async flows call only async presence.

**Validation:** 176 tests pass across core tables/invocation-cache, API
artifact/cache/table/GIS, storage, and all plugin suites. ruff clean on all
changed files. "Missing returns false" and contextual backend failure behavior
preserved.

---

### Finding 1 — One post-activation WebSocket writer (DONE)

**What changed:** All six command-rejection branches in
`_handle_command_submit` (forbidden, not_found, missing_head, command_rejected,
head_conflict, idempotency_mismatch) now route through `hub.deliver_private`
instead of writing directly to `session.websocket.send_json`. The only direct
socket write remaining is the pre-activation `room.ready` handshake at
`views.py:223`. The activated session's single queued sender
(`_sender_loop`) is now the complete outbound interface for every message
after activation.

**Files touched:**
- `apps/api/src/grafy_api/v1/routes/collaboration/views.py` — rejection
  branches call the new `_reject_command` helper (head_conflict kept inline
  for its extra fields); added `_reject_command` helper with docstring noting
  the ready-before-live-events rule.
- `tests/unit/api/test_graph_room.py` — added
  `test_activated_session_has_one_fifo_post_activation_writer` proving a
  blocked socket still yields strict FIFO delivery through exactly one
  sender task.

**Validation:** Full `test_graph_room.py` suite passes (22 tests); ruff clean
on both files. Pre-activation `room.ready` ordering and queue-full
slow-consumer disconnection behavior are preserved (existing tests still
pass).

---

## Done (cont.)

_Entries above are chronological; the most recent completed finding is at the
top of this section._

---

## Finding 1 — One post-activation WebSocket writer

- **Verdict:** Recommend immediately.
- **Status:** TODO
- **Prerequisite:** None
- **Evidence:** `room.ready` is correctly written before activation at
  `apps/api/src/grafy_api/v1/routes/collaboration/views.py:223`. After
  activation, six command-rejection branches still write directly at
  `views.py:373`, while receipts and normal traffic use the hub queue. The
  activated sender owns queued writes at
  `apps/api/src/grafy_api/v1/routes/collaboration/hub.py:476`.
- **Problem:** An activated session has two concurrent socket writers.
  Rejections can overtake accepted messages or heartbeats and bypass queue
  backpressure and slow-consumer handling.
- **Simpler representation:** Keep the pre-activation ready write direct. Route
  every subsequent message, including rejections, through `hub.deliver_private`.
  One queue and sender become the complete outbound interface.
- **Smallest scope:** Collaboration `views.py` and `test_graph_room.py`; no
  protocol or frontend change.
- **Risks:** Preserve the ready-before-live-events ordering and established
  queue-full disconnection behavior.
- **Validation:** Existing rejection coverage starts at
  `tests/unit/api/test_graph_room.py:518`, slow-consumer behavior at `:847`.
  Add a blocked-socket concurrency test proving FIFO delivery and exactly one
  post-activation writer.

## Finding 2 — Remove synchronous `storage.exists`

- **Verdict:** Recommend.
- **Status:** TODO
- **Prerequisite:** None
- **Evidence:** `FileStoragePort` exposes both asynchronous `stat` and
  synchronous `exists` at `libs/core/src/grafy_core/ports/storage.py:47`. S3
  `exists` performs synchronous remote `head` I/O at
  `libs/storage/src/grafy_storage/adapters/s3.py:123`. It is called from async
  workflows including `artifact_collections.py:456`, `tables.py:440`,
  `invocation_cache.py:106`, and artifact services at `services.py:1649`.
- **Problem:** Callers learn two interfaces for the same remote HEAD operation,
  and S3 presence validation can block the event loop.
- **Simpler representation:** Use `await storage.stat(...) is not None` as the
  sole public presence operation. Retain private synchronous existence logic
  only inside storage work already dispatched through `asyncio.to_thread`.
- **Smallest scope:** Core storage interface, Local/S3 adapters, four caller
  modules, storage fakes, and affected tests.
- **Risks:** Preserve "missing returns false" and contextual backend failures.
  Keep chunk checks sequential unless bounded concurrency is separately
  justified.
- **Validation:** Existing S3 stat tests, table accessibility tests, and
  cache-eviction tests cover most behavior. Add a fake exposing only async
  `stat` and assert async artifact/cache flows never call synchronous remote I/O.

## Finding 3 — Core-owned workspace Module definition resolution

- **Verdict:** Recommend.
- **Status:** TODO
- **Prerequisite:** None
- **Evidence:** The workspace-library module validates definitions at
  `libs/core/src/grafy_core/application/modules.py:323` using its
  implementation at `modules.py:333`. The HTTP catalog calls that listing and
  validates it again at `apps/api/src/grafy_api/v1/routes/catalog/services.py:75`,
  through a near-copy beginning at `services.py:134`. Pinned execution repeats
  the same path through `get_definition`.
- **Problem:** Two roughly 90-line implementations must agree on nested Modules,
  disabled edges, unknown operators, missing revisions, optional-to-required
  connections, and errors. Listing also repeats nested revision reads.
- **Simpler representation:** Deepen the workspace-library module with one
  canonical definition-resolution interface. The HTTP catalog becomes an
  adapter that maps results and errors rather than reimplementing validation.
- **Smallest scope:** Core `application/modules.py`, catalog `services.py`,
  composition wiring, compiler construction, and related tests.
- **Risks:** Preserve pinned unpublished/deprecated/withdrawn releases,
  list-time skipping of invalid definitions, and lightweight configurations
  where Modules are unavailable.
- **Validation:** Existing discovery, nested optional-input, invalid target, and
  withdrawn-pin tests are concentrated in `tests/unit/api/test_modules.py:588`.
  Add a core resolver contract and a counting repository assertion proving one
  nested-revision read per release.

## Finding 4 — Phase-local execution adjacency indexes

- **Verdict:** Recommend.
- **Status:** TODO
- **Prerequisite:** None
- **Evidence:** Topological sorting scans all edges for each dequeued node at
  `apps/api/src/grafy_api/v1/routes/executions/runtime/compiler.py:239`. The
  coordinator reconstructs incoming edges with an `O(VE)` comprehension at
  `coordinator.py:36`. Edge input assembly scans incoming edges again for every
  port at `edge_values.py:42`.
- **Problem:** A transport-friendly flat edge list is repeatedly treated as a
  lookup structure. Sparse DAG traversal becomes `O(VE)`, with additional
  `O(P × E_in)` port scans.
- **Simpler representation:** Build ordered outgoing, incoming, and by-port
  dictionaries once inside each owning phase. Preserve lists to retain request
  order and parallel-edge multiplicity. Do not introduce a generic graph
  framework or expand `CompiledGraph` for one caller.
- **Smallest scope:** Compiler, coordinator, edge resolver, and focused tests.
  No wire or persistence change.
- **Risks:** Preserve zero-indegree ordering, duplicate edges, external pinned
  predecessors, input-plug ordering, and deterministic variadic fan-in.
- **Validation:** Existing compiler ordering and pin coverage starts at
  `tests/unit/api/test_graph_compiler.py:70`; plug ordering tested at
  `test_edge_values.py:129`. Add cycle, fan-in/out, duplicate-edge, and large
  sparse-DAG equivalence cases without timing assertions.

## Finding 5 — Co-locate presence with registered room membership

- **Verdict:** Recommend.
- **Status:** TODO
- **Prerequisite:** Finding 1 first
- **Evidence:** The hub keeps independent `_rooms` and `_presence` maps at
  `apps/api/src/grafy_api/v1/routes/collaboration/hub.py:76`.
  `apply_presence_update` recreates a missing presence entry without proving the
  exact session is still registered and open at `hub.py:184`. Close removes both
  maps independently at `hub.py:501`.
- **Problem:** A delayed update can run after close, recreate presence, and
  leave an orphan participant. The already-closed session will not perform
  another cleanup.
- **Simpler representation:** Store one room-member value containing the
  registered session and optional presence. Under the hub lock, updates must
  match the exact open registered session. TTL expiry clears presence, not
  membership.
- **Smallest scope:** `hub.py` and graph-room tests; no wire change.
- **Risks:** Preserve join gating, rate limits, best-effort delivery, TTL
  semantics, and idempotent close.
- **Validation:** Keep current presence sequence/rate tests at
  `tests/unit/api/test_graph_room.py:770`. Add deterministic close-then-update
  and close/update-race tests proving presence cannot be recreated.

## Finding 6 — Deepen the authored-document reducer

- **Verdict:** Recommend.
- **Status:** TODO
- **Prerequisite:** None
- **Evidence:** Canonical command-driven invalidation already lives in
  `apps/web/src/features/workbench/model/graph-document.ts:261` and is applied
  atomically by `apps/web/src/features/workbench/canvas/graph-document-adapter.ts:60`.
  A legacy React Flow-shaped traversal remains at
  `apps/web/src/features/workbench/canvas/types.ts:554` and is invoked after
  semantic commands at `apps/web/src/features/workbench/ui/Workbench.tsx:2285`.
  Separately, every `applyGraphCommand` normalizes and deep-clones the whole
  document at `graph-document.ts:368`, even when the reducer applies a batch.
- **Problem:** A gesture can apply canonical invalidation and then overwrite
  overlays using stale renderer state. New command kinds must update multiple
  traversal policies. A `K`-command batch also performs `O(K × (V+E))` full
  normalization and configuration cloning.
- **Simpler representation:** Make the reducer the sole authored-document and
  overlay owner. Delete manual command-driven invalidation. Add an explicit
  transient upload-start reducer action. Normalize the starting document once
  per batch, then call a private canonical single-command transition; retain a
  public normalizing wrapper for standalone/room callers.
- **Smallest scope:** `graph-document.ts`, `graph-document-adapter.ts`,
  `canvas/types.ts`, `graph-authoring.ts`, `Workbench.tsx`, and focused tests.
- **Risks:** Preserve pre-command invalidation roots, disabled-edge behavior,
  delayed bound-edge commits, upload `uploading`/failure transitions,
  hostile-field stripping, batch atomicity, and no-alias/deep-clone guarantees.
- **Validation:** Retain canonical command and reducer tests at
  `graph-document.test.ts:481` and `graph-document-adapter.test.ts:114`. Replace
  legacy helper tests with reducer-level edge, upload, hostile payload, and
  sequential-versus-batch equivalence tests.

## Finding 7 — Transition-owned execution lifecycle

- **Verdict:** Recommend after an explicit state-design pass.
- **Status:** TODO
- **Prerequisite:** Explicit state design
- **Evidence:** `_RunExecutionRecord` stores correlated `status`, `task`,
  `result`, `error`, and `retained_terminal` fields at
  `apps/api/src/grafy_api/v1/routes/executions/runtime/manager.py:224`.
  Transitions are distributed across cancellation at `manager.py:432`, run
  startup at `manager.py:503`, callbacks, and completion at `manager.py:572`.
  Durable status/timestamps are separately mutable at
  `libs/core/src/grafy_core/domain/execution_history.py:47`.
- **Problem:** Constructible states include running without a start timestamp,
  queued with a start timestamp, terminal records with live tasks/active nodes,
  or success with an error. Later mutation bypasses construction-time checks.
- **Simpler representation:** Give durable execution explicit transition
  operations that advance status, timestamps, workflow identity, and error
  together. In the runtime record, distinguish active state from one typed
  terminal outcome; avoid a class hierarchy per wire status and delete redundant
  terminal-retention state. Keep flat rows and responses at
  persistence/presentation seams.
- **Smallest scope:** Manager, execution-history module, execution-history
  adapter, result presentation, and lifecycle tests.
- **Risks:** Preserve cancel-versus-complete winners, failed partial results,
  cancellation with a completed result, history retry/reconciliation, journal
  sealing, room publication, lease release, and shutdown callback races.
- **Validation:** Existing lifecycle/race contracts at
  `tests/unit/api/test_execution_manager.py:380`, `:981`, and `:1045`. Add a
  legal transition table, exactly-one-terminal property checks, and agreement
  across snapshot, journal, room message, and durable history.

## Finding 8 — One frontend execution-observation lifecycle

- **Verdict:** Recommend.
- **Status:** TODO
- **Prerequisite:** Prefer finding 6 first
- **Evidence:** Locally started runs construct an observation guard at
  `apps/web/src/features/workbench/ui/useRunExecution.ts:476`, then own SSE,
  polling, and terminal logic. Room-discovered runs reconstruct the same state
  at `useRunExecution.ts:1495`, with a second SSE subscription at line 1547 and
  second poll loop at line 1721.
- **Problem:** The implementations already differ: local `404/410` handling
  clears busy node overlays, while the shared path beginning at `:1745` clears
  activity state but can leave nodes queued/running/cancelling.
- **Simpler representation:** Separate run initiation/preflight from one
  observer keyed by `execution_id`. Feed either a local-start or room-discovery
  seed into one closed observation phase: idle, observing, terminal, or
  unavailable. One implementation owns SSE ordering, polling, cancellation,
  progress batching, terminal projection, and cleanup. Keep the concrete HTTP
  adapter and the accepted SSE-plus-poll design.
- **Smallest scope:** `useRunExecution.ts`, a cohesive sibling observation
  module/hook, its test file, and the single Workbench composition call.
- **Risks:** Preserve graph-generation guards, terminal-SSE precedence over
  stale polls, progress caps, nested Module status aggregation, room-clear
  behavior, cancellation restoration, and origin-specific announcements. A
  matching room announcement must not create a second subscription.
- **Validation:** Existing local and shared contracts are in
  `useRunExecution.test.ts:297`. Add parameterized local/shared seeds,
  matching-room single-subscription, and shared-unavailable overlay cleanup.

## Finding 9 — Execution history owns active-run uniqueness

- **Verdict:** Recommend as a migration after lifecycle transitions are explicit.
- **Status:** DONE (2026-08-23, migration `0013_thin_execution_schema`)
- **Prerequisite:** Design with finding 7; implement after transitions
- **Evidence:** `graph_executions.status` already records active states at
  `libs/persistence/src/grafy_persistence/schema.py:565`. A second
  `graph_active_execution_slots` table repeats workspace, graph, and execution
  identity at `schema.py:1190`, without an execution-row foreign key. Start,
  completion, and recovery coordinate both representations at
  `apps/api/src/grafy_api/v1/routes/executions/services.py:93`.
- **Problem:** A slot can reference a missing or terminal execution; an active
  execution can lack a slot; multiple active execution rows can coexist while
  one slot masks them.
- **Simpler representation:** Add a partial unique index on
  `(workspace_id, graph_id)` for `queued`, `running`, and `cancelling`. Add an
  execution-history query for the active run, then remove the slot table, model,
  interface methods, in-memory dictionary, and collaboration-owned deletion
  check. This passes the deletion test.
- **Smallest scope:** New migration, execution-history repository/interface,
  start/completion/recovery, graph-deletion gate at
  `libs/core/src/grafy_core/application/collaboration.py:901`, in-memory adapter,
  and affected tests.
- **Risks:** Reconcile duplicate active rows and slot/status disagreement before
  index creation. Handle SQLite/PostgreSQL index syntax, precise
  integrity-error translation, downgrade backfill, and
  terminal-then-next-start races.
- **Validation:** Add a real two-transaction race, terminal release, startup
  interruption, deletion rejection, migration-conflict fixtures, and
  dialect-specific partial-index coverage.
- **Resolution:** Implemented as recommended. The partial unique index
  `uq_graph_executions_one_active_per_graph` on `graph_executions(workspace_id,
  graph_id) WHERE status IN ('queued','running','cancelling')` is the sole
  authority (same SQL predicate on SQLite and PostgreSQL). Conflicting starts
  surface as translated integrity errors reporting the existing execution id;
  the migration validates duplicates up front and fails without choosing a
  winner; the downgrade recreates slots from active executions; deletion gates
  through `execution_history.find_active_execution_id`; two-transaction races,
  terminal release, startup interruption, and deletion rejection are covered by
  tests (`tests/unit/persistence/test_execution_history_persistence.py`,
  `tests/unit/persistence/test_migrations.py`).

## Finding 10 — Keep generated-node ports typed throughout the domain module

- **Verdict:** Recommend, lower priority.
- **Status:** TODO
- **Prerequisite:** None
- **Evidence:** `GeneratedNodePort` already owns name/type/shape validation at
  `libs/core/src/grafy_core/domain/generated_nodes.py:97`.
  `GeneratedNodeRelease` instead stores eight flat port fragments at
  `generated_nodes.py:145`, reconstructs ports through properties at line 231,
  and decomposes them again during creation. Persistence mirrors the flat
  columns at `libs/persistence/src/grafy_persistence/schema.py:1287`.
- **Problem:** Callers must know both the typed and flattened shapes. Every
  construction decomposes ports, while validation, digests, runtime contracts,
  and the runner reconstruct them.
- **Simpler representation:** Store `input_port` and `output_port` directly on
  `GeneratedNodeRelease`. Let a persistence adapter alone flatten/reconstruct
  the existing columns. This improves locality without changing the database or
  HTTP response.
- **Smallest scope:** Generated-node domain model, operator, runner/response
  call sites, persistence mapper, and generated-node tests.
- **Risks:** Preserve digest bytes exactly, immutable revision identity, existing
  response fields, and validation during ORM hydration.
- **Validation:** Add fixed digest fixtures, invalid-port construction, both port
  shapes/types, persistence reload/list, runner access, and
  response-compatibility tests.

## Finding 11 — Make artifact content a discriminated value

- **Verdict:** Recommend as a staged cross-cutting migration.
- **Status:** TODO
- **Prerequisite:** Finding 2 first
- **Evidence:** `ArtifactObject` independently exposes `storage_backend`,
  nullable bucket/key, and nullable inline payload at
  `libs/core/src/grafy_core/artifacts.py:189`. Persistence permits all
  combinations at `libs/persistence/src/grafy_persistence/schema.py:485`, with
  direct mapping at `orm.py:109`. Writers set one interpretation while
  resolvers such as `runtime/resolvers.py:102` and cache/artifact consumers
  infer another from nullability.
- **Problem:** Both, neither, incomplete stored locators, inline-with-locator,
  stored-with-inline, and arbitrary backend combinations are constructible and
  persistable.
- **Simpler representation:** Replace the correlated fields with one
  `ArtifactContent` value: `InlineArtifactContent(payload)` or
  `StoredArtifactContent(backend, bucket, object_key)`. Keep identity, content
  type, size/hash, and metadata on `ArtifactObject`. Persistence alone
  translates the value to flat columns.
- **Smallest scope:** Core model/writers/resolvers, explicit flat-row persistence
  mapper, API artifact/cache consumers, table/collection handling, and direct
  GIS/OCR/LLM consumers. Follow with a migration that reconciles legacy rows and
  adds enforceable XOR/discriminator checks.
- **Risks:** Do not guess ambiguous deployed rows. Preserve valid inline
  hashes/sizes, table logical-content integrity semantics, unknown backend
  policy, SQLite rebuild/downgrade behavior, and historical plugin artifacts.
- **Validation:** Add a complete invalid-combination matrix, both variant round
  trips, legacy-row migration fixtures, inline/stored streaming, projection,
  cache staleness, table/collection integrity, and plugin-adapter tests.

---

## Demoted or rejected candidates (not tracked)

- Auth reservation sets → counters: bounded bookkeeping, not material.
- SQL alias/table pair: worker already creates paired aggregate immediately.
- Nested LLM structured result: validation rejects inconsistent combinations;
  a `llm.completion@2` migration adds more machinery than it removes.
- GIS model reuse: host intentionally does not depend on the optional GIS plugin.
- Saved-graph HTTP/domain DTO consolidation: retained because it protects the
  public OpenAPI/generated-client seam.
- Generic graph indexes on the frontend: rejected without profiling or a
  graph-size contract.
- File-size-driven splits of `Workbench.tsx`, `WorkflowNode.tsx`, renderers, or
  artifact modules: rejected.
- Collapsing inline/Prefect or SQL parent/worker/runtime adapters: retained as
  real multi-adapter and security seams.

## Rule gap to add (ADR-level)

Once a transport's serialized outbound sender is activated, every later message
must use it; direct transport writes are allowed only in an explicitly named
pre-activation handshake. `[R23: Maintain The Rules]`
