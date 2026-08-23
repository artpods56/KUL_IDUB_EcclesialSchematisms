# Backend Simplification Audit — What Is Hard to Justify

> **Implementation update (2026-08-23).** The database-thinning refactor has
> landed: `graph_execution_idempotency`, `graph_command_journal`, and
> `graph_active_execution_slots` are deleted, and
> `graph_execution_requested_nodes` + `graph_execution_node_results` are merged
> into one monotonic `graph_execution_nodes` table (§3.3, §3.4). The schema now
> defines exactly **27 application tables**. Three of this audit's deletion
> recommendations were **rejected by product decision and must not be treated
> as approved work**: `personal_access_tokens` are retained for near-term
> external automation and MCP access (§3.2), `user_graph_states` is retained
> planned product state for favorites and recently-opened graphs (§3.5), and
> `oidc_bootstrap_owner_mappings` is retained for its LDAP-federated
> bootstrap-owner security purpose — ADR 0003 records that decision and this
> audit's §3.7 "first login owns local" recommendation is rejected. The
> per-table verdicts below reflect the tree as audited and are kept for
> provenance; where they describe deleted or retained tables, the update above
> wins.

>
> Evidence-based audit of the Grafy backend looking for things that are
> **hard to justify at this moment**: tables, subsystems, protocol elements,
> configuration, and documentation that cost more than they earn. Every claim
> cites file:line evidence from the current tree. No code was changed.
>
> Method: five parallel subsystem audits (per-table consumer maps, collaboration
> protocol, execution runtime, identity/auth, structure/config/docs) plus
> independent verification of headline claims against the working tree and the
> live dev database (`.notarius-artifacts/workbench/notarius.sqlite3`, read-only).
>
> Companion docs: `REFACTOR_PLAN.md` (prior code-quality audit, mostly done),
> `SMELL_REVIEW.md`, `SOLID_REVIEW.md`, `RELEASE_AUDIT_REPORT.md`. This report
> deliberately does **not** repeat those — it asks a different question: what
> should be **deleted, merged, or demoted**, and what may we **stop carrying**?

---

## TL;DR

1. **The "33 tables" number is real and mostly bad.** 31 tables in
   `schema.py` (+`alembic_version`). Of the 31: **2 are dead**, **1 is
   write-only**, **2 are mergeable**, **1 is a one-shot ceremony**, **1 is a
   full credential type with zero consumers**, and **1 is a fully-built feature
   the UI never ships**. The rest are genuinely load-bearing.
2. **Prefect is the biggest unjustified external system.** A pinned container,
   a ~12 MB dependency, 4 env vars, a readiness coupling, and ~950 code/test
   lines — for a backend whose one real feature (concurrent MAP items) is
   already implemented backend-neutrally and merely disabled for inline by a
   3-line special case (`services/composition.py:181-183`).
3. **PATs are issue-only credentials.** Their only consumer (the MCP server)
   was deleted on 2026-08-19; no route accepts a Bearer token, no UI manages
   them, and the parser + digest lookup have zero call sites. ~1,500 lines
   including tests.
4. **The collaboration cluster (6 tables, one migration, ~20% of the schema)
   contains 2 dead + 1 write-only table.** The live protocol (head + receipts +
   checkpoints + epoch + heartbeat) is small and justifiable for the pilot; the
   surplus (journal, execution-idempotency, active-slots) is not.
5. **Unbounded growth on four table families** with no retention job:
   `graph_executions`/node results, `saved_graph_revisions` (a full document
   copy per checkpoint), `staged_uploads` (rows **and** disk files),
   `security_audit_events`. Only the auth trio has a working cleanup loop.
6. **Retired features left husks and stale docs**: empty route dirs with stale
   `.pyc` files, an empty `libs/agent` package, a stale `grafy_mcp` editable
   install in the venv, and a large MCP doc-drift set (`.env.example` still
   documents the removed `/mcp` endpoint; README still claims "no
   authentication").
7. **Config surface**: 45 Settings variables + ~13 names outside Settings;
   exactly **one** is truly mandatory (`GRAFY_COMMAND_HMAC_KEY`). A 2-line
   `.env` boots the whole app. `execution_backend` defaults to `prefect` —
   a trap for anyone following a simple install.

**Bottom line:** the workbench core (graphs, artifacts, execution engine,
secrets, OIDC login, rooms) is well-justified. The excess is concentrated in
*protocol surplus* (collab journal/receipts split, execution idempotency),
*dead credentials* (PAT), *one external service* (Prefect), *one-shot
ceremony* (bootstrap owner mapping), *unshipped feature surfaces*
(star/archive/folders), and *retirement debt* (husks, docs, venv).

---

## 1. The 31 tables, one by one

Classification: **CORE** (heavily used, needed) · **FEATURE-TIED** (one
feature; feature ships and is used) · **DEAD** (no production consumer) ·
**WRITE-ONLY** (written, never read) · **MERGE** (redundant with a sibling).

| Table | Verdict | One-line justification |
|---|---|---|
| `users` | CORE | Identity anchor; 9 tables FK to it; checked on every request (`auth/services.py:393-401`). |
| `oidc_identities` | CORE | `(issuer, subject)` → user; the only external identity mapping key. |
| `oidc_login_transactions` | CORE | PKCE handshake state; TTL 300 s; well-bounded, has a retention job. |
| `workspaces` | CORE (structural) | Every content table FKs to it; every URL is workspace-prefixed. Collapsing it is the one big-bang refactor in the codebase. For a single owner it is a 2-row constant. |
| `workspace_memberships` | FEATURE-TIED (team sharing) | Per-request authorization anchor (`application/identity.py:410-432`); 3 roles / 20 capabilities. If the product stays single-owner, this is the largest single simplification available — and the one that kills multi-user rooms. |
| `auth_sessions` | CORE | The only live credential the UI uses (get + logout). The session **list/revoke** endpoints (`auth/views.py:193-240`) have no UI. |
| `personal_access_tokens` | **DEAD (consumer-less)** | Full lifecycle (table, 3 endpoints, scopes, revocation cascades, ~600 test lines) issuing tokens nothing in the tree can use — §3.2. |
| `security_audit_events` | **WRITE-ONLY** | ~30 write sites; `list_for_workspace`/`delete_before` have zero production callers; no retention job → unbounded growth — §3.6. |
| `oidc_bootstrap_owner_mappings` | **ONE-SHOT** | One row, consumed at first login, permanently empty afterwards; table + CLI + env var for a ceremony a column could do — §3.7. |
| `saved_graphs` | CORE | Primary aggregate; FK anchor for 9 tables. |
| `saved_graph_revisions` | CORE (pinning) / **dead browsing surface** | FK target of 4 tables (executions, materialized outputs, checkpoint mappings, module releases). But `list_revisions` (port/application/repo) has **zero route callers** — revision history is not exposed anywhere. Each row is a **full document copy**: the schema's biggest per-graph growth vector — §3.9. |
| `graph_folders` | FEATURE-TIED (templates) | Exactly one UI read: target-folder picker in `TemplateLibrary.tsx:129-130`. The create/rename/delete endpoints (~250 LOC) have no UI. |
| `graph_organizations` | FEATURE-TIED (templates) | One live write path (template instantiation `application/templates.py:175-181`); `archived` is mapped in the browser payload but never rendered. |
| `user_graph_states` | **UNSHIPPED FEATURE** | Star + last-opened: 1 table + 4 endpoints + ~120 LOC; `use-api.ts:86-88` maps the fields and **no component reads them**. Ship the UI or cut it — §3.5. |
| `artifact_objects` | CORE | Every operator, resolver, and the artifact routes. |
| `invocation_cache_entries` | FEATURE-TIED (perf) | Pure memoization for `exact`-policy nodes; self-contained and droppable; outputs JSON is a full artifact-ref copy. |
| `materialized_node_outputs` | CORE | The "latest output" pins the whole run-reuse model is built on. |
| `graph_executions` | CORE (no retention) | Execution history the UI shows; **no DELETE exists anywhere in production** — §3.9. |
| `graph_execution_requested_nodes` | **MERGE** | Strict column-prefix of `graph_execution_node_results` (identical PK shape + `position`) — §3.3. |
| `graph_execution_node_results` | CORE | History detail rows. |
| `node_secrets` | CORE | Encrypted write-only credentials; enforced AAD binding; used by LLM/SQL plugins. |
| `staged_uploads` | FEATURE-TIED (uploads) | Shadow of the disk: one production read is an existence check (`core/staged_upload_paths.py:58`); `list`/`remove` have zero production callers; rows **and** files grow unbounded — §3.9. |
| `collaborative_graph_heads` | CORE (collab) | Live head between checkpoints; every mutation routes through it; startup invariant `verify_every_graph_has_head` (`main.py:337`). `name`+`document` duplicate `saved_graphs` — §3.8. |
| `graph_command_journal` | **WRITE-ONLY** | Full payload persisted per command; port declares only add/clear; **no read path exists** — §3.4. |
| `graph_command_receipts` | CORE (collab) | The idempotency that actually works (HMAC compare, `application/collaboration.py:339-371`); tombstones survive journal clears. `outcome` column is dead (always `accepted`). |
| `graph_checkpoint_mappings` | FEATURE-TIED (collab) | Checkpoint-retry idempotency; small rows; keep. |
| `graph_execution_idempotency` | **DEAD** | Full vertical slice (schema + port + repo + domain + tests), **zero application callers** — §3.4. |
| `graph_active_execution_slots` | **MERGE** | 4-column mutex duplicating `graph_executions.status`, FKs into the collab head (not the execution row) — §3.3. |
| `modules` | CORE | Workspace library; live feature with UI. |
| `module_releases` | CORE | Pinned immutable releases; FK target of module pins. |
| `templates` | CORE | Copy-by-value snapshot source; deliberate `snapshot_document` duplication (no FK to source — survives source deletion). |

**Score: 17 CORE · 6 FEATURE-TIED · 2 DEAD · 1 WRITE-ONLY · 2 MERGE ·
1 UNSHIPPED-FEATURE.** The collaboration cluster alone (all 6 tables created in
one migration, `0009_collaborative_graph_heads.py`) accounts for 2 dead +
1 write-only + 1 merge.

---

## 2. What is genuinely justified (do not cut)

So the report isn't just a deletion list — these were examined and **kept**:

- **`libs/core/runtime` vs API split** — load-bearing: operators and all four
  plugins import `grafy_core.runtime.persistence/resolvers` and must not depend
  on the graph engine; `NodeRuntime` + `InMemoryUnitOfWork` is what lets
  `scripts/smoke_workbench.py` and unit tests run nodes without the API.
- **The room protocol core** — head + receipts + checkpoints + epoch +
  heartbeat. Each solves a documented corruption class
  (`RELEASE_AUDIT_REPORT.md`), each is small, and the frontend genuinely uses
  co-editing (`Workbench.tsx:1444`) and presence (`:3891`).
- **OIDC + PKCE + cookie sessions** — the right size for what it does:
  digests-only storage, AEAD-encrypted PKCE verifier, in-process abuse
  control (appropriate for one process), login-rotation.
- **`repositories.py` (1,986 lines) / `schema.py` (1,271 lines)** — big but
  per-aggregate cohesive (prior SOLID review: 0 violations).
- **The SSE/replay machinery in `manager.py`** — 100% backend-neutral, small
  (118-line bounded journal + Last-Event-ID resume), justified by live progress.
- **The test suite (123 files / ~41k lines)** — no dead test code for any
  retired feature; it is the reason deletions below are cheap to verify.

---

## 3. Ranked findings

Tier 1 = delete now (low risk, nothing functional lost).
Tier 2 = merge/simplify (medium effort, real migration involved).
Tier 3 = product judgment (big bet, capability tradeoff).

### Tier 1

#### 3.1 Delete the Prefect backend — *the single biggest win*

**Evidence.** Production compose runs a pinned Prefect container
(`infra/docker/compose.yaml:62-88` + `prefect-data` volume `:187-188`);
`apps/api/pyproject.toml:18` pins `prefect==3.6.21` (~12 MB wheel); 4 env vars
(`.env.example:47-48,64`; `infra/docker/.env.production.example:6,22,26`);
`/ready` probes Prefect health (`health.py:29-40`). The engine is one local
adapter: one `@flow` per run, `persist_result=False`, `cache_policy=NO_CACHE`
(`runtime/prefect.py:75-76,121-122,173`) — CONTEXT.md:412-414 states Grafy
remains the source of truth for everything Prefect could persist.

**What it actually buys.** (a) Concurrent MAP items — but the machinery
(`asyncio.TaskGroup` + `Semaphore`, `runtime/node_execution.py:218-240`) is
backend-neutral; inline is forced to 1 by exactly
`composition.py:181-183`. (b) Task retries — default **0**
(`settings.py:74`). (c) An external secondary UI — zero references in
`apps/web/src`.

**Delete:** `runtime/prefect.py` (234), composition/main/settings/health wiring,
compose service + volume + env, the pinned dependency,
`tests/unit/api/runtime/test_prefect_execution_engine.py` (714) and the
prefect cases in `test_settings.py`/`test_readiness.py`, the Justfile `prefect`
recipe, and ~15 doc references. Integration execution tests already run inline
and keep passing.

**Keep/adjust:** the `workflow_run_id` column (works with inline uuids), the MAP
semaphore — and **remove** the `composition.py:181-183` special case so inline
honors `GRAFY_MAP_MAX_CONCURRENCY` (default 4). Nothing is lost.

**Why it's hard to justify today:** the architecture is already single-process
(`single_owner.py:26-33`, compose `WEB_CONCURRENCY: "1"`), so every "reason"
Prefect offers (horizontal workers, distributed tasks, durable results) is
switched off by the rest of the design. It is a container-shaped ceremony.

#### 3.2 Delete the PAT layer

> **Update (2026-08-23): retained.** PATs are intentionally kept — table,
> issuance/list/revocation HTTP surface, scopes, and revocation semantics — for
> near-future external automation and MCP access. No bearer consumer is
> implemented yet; token digests and revocation behavior are unchanged. This
> section is no longer approved deletion work.

**Evidence.** The only bearer consumer was the MCP server, removed 2026-08-19
(`9d3551c`, `250aa51`, `33e9f93`). Now:
- `browser_actor` **rejects any `Authorization` header** with 401
  (`auth/dependencies.py:29-39`);
- `_parse_personal_access_token` has **zero call sites**
  (`auth/services.py:1021`);
- `get_personal_access_token_by_digest` has zero **production** call sites
  (port `ports/identity.py:116`, repo `repositories.py:335` — tests only);
- the frontend has **no PAT UI** (generated types only, `grafy.ts:836-865`);
- the dev DB has 0 rows;
- the layer is internally inconsistent: domain allow-set is 12 capabilities,
  the HTTP scope enum exposes 11 (`create_template` unreachable,
  `auth/models.py:82-93` vs `domain/identity.py:86-101`);
- `.env.example:50-55` still tells operators to authenticate a `/mcp` endpoint
  with a PAT.

**Delete:** table, 3 endpoints (`workspaces/views.py:310-397`),
`IdentityService` PAT methods + revocation cascades
(`application/identity.py:91-199,492,551,606,648-657,696-719`),
`issue/_parse` (`auth/services.py:347-369,1020-1027`), scope model
(`domain/identity.py:86-101`), 2 settings, ~600 test lines, stale docs.
≈1,500 lines, one fewer credential type to reason about.

**Risk:** none in-tree. Re-open only if an external-automation credential comes
back.

#### 3.3 Delete the dead collaboration/execution tables

> **Done (2026-08-23).** All three items below are implemented in migration
> `0013_thin_execution_schema`. The merged table is `graph_execution_nodes`
> with a monotonic requested → terminal row lifecycle; it keeps two explicitly
> named positions (`position` = request order, `result_position` =
> compiled-plan result order) because the two genuinely differ.

- **`graph_execution_idempotency`** — complete vertical slice
  (`schema.py:1117-1138`, `ports/collaboration.py:82-91`,
  `repositories.py:1720-1757`, `domain/collaboration.py:896-909`) with **zero
  application callers** (grep across apps/libs/plugins + tests). The idempotency
  the protocol needs already lives in `graph_command_receipts`. Delete table +
  4 methods + model + tests. Nothing lost.
- **`graph_active_execution_slots`** → replace with a **partial unique index**
  on `graph_executions(workspace_id, graph_id)` for
  `status IN (queued, running, cancelling)` + an active-run query. This is open
  REFACTOR_PLAN finding 9 (evidence at `REFACTOR_PLAN.md:498-527`): the slot
  duplicates `graph_executions.status` (`schema.py:523`) with no FK to the
  execution row (`schema.py:1141-1156`), and FKs into the *collaboration* head —
  so a generic execution invariant is structurally owned by the collab feature.
  Rewires `executions/services.py:95-120,190,245` and the delete gate
  (`application/collaboration.py:901`). Watch the noted risks (reconcile
  duplicates first, partial-index dialects, terminal-then-next-start race).
- **`graph_execution_requested_nodes`** → **merge into
  `graph_execution_node_results`**. The requested table is a strict column
  prefix (identical PK + `position`, `schema.py:563-596` vs `599-633`). One
  `graph_execution_nodes` table with `status` + nullable outcome columns serves
  every read in `repositories.py:1091-1403`; the in-memory UoW *already* models
  it that way (requested ids live on the `GraphExecution` entity,
  `domain/execution_history.py:55`). Caveat: `position` semantics differ
  (request order vs topological order) — pick request order. Bonus: fixes the
  SQL/in-memory asymmetry and removes the cross-check query
  (`repositories.py:1157-1170`).

#### 3.4 Delete the command journal (or wire a replay consumer)

> **Done (2026-08-23).** The journal is deleted; complete collaborative head
> snapshots are the recovery path and no incremental replay store is retained
> (see ADR 0002). Receipts keep their HMAC idempotency semantics.

**Evidence.** Written at 3 sites (`application/collaboration.py:267,440,849`),
cleared on epoch reset (`:629`), repository add/clear only
(`repositories.py:1613-1641`). The port (`ports/collaboration.py:23-115`)
declares **no read method**; grep finds no read path anywhere. Reconnect
rehydration works via `room.ready` head snapshots, not journal replay
(`collaboration/views.py:151-224`; `graph-room-session.ts:559-565`). Every
accepted command's **full payload** is persisted forever and never read — the
largest per-row cost in the collab cluster, on top of the payload already living
in the head and (as an HMAC) in receipts.

**Decision.** If an offline replay/forensics feature is *not* planned: delete
table + 3 build sites + port/repo methods + tests (~M effort). If it *is*
planned: the receipts are the wrong home for it (they carry no payload) —
store the payload in the receipt row and keep exactly one durable command
record. Either way, the current "write a second full copy, never read it"
state is not justifiable. (Note: if you delete the journal, do it **with or
after** the payload-into-receipt move in §3.8, or payload-level replay data is
lost forever.)

#### 3.5 `user_graph_states` + the star/archive surface: ship or cut

> **Update (2026-08-23): retained.** `user_graph_states` is intentionally kept
> as planned product state for personal favorites and genuinely user-specific
> "Recently opened" graphs. It remains per-user state and never grants access.
> The frontend feature is still pending; this section is no longer deletion
> work.

**Evidence.** Fully built below the hook: table (`schema.py:401-433`), 4
endpoints (`saved_graphs/views.py:194-236`), service
(`application/saved_graphs.py:392-458`), repository
(`repositories.py:817-831`). The browser lists `starred`/`archived`/
`last_opened_at` (`use-api.ts:86-88`) and **no component reads any of them**;
nothing calls the star/opened endpoints. `graph_folders` and
`graph_organizations` are in the same boat: exactly one live UI consumer
(template target-folder picker, `TemplateLibrary.tsx:129-130,223`), no
create/rename/delete/archive UI.

**Options.** (a) Ship the star/archive/folder UI (the backend is done — this
becomes a frontend task and the tables become justified); (b) cut
`user_graph_states` + endpoints + ~120 LOC and trim the folder/archive surface
to what templates need. Leaving "fully built, fully unused" in the tree is the
worse third option.

#### 3.6 `security_audit_events`: wire a reader + retention (or delete ~30 sites)

**Evidence.** ~30 write sites (`auth/services.py`, `main.py:87-218`,
`application/{identity,collaboration,saved_graphs}.py`); `list_for_workspace`
and `delete_before` (`repositories.py:427-453`) are called **only by tests**;
the periodic cleanup (`main.py:340-357` → `auth/services.py:607-623`) covers
login transactions/sessions/PATs — **not** audit rows; the
`ix_security_audit_events_retention` index is a vestige of a retention feature
that was never wired. Dev DB: 34 rows in ~19 h, mostly failed pre-auth probes —
and the growth never stops.

**Fix (cheap):** call `delete_before` from the cleanup loop with a
`security_audit_retention_days` setting (~20 lines) and add one
owner-readable endpoint (reusing `list_for_workspace`) or explicitly accept
`sqlite3`-level inspection. It is the only table in the schema that is an
append-only log with neither horizon nor reader.

#### 3.7 Replace the bootstrap-owner ceremony with "first login owns `local`"

> **Rejected (2026-08-23).** This recommendation is not approved work.
> Production users are federated from an LDAP-backed identity provider, and an
> operator must be able to preselect the exact federated identity that becomes
> administrator by stable `(issuer, subject)` — not email, display name, or a
> "first user" rule. ADR 0003 records that decision; the mapping stays.

**Evidence.** `oidc_bootstrap_owner_mappings` gets exactly one row
(`grafy-admin bootstrap-oidc-owner`, `admin.py:18-27`); the matching identity's
first login consumes it and the table is **permanently empty**
(`application/identity.py:300-351,721-750`). Dev DB row:
`consumed_at=2026-08-17`. What it costs: 1 table, 1 admin command, 1 env var
(`GRAFY_OIDC_BOOTSTRAP_SUBJECT`), 1 Justfile recipe, ~120 lines, 1 operator
step per deployment.

**Fix:** auto-ownership — if the seeded `local` workspace has no owner, the
first login gets it. (Same shape as option (b) of §3.5: a one-shot table
serving a migration-era ritual.)

#### 3.8 The collaboration head duplicates the live document

**Evidence.** `collaborative_graph_heads.name+document`
(`schema.py:997-998`) duplicate `saved_graphs.name+document`
(`schema.py:332-333`). The head is the **live** source of truth between
checkpoints (`application/collaboration.py:405-411`); `saved_graphs` holds the
last-checkpointed copy and is stale between checkpoints. Two different endpoints
read the two copies and can disagree (`/me/graphs` joins the head,
`repositories.py:571-576`; the workspace list reads `saved_graphs`,
`models.py:589-597`). Every graph therefore carries ≥2 full document copies.

**Fix:** keep `name`+`document` only on the head; reduce `saved_graphs` to
identity columns (the FK anchor it already is for 9 tables). Readers join the
head — the 8-table browser join already does this, and the startup invariant
guarantees a head exists. Moderate refactor of `application/saved_graphs.py`
(696 lines) + the ORM aggregate. (Do **not** merge head ↔ revisions: the
head is live-mutated, revisions are immutable FK targets for 4 tables.)
**Follow-on:** move the command payload into the receipt row and drop the
stored HMAC + `GRAFY_COMMAND_HMAC_KEY` — one fewer long-lived deployment
secret to generate/backup (release audit calls the keys out explicitly),
`hmac.compare_digest` → payload equality (equivalent for replay detection;
loses key-version rotation). Do this **before or with** the journal deletion.

Also drop the dead `outcome` column on `graph_command_receipts` (always
`accepted` — `application/collaboration.py:368-371` computes the outcome
in-memory and never stores it).

### Tier 2 (structural, no capability change)

#### 3.9 The unbounded-growth tables need a retention decision

| Family | Why it grows forever | Note |
|---|---|---|
| `graph_executions` + node results | No DELETE exists in production code (grep: only `graph_active_execution_slots` is ever deleted). | Execution history is a product feature — it needs a policy, not a code path. |
| `saved_graph_revisions` | One **full document JSON** per checkpoint; no retention; the browsing surface (`list_revisions`) has zero route callers. | Biggest per-graph storage vector. Decide: expose the history UI or bound revisions. |
| `staged_uploads` (+ the files on disk/S3) | The only production read is an existence check (`core/staged_upload_paths.py:58`); `list`/`remove` have zero production callers; no cleanup job. The release audit already flags "no safe TTL or workspace quota" (`RELEASE_AUDIT_REPORT.md:148-150`). | Either promote the lifecycle (staged → referenced → reclaimable) or at minimum add age-based retention that respects in-flight graph references. |
| `security_audit_events` | §3.6. | |

The only family with a working retention job is the auth trio
(`main.py:340-357`). Four families with nothing is not a design; it's an
omission.

#### 3.10 Move the execution engine out of the route package

`v1/routes/executions/` is a 5,444-line "route module" whose 3,916-line
`runtime/` subpackage (compiler, manager, coordinator, preflight, …) **is the
execution engine**, reachable only through an HTTP package. Its former home
`grafy_api/services/execution/` is an empty directory holding 14 stale `.pyc`
files. Move `runtime/` to `services/execution/` (imports in `main.py`,
`services/composition.py`, `tests/unit/api/runtime/*`). High benefit (a real
route becomes 1,527 lines; the engine stops hiding under HTTP), medium effort.

#### 3.11 Stop applying the 4-file route pattern mechanically

13 route dirs, 11 live, ranging from 291 to 5,444 lines. Five
`dependencies.py` files (16–20 lines each) are a single 2-line factory;
`templates` (295), `uploads` (291), `modules` (407) have no services at all;
`catalog/views.py` is a 33-line shell. Merge templates/uploads/modules into one
file each, node_secrets into two, fold catalog views in. Keep the pattern only
where it earns it (artifacts, auth, executions, saved_graphs, collaboration).
Low effort, medium benefit (6 near-empty files gone).

#### 3.12 Clean the retirement debt (one afternoon)

- `v1/routes/agent_authoring/` and `v1/routes/generated_nodes/`: source-free
  husks holding only stale `.pyc` files (the code lives on side branches
  `codex/generated-node-prototype-snapshot`, `codex/agent-node-environments`).
- `libs/agent/`: pycache-only package (pydantic-ai remnants), not a workspace
  member.
- `apps/mcp` removal leftovers: stale `grafy_mcp` editable install in the venv
  (`.venv/.../__editable__.grafy_mcp-0.1.0.pth` → deleted directory); resync.
- Docs drift set (all verified): `.env.example:50-55` (removed `/mcp` endpoint
  + "use a PAT" instructions); `README.md:426-428` ("does not yet have
  authentication" — false, full auth stack exists); `README.md:76-77` ("no
  worker service" — reconcile with the compose `prefect` service);
  `README.md:454-456` (`api-plugins` image also includes GIS+LLM, per
  `api.Dockerfile:37-41`); MCP references in `docs/design/backend-architecture.md`
  (describes nonexistent `apps/mcp`), `docs/design/authentication-and-workspace-tenancy.md`
  (23 hits), `docs/design/workbench-realtime-collaboration.md` (22),
  `docs/plans/authentication-workspace-refactor.md` (44, incl. obsolete Phase 6),
  ADRs 0002/0003 (32 total), `SMELL_REVIEW.md`/`SOLID_REVIEW.md` (review scope
  lines).
- `CONTEXT.md:260-289` "Plugin release / Plugin root" vocabulary: **pure
  design, zero implementation** (only the untracked
  `docs/design/plugin-unification.md`). CONTEXT.md is honest about this at
  `:272-277`, but the vocabulary table presents it as current product
  vocabulary. Mark it clearly as intended-but-unbuilt or move it to docs/.

#### 3.13 Config surface: document the minimal install, fix the trap

45 Settings vars + ~13 names outside Settings (`settings.py`; compose;
Justfile). Exactly one is mandatory (`GRAFY_COMMAND_HMAC_KEY`, fail-closed at
`settings.py:208-219` + `main.py:282`). A 2-line `.env` (+ OIDC trio for login)
boots everything. Action items: (a) flip the `execution_backend` default from
`prefect` to `inline` (or delete it with §3.1) — today a simple local install
silently needs a Prefect server it doesn't have; (b) the 12 auth rate-limit
knobs (`settings.py:61-68`) and 3 presence knobs (`:99-110`) are never exposed
in `.env.example` — keep defaults, document as "not usually touched";
(c) publish the 3-line minimal-install `.env` in the README.

### Tier 3 (product judgment — the big bet)

#### 3.14 Collapse the identity layer to single-owner — *if* that stays the product

**State of play.** Deployment is effectively single-owner: one API process
(`single_owner.py:26-33`), one owner on a VPS
(`infra/docker/README.md:1-16`), release posture "private trusted pilot; public
multi-user not yet recommended" (`RELEASE_AUDIT_REPORT.md:8-9`). Yet the
identity layer is fully multi-user by design: 9 tables, ~7,070 backend lines +
~3,700 tests, ~20,400 lines counting frontend.

**What would actually collapse** (keeping OIDC login + sessions):
`workspace_memberships` (roles/capabilities → "the session user is the owner"),
`oidc_bootstrap_owner_mappings` (§3.7), member endpoints + `WorkspaceMembersDialog`,
team-workspace creation, the PAT layer (§3.2), and the team-library role gates.
`workspaces` **stays** as the tenancy key (every content table FKs to it —
collapsing that is the big-bang, not worth it).

**What is lost:** a second login, member roles, team sharing of graphs/modules/
templates, and — decisively — **multi-participant co-editing rooms** (presence,
concurrent commands, shared execution admission are all membership-authorized).
If team collaboration is a stated near-term goal, **do not do this** — the layer
is the thing the product is moving toward, and the pilot is running on it. If
the product is a single-owner workbench for the foreseeable future, this is the
single largest simplification available in the codebase (≈8–10k lines, ~4
tables, an entire role/capability model).

**Recommendation:** make the call explicitly and write it down as an ADR;
carrying "fully multi-user identity for a single-owner pilot" without that
decision is the most expensive ambiguity in the backend.

---

## 4. The table-count roadmap

> Implemented 2026-08-23: the idempotency deletion, journal deletion, node-table
> merge, and slots → partial unique index steps below are done (31 → 27
> application tables). The PAT and `user_graph_states` deletion rows were not
> executed — both tables are retained by product decision, so the count stops
> at 27 rather than continuing to 25.

| Step | Action | Tables 31 → |
|---|---|---|
| ~~now~~ done | delete `graph_execution_idempotency` (§3.3) | 30 |
| ~~now~~ **rejected** | delete `personal_access_tokens` (§3.2) — retained for external automation/MCP | — |
| ~~now~~ done | delete `graph_command_journal` (§3.4, no replay planned) | 29* |
| ~~now~~ **rejected** | delete `user_graph_states` (§3.5) — retained planned product state | — |
| ~~next~~ done | merge `graph_execution_requested_nodes` into one `graph_execution_nodes` table (§3.3) | 28* |
| ~~next~~ done | merge `graph_active_execution_slots` → partial unique index (§3.3) | 27 |
| ~~next~~ **rejected** | `oidc_bootstrap_owner_mappings` → column/first-login rule (§3.7) — ADR 0003 keeps the exact `(issuer, subject)` mapping | — |
| product call | §3.14 membership collapse | −1 to −3 |

\* numbering reflects the actual implemented order; the endpoint is 27 tables.

The two biggest *storage* levers are not table counts: bound
`saved_graph_revisions` (full document per checkpoint, dead browsing surface)
and give the four unbounded families a retention policy (§3.9).

---

## 5. Suggested order of attack

1. **One afternoon, zero risk:** retirement-debt cleanup (§3.12) — husks,
   stale venv install, MCP docs, README corrections.
2. **This week, low risk:** delete PAT layer (§3.2) + delete
   `graph_execution_idempotency` (§3.3) + wire audit retention (§3.6) +
   config minimal-install (§3.13).
3. **This sprint:** delete Prefect (§3.1, with the inline MAP-concurrency fix);
   merge execution node tables (§3.3).
4. **Next sprint:** journal decision (§3.4) + head/saved_graphs document
   dedup (§3.8) together (they interact); slots → partial index (REFACTOR_PLAN
   #9, now that finding 7's transitions are in); star/folder ship-or-cut
   decision (§3.5).
5. **Product decision, then:** identity collapse (§3.14) — as an ADR, not a
   refactor-by-default; retention policies for the four unbounded families
   (§3.9).
6. **Cosmetic, any time:** move `executions/runtime` (§3.10), collapse small
   route modules (§3.11).

---

## Appendix — verification notes

- All file:line citations checked against the working tree at audit time.
  Headline claims re-verified independently after the subsystem reports:
  `composition.py:181-183` (inline MAP concurrency), PAT zero-call-sites,
  `browser_actor` Authorization rejection, `settings.py:208-219`
  (fail-closed HMAC), `schema.py` table inventory, migration 0009 creating all
  six collab tables, live DB row counts (1 user / 2 workspaces / 2 memberships /
  0 content rows — this audit is schema-level, not data-shape),
  `user_graph_states` UI consumption (hook-only), `list_revisions`
  (no route callers), no production DELETE of `graph_executions`.
- The dev DB is a fresh single-login environment; "is the feature used" is
  answered by UI surface + call-site greps, not usage counts.
- Prior audits remain valid where they overlap: `REFACTOR_PLAN.md` findings 1–7
  are DONE, 8–9 TODO (finding 9 is executed here as §3.3); `SMELL_REVIEW.md`
  findings 1–8 are unchanged by this audit.