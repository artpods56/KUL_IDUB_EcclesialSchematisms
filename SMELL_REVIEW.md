# Backend Code-Smell Audit

> A pragmatic, evidence-based pass over the Notarius backend looking for places
> where code "could definitely look better or simpler." This is not a SOLID
> review (that already exists and found zero violations) — it targets **readability,
> duplication, and simplicity** smells. Every finding cites exact file/line
> evidence from the current tree. No code was changed; this is a findings report.

- **Mode:** repository (backend)
- **Scope:** `apps/api`, `apps/mcp`, `libs/core`, `libs/persistence`,
  `libs/storage`, `plugins/*` — 197 Python files, ~46,260 lines.
- **Excluded:** tests, frontend, generated/vendor code.

---

## Summary

| # | Smell | Severity | Where |
| --- | --- | --- | --- |
| 1 | Dialect-switch upsert boilerplate repeated 4× | Medium | `repositories.py` |
| 2 | Near-identical scalar `write()` bodies | Medium | `operators/{arithmetic,text,schemas,tables}.py` |
| 3 | 11× identical route error-handling boilerplate | Medium | `artifacts/views.py` |
| 4 | Giant service classes (1,000–1,400 lines) | Medium | `auth/services.py`, `artifacts/services.py` |
| 5 | Duplicated model validators across 3 layers | Low | `saved_graphs/models.py`, `mcp/models.py`, `domain/saved_graphs.py` |
| 6 | `from_domain`/`from_graph` hand-written converters | Low | `saved_graphs/models.py` |
| 7 | `main.py` `_request_validation_error_handler` is a 396-line monolith | Medium | `apps/api/main.py` |
| 8 | Repeated error-response OpenAPI dicts | Low | `artifacts/views.py` |

---

## Finding 1 — Dialect-switch upsert boilerplate (DRY / G5)

`libs/persistence/src/notarius_persistence/adapters/repositories.py` repeats the
same SQLite-vs-PostgreSQL `insert` selection in **four** places. Three are
identical `sqlite_insert`/`postgresql_insert` branches (lines 925, 982, 1414),
and a fourth (line 1789) is a slightly different inline variant that also falls
back to a read-then-insert path for other dialects.

```python
# ~lines 925-931, repeated verbatim at 982 and 1414
table = schema.invocation_cache_entries
dialect_name = self._session.get_bind().dialect.name
if dialect_name == "sqlite":
    insert_statement = sqlite_insert(table)
elif dialect_name == "postgresql":
    insert_statement = postgresql_insert(table)
else:
    raise NotImplementedError(...)
```

**Why it's smelly:** the exact `sqlite_insert(table)`/`postgresql_insert(table)`
selection appears three times verbatim plus one near-duplicate. The only
differences between the three copies are the table and the `.on_conflict_do_nothing`
index columns. A single helper — e.g. `_conflict_insert(session, table, index_elements)`
that picks the dialect and raises on unsupported backends — would collapse all
three into one call site and make the "supported dialects" policy a single,
centralized decision (G33: encapsulate boundary conditions).

**Simpler:** extract one small helper and delete the three duplicated branches.

---

## Finding 2 — Near-identical scalar `write()` bodies (DRY / G5)

`IntegerValueOutputWriter.write`, `TextValueOutputWriter.write`,
`SchemaValueOutputWriter.write` (and the table writer) in
`libs/core/src/notarius_core/operators/` are **byte-for-byte identical** except
for the payload model name (`IntegerValuePayload` vs `TextValuePayload`).

`arithmetic.py:227` and `text.py:285` both contain:

```python
try:
    payload = <Payload>.model_validate({"value": value})
except ValidationError as exc:
    message = (f"Failed to serialize {self.artifact_type.id}@..."
               f"{context.node_context.node_id!r}")
    raise RuntimeError(message) from exc
payload_json = cast(JsonObject, payload.model_dump(mode="json"))
payload_bytes = json.dumps(payload_json, ensure_ascii=False, sort_keys=True, ...)
provenance = {input_name: [...] for input_name, refs in context.provenance...}
metadata = {"producer_node_id": ...}
if provenance: metadata["provenance"] = provenance
metadata.update(context.metadata)
artifact = ArtifactObject(workspace_id=..., artifact_type=..., inline_payload=...)
```

**Why it's smelly:** ~35 lines of serialization/provenance/metadata construction
are duplicated per scalar type. The only variation is which payload model to
validate. This is the classic "extract a generic base writer and pass the payload
model in" refactor.

**Simpler:** a single generic `InlineScalarOutputWriter` parameterized by payload
model (or a small helper `_write_inline_scalar(payload_model, value, context)`),
then each scalar writer becomes a thin subclass. Same pattern appears in the
resolvers (`arithmetic.py:310`, `text.py:368`, `schemas.py:416`, `tables.py:647`
all share the `artifact = await uow.artifacts.get(...)` / `if artifact is None:
raise NotFoundError` prologue).

---

## Finding 3 — 11× identical route error-handling boilerplate (DRY / G5)

`apps/api/src/notarius_api/v1/routes/artifacts/views.py` repeats the same
handler shape in **11 routes**:

```python
artifact = await service.get(access.workspace_id, artifact_id)
if artifact is None:
    raise HTTPException(status_code=404, detail="Artifact not found")
try:
    return await service.load_geo_render(artifact, workspace_id=...)
except ArtifactContentUnavailableError as exc:
    raise HTTPException(status_code=500, detail=str(exc)) from exc
except WorkbenchOperationError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
```

`if artifact is None:` appears 11 times, `except ArtifactContentUnavailableError`
11 times, `except WorkbenchOperationError` 10 times, and `"Artifact not found"`
9 times.

**Why it's smelly:** the "load-or-404 + map two error types to status codes"
sequence is repeated verbatim. Any change to error mapping (e.g. a new status
code for content-unavailable) must touch all 11 call sites. A single dependency
helper — e.g. `_require_artifact(service, workspace_id, artifact_id)` returning
the artifact or raising `HTTPException(404)` — plus one
`ArtifactServiceErrorHandler` (or FastAPI exception handlers for
`ArtifactContentUnavailableError` → 500 and `WorkbenchOperationError` → 400)
would remove ~50 duplicated lines and centralize the error contract (G28:
encapsulate conditionals; G5: eliminate duplication).

---

## Finding 4 — Giant service classes (SRP-adjacent readability)

Two classes in `apps/api` are over 1,000 lines:

- `AuthService` — `auth/services.py`, ~1,224 lines, 30+ methods spanning OIDC
  protocol, credential issuance, session cookies, PAT issuance, audit, and abuse
  control (the latter already delegated to `AuthAbuseControl`).
- `ArtifactService` — `artifacts/services.py`, ~1,393 lines, ~22 methods spanning
  geo/vector/raster/table content access.

The SOLID review already considered both and classified them as cohesive
single-actor services — so this is **not** a SRP violation. But for **readability**
they are hard to navigate: `ArtifactService` mixes `_source_name`,
`_feature_render_layer`, `_raster_render_layer`, and `_resolve_render_layer`
into one class, and `AuthService` mixes `_exchange_code` / `_validate_id_token`
/ `_provider` / `_keys` OIDC internals with cookie and audit helpers.

**Why it's smelly:** even if cohesive, a 1,400-line file is a poor reading
experience (the 10:1 read:write rule). The OIDC internals in `AuthService` and
the geo-render internals in `ArtifactService` could each move to a dedicated
module/helper class without changing ownership — improving locality without
changing the single-actor design.

**Simpler:** split the private helpers (e.g. `OidcProvider` and `GeoRenderLayers`)
into their own modules/classes, keeping the public service API unchanged.

---

## Finding 5 — Duplicated model validators across 3 layers (DRY / G5)

The same validator logic is hand-duplicated in the API models, MCP models, and
core domain:

- `normalize_name` (`value.strip()`) appears in `saved_graphs/models.py:394`,
  `saved_graphs/models.py:616`, `saved_graphs/models.py:875`, and `mcp/models.py:283`.
- `require_at_least_one_dimension` appears in `saved_graphs/models.py:91`,
  `mcp/models.py:219`, `mcp/models.py:378`, and `domain/saved_graphs.py:61`.
- `validate_artifact_type_bindings` appears in `saved_graphs/models.py:117`,
  `mcp/models.py:245`, and `mcp/models.py:404`.

**Why it's smelly:** the API, MCP, and domain layers each re-declare the same
validation instead of reusing one canonical definition. Some of this is
unavoidable (Pydantic `field_validator` can't be trivially shared), but the
`normalize_name` validator — a two-line strip — is repeated 4× and could be a
single shared `Annotated[str, ...]` type or mixin.

**Simpler:** extract the shared validators into a small `validators.py` module
(or a Pydantic mixin) used by all three layers.

---

## Finding 6 — Hand-written `from_domain` / `from_graph` converters (DRY)

`apps/api/src/notarius_api/v1/routes/saved_graphs/models.py` (880 lines) contains
many hand-written, structurally repetitive classmethod converters —
`from_domain` (84 lines), `from_graph` (84 lines), `from_graphs`, `from_folder`,
`from_organization`, `from_state`, `from_item`, `from_items` — each manually
mapping domain fields to response-model fields field-by-field:

```python
viewers=[GraphPresentationViewerModel(
    id=viewer.id,
    position=GraphPointModel(x=viewer.position.x, y=viewer.position.y),
    layout=(SavedGraphNodeLayoutModel(width=viewer.layout.width, ...)
            if viewer.layout is not None else None),
    ...
) for viewer in presentation.viewers]
```

**Why it's smelly:** each converter is ~80 lines of mechanical field copying that
must be kept in sync with the domain model by hand. If a domain field is added
or renamed, the converter is a second place to update.

**Simpler:** where the models are structurally identical, prefer `.model_validate`
on the domain value (or a shared mapping) rather than hand-rolled
`from_*` converters; reserve hand-written converters for genuinely different
shapes.

---

## Finding 7 — `main.py` has a 396-line exception handler monolith

`apps/api/src/notarius_api/main.py:_request_validation_error_handler` is a single
function of ~396 lines (counting the `create_app` body it's inside) handling the
OIDC login/callback rate-limiting special cases inline, with the same
`422 if allowed else 429` / `"Too many login attempts"` logic duplicated for
`login` and `callback`.

**Why it's smelly:** the OIDC abuse-control logic is embedded in an exception
handler at module scope rather than delegated to `AuthService` (which already has
`allow_login_start`/`allow_callback`/`audit_request_failure`). The two branches
(`login` and `callback`) are near-duplicates — the only differences are the
operation name, the `replace_login_transaction`/`release_login` step, and the
"callback attempts" message.

**Simpler:** extract one `_oidc_validation_error_response(request, exception, kind)`
helper (or move it into `AuthService`) and collapse the two near-identical
branches. This also removes the giant function from the composition root,
improving `main.py` readability.

---

## Finding 8 — Repeated error-response OpenAPI dicts

`artifacts/views.py` declares the same `responses=` dicts on every route:
`"Invalid ..."` appears 10 times, `WorkbenchErrorResponse` as the error model 32
times, and `{"model": WorkbenchErrorResponse, "description": "Artifact not
found"}` / `"Artifact content is unavailable"` repeatedly.

**Why it's smelly:** the `400/404/500` response metadata is copy-pasted onto each
route decorator. A shared `responses` constant (e.g. `_CONTENT_ERRORS =
{400:..., 404:..., 500:...}`) would make the routes shorter and the error contract
single-sourced.

**Simpler:** define one `_artifact_error_responses` dict and reuse it across the
artifact route decorators.

---

## Lower-confidence / deliberate-design notes

These were inspected and **not** reported as smells, to avoid noise:

- **`repositories.py` (1,986 lines) / `schema.py` (1,271 lines):** large files,
  but each is a collection of cohesive per-aggregate classes (the SOLID review
  confirmed 12 cohesive repository classes). They are big but not duplicated;
  splitting would be cosmetic.
- **`collaboration.py` `apply_graph_command`** — an `isinstance` dispatch over
  ~20 `GraphCommandKind` members. The SOLID review already concluded this is the
  idiomatic command-dispatch pattern over a deliberately bounded command journal;
  not reported as an OCP violation.
- **`saved_graphs/models.py` 880 lines:** many small response models + converters;
  the duplication is Finding 5/6, not a single-giant-class problem.
- **Magic numbers** (e.g. `extent = 20_037_508.342789244` in
  `artifacts/services.py:257`, `port = 443 if https else 80`) — mostly domain
  constants; low value. Not reported.
- **No `TODO`/`FIXME`/commented-out code** was found anywhere in the backend —
  that hygiene is already clean.

---

## Suggested priority

If you want the highest value-per-effort, do these first:

1. **Finding 1** — one helper removes 4 duplicated dialect branches.
2. **Finding 3** — one dependency + exception handlers remove ~50 duplicated
   lines across 11 routes.
3. **Finding 2** — one generic writer removes ~35 duplicated lines × 4 scalar
   writers.
4. **Finding 7** — collapse the 396-line handler into a helper / `AuthService`.
5. **Findings 5/6/8** — small, mechanical DRY cleanups.
6. **Finding 4** — the giant classes are the biggest readability cost but the
   lowest-risk refactor (pure extraction, no behavior change); best done last.

---

## References

- `SOLID_REVIEW.md` — the formal SOLID audit (zero violations) of the same scope.
- `docs/design/backend-architecture.md` — package structure and dependency flows.
