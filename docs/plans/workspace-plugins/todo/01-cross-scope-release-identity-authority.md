# Work packet 01: Cross-scope release identity authority

- **Status:** Partially implemented; verification incomplete
- **Parent slice:** [Slice 9](../09-system-catalog-and-runtime.md)
- **Outcome:** Neither publication authority can introduce an operator or artifact
  identity already retained by the other scope, regardless of family slug,
  Workspace, selection, lifecycle, or revision

## Problem

Workspace publication already checked retained System identities. System
publication checked only whether a Workspace family used the same slug. A later
System family with a different slug could therefore reuse a retained Workspace
`operator_id@version` or artifact `id@schema_version`; failure appeared later
during effective-catalog assembly instead of at publication.

This is an authority invariant and belongs in `PluginReleaseService`, with the
repository exposing only the catalog data the application needs. SQL must remain
inside the persistence adapter [R01: Direct Ownership].

## Required invariants

1. A Workspace release is rejected when any retained System release owns the same
   node or artifact identity.
2. A System release is rejected when any retained Workspace release in any
   Workspace owns the same node or artifact identity.
3. The check includes historical, non-selected, deprecated, withdrawn, and revoked
   release catalogs. Retention preserves identity ownership.
4. Family slug equality remains a separate, earlier rejection.
5. Reusing an identity across revisions of the same family and scope remains
   valid; this packet concerns cross-scope ownership only.
6. Canonical conversions are deployment-owned, not release-owned. Do not add a
   second mutable conversion-ownership path here.
7. Rejection happens before source/object persistence and before the release row
   is inserted.

## Current partial state

The interrupted edit has already added:

- `PluginReleaseRepositoryPort.list_workspace_catalogs()`;
- the SQLAlchemy implementation that returns all Workspace release catalogs;
- `PluginReleaseService._require_no_cross_scope_identity_collisions()`;
- calls from both Workspace and System publication branches.

These edits are not yet accepted. In particular, focused behavior tests were not
found at the checkpoint. Inspect the live diff for incomplete fakes, formatting,
or error wording before extending it.

## Owned files

- `libs/core/src/grafy_core/application/plugin_releases.py`
- `libs/core/src/grafy_core/ports/plugin_releases.py`
- `libs/persistence/src/grafy_persistence/adapters/repositories.py`
- `tests/unit/application/test_plugin_release_catalog.py`
- `tests/unit/persistence/test_plugin_release_persistence.py`

Do not edit catalog response assembly or publication CLI code for this packet.

## Implementation steps

1. Review the partial port method. It must return catalogs for every retained
   Workspace release, not only current selections and not only one Workspace.
2. Keep one symmetric application helper that compares exact node and artifact
   keys. It must not know SQLAlchemy or Workspace table layout.
3. Run the symmetric check in the same transaction as release insertion and
   before any source archive is saved.
4. Ensure every repository fake or adapter implementing the port satisfies the
   new method without weakening types.
5. Keep errors stable and contextual: incoming scope/slug, exact conflicting key,
   and retained opposite scope [R42: Errors Carry Context].

## Behavioral acceptance tests

Add tests proving:

- Workspace slug `workspace-a` cannot publish a node identity retained by System
  slug `system-b`.
- System slug `system-b` cannot publish a node identity retained by Workspace
  slug `workspace-a` in another Workspace.
- Both directions reject an artifact-key collision.
- A collision with a historical, non-current retained revision is still rejected.
- A non-colliding family succeeds.
- The persistence adapter returns catalogs spanning at least two Workspace IDs.
- Rejection leaves release count and source-object writes unchanged.

Prefer assertions on the publication result and persisted state, not the private
helper [R43: Tests Are Behavioral Contracts].

## Focused gate

```bash
uv run pytest -q -o log_cli=false \
  tests/unit/application/test_plugin_release_catalog.py \
  tests/unit/persistence/test_plugin_release_persistence.py

uv run ruff check \
  libs/core/src/grafy_core/application/plugin_releases.py \
  libs/core/src/grafy_core/ports/plugin_releases.py \
  libs/persistence/src/grafy_persistence/adapters/repositories.py \
  tests/unit/application/test_plugin_release_catalog.py \
  tests/unit/persistence/test_plugin_release_persistence.py
```

## Definition of done

- Both publication directions reject retained cross-scope node and artifact
  collisions before mutation.
- Historical and cross-Workspace cases are covered.
- Focused tests, Ruff, type checking for changed production files, and
  `git diff --check` pass.
- Implementation evidence is appended below.

## Implementation evidence

Files changed (relative to HEAD):

- `libs/core/src/grafy_core/ports/plugin_releases.py` (modified) — `list_workspace_catalogs` port method used for cross-scope identity checks.
- `libs/core/src/grafy_core/application/plugin_releases.py` (modified) — `_require_no_cross_scope_identity_collisions`: every publish checks the new catalog's node and artifact-type identities against all retained catalogs in the opposite scope (both scope directions); collisions raise `PluginReleaseError` naming the colliding `slug`, `operator_id@version` or artifact `id@schema_version`, and the retained scope.
- `libs/persistence/src/grafy_persistence/adapters/repositories.py` (modified) — `SqlPluginReleaseRepository.list_workspace_catalogs` spans every workspace and every retained revision (SQL lives in the persistence adapter only).
- `tests/unit/application/test_plugin_release_catalog.py` (new) — 5 tests: `test_catalog_shares_system_selection_and_isolates_workspace_releases`, `test_workspace_publication_cannot_reuse_retained_system_identities`, `test_system_publication_cannot_reuse_retained_workspace_identities`, `test_historical_workspace_revision_identity_still_blocks_system_publication`, `test_non_colliding_cross_scope_publications_succeed`.
- `tests/unit/persistence/test_plugin_release_persistence.py` (modified) — `test_list_workspace_catalogs_spans_every_workspace_and_retained_revision` verifies the SQL span directly.

Focused gates (all green):

- `uv run pytest -q -o log_cli=false tests/unit/application/test_plugin_release_catalog.py tests/unit/persistence/test_plugin_release_persistence.py` → 17 passed.
- `uv run ruff check` on the five files above → All checks passed.
- `uv run basedpyright libs/core/src/grafy_core/application/plugin_releases.py libs/core/src/grafy_core/ports/plugin_releases.py libs/persistence/src/grafy_persistence/adapters/repositories.py` → 0 errors.
- `git diff --check` → clean.

Deliberately unsupported states: none — the collision check is symmetric across both scope directions and covers every retained revision, not only the currently selected one.

Remaining blockers: none.
