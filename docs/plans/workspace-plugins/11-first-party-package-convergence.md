# Slice 11: First-party package convergence

- **Status:** In progress
- **Updated:** 2026-08-24
- **Depends on:**
  [Slice 10](10-artifact-contract-and-runtime-parity.md)
- **Outcome:** First-party Plugin families use one visible project convention;
  canonical artifact contracts stay producer-neutral and runtime registrations
  are owned by the Plugin that declares them

## Target project shape

```text
plugins/arithmetic/
  pyproject.toml
  uv.lock
  src/grafy_plugin_arithmetic/
    declaration.py
    artifacts.py       # if owned
    models.py          # if needed
    nodes.py
    persistence.py     # if needed
    plugin.py
    __init__.py
  tests/
```

System projects use a family-specific import package because they may be loaded
as an exact deployment fast path. Workspace projects keep the fixed isolated
`src/grafy_plugin/` package. Publication normalizes both into the same release
contract; package import names are not catalog identity.

Canonical cross-Plugin artifact contracts such as `scalar.integer@1`,
`scalar.text@1`, and `table.data@1` live in a shared core contract namespace.
A Plugin owns implementations, writers, and resolvers, but consumers depend on
the stable contract rather than importing the producer Plugin.
Cross-Plugin artifact conversions are also producer-neutral, but their
implementation is deployment-owned rather than assigned to either endpoint's
release.

Preserve existing operator IDs, artifact keys, and serialized configuration.
Plugin slugs that currently contain `builtin.*` or `external.*` remain identity
until an explicit pre-baseline migration chooses stable family slugs; after a
System baseline is published, origin-looking text is historical identity and
must not be silently renamed.

## Implementation checklist

- [ ] Consume the producer-neutral contracts completed in Slice 10; do not add
      core-to-Plugin implementation re-export dependencies.
- [ ] Move arithmetic, text, sequence, image, schema, prompt, and table Plugin
      declarations/implementations into `plugins/<family>/` projects.
- [ ] Keep graph-module boundary code outside the Plugin release taxonomy.
- [x] Make integer/text writers and resolvers ordinary owning-Plugin
      registrations; remove composition-root special cases.
- [ ] Make GIS, SQL, OCR, and LLM System Plugin projects use the same declaration
      and release metadata convention.
- [ ] Give every System project its own lock, owning tests, and exact SDK-wheel
      supply rule; remove Workspace path escapes from publishable freezes.
- [ ] Move families in dependency order: arithmetic; text and sequence; image;
      schema and prompt; table; then GIS, SQL, OCR, and LLM.
- [ ] Keep SQL `isolated-only`; user-authored query execution is never eligible
      for the host fast path.
- [ ] Preserve operator IDs, operator versions, artifact keys, serialized config,
      and public graph behavior during file movement.
- [ ] Update workspace packaging, type-check paths, tests, and development docs.
- [x] Separate deployment-owned mutable Workspace authoring roots from the
      repository `plugins/<family>/` System project root.

## Verification checklist

- [ ] Each System Plugin project constructs and freezes independently.
- [ ] Cross-family consumers import shared artifact contracts, not another
      Plugin implementation package.
- [ ] Catalog snapshots prove operator and artifact identities are unchanged.
- [ ] Runtime composition obtains all writers/resolvers from registrations.
- [ ] Focused family suites and strict type checking pass after each move.

## Exit criteria

- [ ] Adding a first-party family and adding a Workspace family use the same
      conceptual files and Plugin declaration surface.
- [ ] `libs/core` owns contracts/runtime primitives, not first-party business
      operator implementations.
- [ ] No API composition code knows arithmetic/text persistence details.

## Agent handoff

- **Owner:** Codex
- **Branch or PR:** `feat/workspace-plugin-releases` / PR #8
- **Implementation evidence:** Arithmetic/Text own their scalar persistence
  registrations; API composition no longer special-cases those adapters.
  Mutable agent working copies moved out of repository `plugins/` and are
  namespaced by Workspace UUID.
- **Verification evidence:** Composition/scalar suites and authoring/settings
  suites pass; package movement and independent freeze verification remain.
- **Open decisions or blockers:** None.
