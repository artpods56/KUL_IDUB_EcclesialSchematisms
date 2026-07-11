# Notarius Studio Full Refactor: Codex CLI Plan

Created: 2026-05-13

This is a full implementation brief for an overnight Codex CLI run. It is meant
for a strong coding agent with a long-running `/goal` session.

## Goal

Refactor the current Notarius repository into a multi-package monorepo that can
support Notarius Studio:

- `libs/core`: generic structured/contextual sequential extraction engine
- `libs/schematisms`: current schematism-specific models, prompts, recipes, and utilities
- `libs/llm`: concrete LLM adapters and configuration
- `libs/storage`: concrete storage adapters
- `libs/shared`: small shared utilities
- `apps/dagster`: Dagster research/batch app consuming the libraries
- `apps/api`: FastAPI product backend scaffold
- `apps/worker`: extraction job worker scaffold

The product direction:

> Notarius Studio is a platform for structured and contextual extraction from
> sequential long-form sources.

## Strong Preference: Use A Separate Worktree

The main repo may contain many dirty user changes. Do not risk overwriting them.

From the current repo, prefer:

```bash
git status --short
cd ..
git -C KUL_IDUB_EcclesialSchematisms worktree add KUL_IDUB_EcclesialSchematisms-refactor -b codex/notarius-studio-full-refactor HEAD
cd KUL_IDUB_EcclesialSchematisms-refactor
```

If a worktree is not possible, create/switch a branch in-place only after
inspecting dirty files:

```bash
git status --short
git switch -c codex/notarius-studio-full-refactor
```

Never run `git reset --hard`, `git clean`, or checkout/revert user changes.

## Target Layout

```text
.
├── apps
│   ├── api
│   │   └── src/notarius_api
│   ├── worker
│   │   └── src/notarius_worker
│   └── dagster
│       └── src/notarius_dagster
├── libs
│   ├── core
│   │   └── src/notarius_core
│   ├── schematisms
│   │   └── src/notarius_schematisms
│   ├── llm
│   │   └── src/notarius_llm
│   ├── storage
│   │   └── src/notarius_storage
│   ├── shared
│   │   └── src/notarius_shared
│   └── persistence
│       └── src/notarius_persistence
├── docs
├── infra
├── scripts
└── tests
```

## Dependency Rules

```text
apps/api      ┐
apps/worker   ├── may depend on libs/*
apps/dagster  ┘

libs/schematisms -> libs/core
libs/llm         -> libs/core, libs/shared
libs/storage     -> libs/core, libs/shared
libs/persistence -> app-level persistence; may depend on libs/shared
libs/core        -> only generic dependencies
```

`libs/core` must not import:

- `notarius_schematisms`
- schematism-specific models like `SchematismPage`, `PageContext`, deanery/parish-specific helpers
- Dagster
- FastAPI
- SQLAlchemy
- HuggingFace/datasets
- concrete OpenAI/OpenRouter adapters

Boundary check:

```bash
rg -n "Schematism|PageContext|deanery|parish|notarius_schematisms|dagster|fastapi|sqlalchemy|datasets|huggingface|openai|openrouter" libs/core/src/notarius_core
```

Prefer no hits. If there are hits, they must be harmless generic comments, but
rewrite comments to generic language where possible.

## Phase 0: Recon And Baseline

Run:

```bash
git status --short
rg -n "DatasetProcessor|ItemProcessor|SequenceState|ContextStrategy|PageContext|SchematismPage" src tests scripts
rg --files src/notarius/application src/notarius/domain src/notarius/infrastructure src/notarius/orchestration src/notarius/schemas | sort
pytest tests/unit/domain/entities/test_messages.py tests/unit/domain/test_strip_next_page_ocr.py
```

If baseline tests fail for unrelated reasons, record that and continue with
focused tests after each phase.

## Phase 1: Extract `notarius_core`

Create:

```text
libs/core/pyproject.toml
libs/core/src/notarius_core
```

Core should contain generic engine primitives:

```text
notarius_core/
├── application
│   ├── context
│   │   ├── provider.py
│   │   └── strategy.py
│   ├── processors
│   │   ├── dataset_processor.py
│   │   └── item_processor.py
│   └── sequence_state.py
├── domain
│   └── models
│       ├── completions.py
│       ├── dataset.py
│       └── messages.py
├── ports
│   ├── llm.py
│   ├── protocols.py
│   ├── prompts.py
│   └── storage.py
└── prompts
    └── message_builder.py
```

Move/genericize:

- `ChatMessage`, content models, message stripping helpers
- provider-neutral completion response models
- generic `BaseDataItem`, `PredictionDataItem`, `BaseDataset`
- `SequenceState`
- `DatasetProcessor`
- `ItemProcessor`
- generic request/response handlers
- generic context strategies
- generic context providers
- message builder abstractions
- `CompletionRequest`, `CompletionResult`, `LLMCompletionEngine` protocol
- storage protocol needed by `DatasetProcessor`

Required design change:

```python
domain_context: BaseModel | dict[str, Any] | None = None
```

Core may use `pydantic`, `Pillow`, and `jinja2` if that keeps the move simple.
Core must not use concrete provider/client libraries.

Add compatibility re-exports in old `src/notarius` paths so existing imports
continue to work.

Add tests:

```text
tests/unit/core/test_sequence_state.py
tests/unit/core/test_item_processor.py
tests/unit/core/test_context_strategy.py
tests/unit/core/test_messages.py
```

Verify:

```bash
python -c "import notarius_core; print('core ok')"
pytest tests/unit/core
```

## Phase 2: Extract `notarius_schematisms`

Create:

```text
libs/schematisms/pyproject.toml
libs/schematisms/src/notarius_schematisms
```

Suggested structure:

```text
notarius_schematisms/
├── domain/models.py
├── context/providers.py
├── responses/handlers.py
├── recipes.py
├── data
│   ├── aligning.py
│   ├── flattening.py
│   └── merging.py
├── evaluation
│   └── scoring.py
└── prompts/tasks/...
```

Move schematism-specific models:

```text
src/notarius/domain/entities/schematism.py
-> libs/schematisms/src/notarius_schematisms/domain/models.py
```

Own these there:

- `SchematismEntry`
- `SchematismEntryWithValue`
- `SchematismPage`
- `SchematismWithValuePage`
- `PageContext`
- `CleryEntry`
- `ElenchusPage`
- `ElenchusPageContext`

Move schematism prompt task directories:

```text
src/notarius/infrastructure/llm/prompts/tasks/structured_extraction
src/notarius/infrastructure/llm/prompts/tasks/source_generation
src/notarius/infrastructure/llm/prompts/tasks/elenchus_extraction
src/notarius/infrastructure/llm/prompts/tasks/tr_1529_structured_extraction
src/notarius/infrastructure/llm/prompts/tasks/transliterate_structured_extraction
src/notarius/infrastructure/llm/prompts/tasks/xlm
```

to:

```text
libs/schematisms/src/notarius_schematisms/prompts/tasks
```

Move or split schematism utilities:

```text
src/notarius/application/services/data/flattening.py
src/notarius/application/services/data/aligning.py
src/notarius/application/services/data/merging.py
src/notarius/application/services/scoring/*
src/notarius/domain/services/parser.py
```

Decision rule:

- If it references deanery/parish/dedication/material/clergy/schematism pages,
  it belongs in `notarius_schematisms`.
- If it is generic over arbitrary Pydantic outputs, keep/move it in `notarius_core`.

Create recipe helpers such as:

```python
SCHEMATISM_REFINEMENT_CONTEXT_PROVIDERS = ...
SOURCE_GENERATION_CONTEXT_PROVIDERS = ...
TASK_SCHEMA_REGISTRY = ...
```

Add compatibility re-exports from old schematism paths.

Tests:

```text
tests/unit/schematisms/test_models.py
tests/unit/schematisms/test_flattening.py
tests/unit/schematisms/test_aligner.py
```

Verify:

```bash
python -c "import notarius_schematisms; print('schematisms ok')"
pytest tests/unit/schematisms tests/unit/domain/services/test_aligner.py
```

## Phase 3: Extract `notarius_llm`, `notarius_storage`, `notarius_shared`

### `libs/shared`

Create:

```text
libs/shared/src/notarius_shared
```

Move only small cross-cutting utilities:

- logger helpers
- stable constants

Avoid turning shared into a dumping ground.

### `libs/llm`

Create:

```text
libs/llm/src/notarius_llm
├── adapters/engine.py
├── providers/factory.py
├── providers/openai_compatible.py
├── cache/cached_engine.py
└── config.py
```

Move concrete LLM code from:

```text
src/notarius/infrastructure/llm/engine_adapter.py
src/notarius/infrastructure/llm/providers/*
src/notarius/infrastructure/cache/backends/llm.py
src/notarius/schemas/configs/llm_model_config.py
```

`notarius_llm` imports contracts from `notarius_core.ports.llm`.

Core owns:

- `CompletionRequest`
- `CompletionResult`
- `LLMCompletionEngine`

LLM lib owns:

- concrete `LLMEngine`
- concrete cached LLM engine
- OpenAI-compatible provider
- provider factory
- model/client config
- provider-specific retry behavior

### `libs/storage`

Create:

```text
libs/storage/src/notarius_storage
├── adapters/local.py
└── ports.py
```

Move concrete storage from:

```text
src/notarius/infrastructure/persistence/storage/*
```

Core can define storage protocols. `notarius_storage` implements them.

Verify:

```bash
python - <<'PY'
import notarius_core
import notarius_schematisms
import notarius_llm
import notarius_storage
print("libs import ok")
PY
pytest tests/unit/core tests/unit/schematisms tests/unit/infrastructure/test_llm_cache.py tests/unit/infrastructure/test_llm_cache_backend.py
```

## Phase 4: Move Dagster To `apps/dagster`

Create:

```text
apps/dagster/pyproject.toml
apps/dagster/src/notarius_dagster
```

Move:

```text
src/notarius/orchestration
-> apps/dagster/src/notarius_dagster
```

Suggested mapping:

```text
src/notarius/orchestration/assets        -> notarius_dagster/assets
src/notarius/orchestration/defs          -> notarius_dagster/defs
src/notarius/orchestration/jobs          -> notarius_dagster/jobs
src/notarius/orchestration/resources     -> notarius_dagster/resources
src/notarius/orchestration/dill_io_manager.py -> notarius_dagster/io_managers/dill.py
src/notarius/orchestration/hf_io_manager.py   -> notarius_dagster/io_managers/huggingface.py
src/notarius/orchestration/constants.py  -> notarius_dagster/constants.py
src/notarius/orchestration/registry.py   -> notarius_dagster/registry.py
```

Update Dagster entrypoint:

```toml
[tool.dagster]
module_name = "notarius_dagster.defs.dev"
code_location_name = "notarius_dagster"
```

Update scripts/Make/Just references:

```bash
rg -n "notarius\\.orchestration|dagster|defs\\.dev|defs\\.prod" Makefile justfile scripts src tests pyproject.toml
```

Do not redesign the asset graph. Preserve behavior; change ownership/imports.

Verify:

```bash
python -c "import notarius_dagster; import notarius_dagster.defs.dev; print('dagster imports ok')"
dagster definitions validate -m notarius_dagster.defs.dev
pytest tests/unit/orchestration tests/unit/dagster
```

If `dagster definitions validate` is unavailable or fails due to environment
configuration, report the exact error and run import-level verification.

## Phase 5: Add `apps/api` FastAPI Scaffold

Create:

```text
apps/api/pyproject.toml
apps/api/src/notarius_api
├── __init__.py
├── main.py
├── dependencies.py
├── schemas
├── services
└── v1
    ├── router.py
    └── routes
        ├── projects.py
        ├── sources.py
        ├── schemas.py
        ├── recipes.py
        ├── jobs.py
        └── exports.py
```

Expose product concepts:

- Project
- Source
- SourceItem
- OutputSchema
- Recipe
- Job
- JobItem

Minimal routes:

```text
GET    /health
POST   /v1/projects
GET    /v1/projects
GET    /v1/projects/{project_id}
POST   /v1/projects/{project_id}/sources
GET    /v1/projects/{project_id}/sources
GET    /v1/sources/{source_id}
GET    /v1/sources/{source_id}/items
POST   /v1/projects/{project_id}/schemas
GET    /v1/projects/{project_id}/schemas
GET    /v1/schemas/{schema_id}
POST   /v1/projects/{project_id}/recipes
GET    /v1/projects/{project_id}/recipes
GET    /v1/recipes/{recipe_id}
POST   /v1/jobs
GET    /v1/jobs/{job_id}
GET    /v1/jobs/{job_id}/items
POST   /v1/jobs/{job_id}/cancel
POST   /v1/jobs/{job_id}/retry
GET    /v1/jobs/{job_id}/exports/jsonl
GET    /v1/jobs/{job_id}/exports/csv
```

Start with in-memory repositories if DB setup would slow the refactor. Keep the
repository interfaces clean so SQLAlchemy can replace them.

Do not implement auth, multi-tenancy, polished UI, WebSockets, or visual schema
builder in this run.

Verify:

```bash
python -c "from notarius_api.main import app; print(app.title)"
pytest tests/unit/api
```

## Phase 6: Add `apps/worker` Scaffold

Create:

```text
apps/worker/pyproject.toml
apps/worker/src/notarius_worker
├── __init__.py
├── main.py
├── runner.py
└── pipeline
    ├── __init__.py
    ├── recipe_compiler.py
    └── steps
```

Worker behavior:

1. fetch next queued job
2. mark job running
3. load source items and recipe
4. compile recipe into `notarius_core` runtime components
5. run sequential extraction
6. persist each job item output and context trace
7. mark job succeeded or failed

The recipe compiler bridges Studio configuration to core runtime:

- context provider composition
- context strategy
- message builder / prompt renderer
- request handler
- response handler
- LLM engine
- dataset processor

Use fake/in-memory implementations for initial tests.

Persist/query context trace data per job item:

- rendered input context
- previous domain context
- structured output
- output domain context
- model/provider metadata if available
- error details

Verify:

```bash
python -c "import notarius_worker; print('worker ok')"
pytest tests/unit/worker tests/integration/api/test_job_lifecycle.py
```

## Phase 7: Integration Tests And Cleanup

Add or update tests by ownership:

```text
tests/unit/core
tests/unit/schematisms
tests/unit/llm
tests/unit/storage
tests/unit/api
tests/unit/worker
tests/unit/dagster
tests/integration/api
```

Minimum lifecycle integration test:

1. create project
2. create source with two ordered source items
3. create schema
4. create recipe
5. create job
6. run worker once
7. assert job succeeded
8. assert job items contain structured outputs and context traces
9. export JSONL

Use fake LLM engines. Do not call external providers.

Run broad checks:

```bash
pytest tests/unit/core tests/unit/schematisms tests/unit/api tests/unit/worker
pytest tests/unit/infrastructure/test_llm_cache.py tests/unit/infrastructure/test_llm_cache_backend.py
pytest tests/unit/orchestration || true
```

Run final import checks:

```bash
python - <<'PY'
mods = [
    "notarius_core",
    "notarius_schematisms",
    "notarius_llm",
    "notarius_storage",
    "notarius_api.main",
    "notarius_worker",
    "notarius_dagster.defs.dev",
]
for mod in mods:
    __import__(mod)
    print("ok", mod)
PY
```

## Compatibility Policy

Compatibility re-exports are allowed during this refactor. Mark them clearly:

```python
# Transitional compatibility import. Remove after consumers migrate to notarius_core.
```

Use compatibility shims for old paths under:

```text
src/notarius/application/...
src/notarius/domain/...
src/notarius/infrastructure/...
src/notarius/orchestration/...
src/notarius/schemas/...
```

Once all imports are migrated and tests pass, remove shims only if low-risk.
Otherwise leave them and list them in the final report.

## Stop Conditions

Stop and report if:

- user dirty changes would be overwritten
- package config requires a full unrelated rewrite
- `libs/core` cannot avoid importing schematism-specific code
- tests require real external LLM/API calls
- import cycles become too tangled to resolve cleanly
- moving Dagster breaks the repo in a way that prevents API/worker scaffolding

If a later phase blocks, keep completed earlier phases coherent and report the
blocker. Do not trash working Phase 1/2 changes trying to force Phase 5/6.

## Final Report

At the end, report:

```text
Branch/worktree:
- ...

Completed phases:
- Phase 1: ...
- Phase 2: ...
- Phase 3: ...
- Phase 4: ...
- Phase 5: ...
- Phase 6: ...

Major changed areas:
- libs/core/...
- libs/schematisms/...
- libs/llm/...
- libs/storage/...
- apps/dagster/...
- apps/api/...
- apps/worker/...

Verification:
- command: result
- command: result

Known issues:
- ...

Compatibility shims left:
- ...

Recommended next steps:
- ...
```

Be honest about partial completion. The goal is the full refactor, but a clean
partially completed monorepo with clear blockers is better than a tangled tree.
