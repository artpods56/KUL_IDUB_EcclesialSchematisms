# Backend Architecture Reference

> Technical documentation for the Grafy backend: package structure, dependency
> flows, and structural diagrams. This document complements the product
> vocabulary in `CONTEXT.md` and the interaction plan in
> `docs/workbench-interaction-plan.md`. It describes the *current* committed
> backend as built — not a target architecture. Intended Plugin discovery
> (register / publish / isolated freeze, not `grafy.plugins`) is
> [plugin unification](plugin-unification.md).

- **Audience:** contributors who need to know where code lives, how services are
  composed, and which dependencies may legally exist between packages.
- **Scope:** `apps/api`, `apps/mcp`, `libs/core`, `libs/persistence`,
  `libs/storage`, and `plugins/*`. The Next.js frontend (`apps/web`) is covered
  only at its HTTP boundary.

---

## 1. Overview

Grafy is a node-first workbench for building and running typed artifact
graphs. The backend is a Python monorepo (managed by `uv`) organized as a
**hexagonal / ports-and-adapters architecture** with one clean composition root.

The backend has three entry points into the same domain:

1. **FastAPI HTTP API** (`apps/api`) — the primary REST surface under `/v1`,
   consumed by the Next.js workbench and by agent clients over REST.
2. **FastMCP Streamable HTTP server** (`apps/mcp`) — agent graph-discovery and
   collaboration-aware authoring tools, mounted at `/mcp` on the API process.
3. **Node plugins** (`plugins/*`) — independently packaged extensions that
   register nodes, artifact types, conversions, resolvers, and writers into the
   core registry.

```mermaid
flowchart LR
    subgraph Clients["Clients"]
        Web["Next.js workbench"]
        Agent["Codex / MCP clients"]
    end
    Web --> API["FastAPI workbench API\napps/api"]
    Agent --> MCP["FastMCP Streamable HTTP\napps/mcp (mounted at /mcp)"]
    MCP --> API
    API --> Core["grafy_core\nlibs/core"]
    API --> Persistence["grafy_persistence\nlibs/persistence"]
    API --> Storage["grafy_storage\nlibs/storage"]
    API -. "discovers entry points" .-> Plugins["Installed node plugins\nplugins/*"]
    Plugins --> Core
    Core --> Persistence
    Core --> Storage
    Persistence --> SQL["SQLite / PostgreSQL"]
    Storage --> Objects["Local FS / S3"]
```

---

## 2. Package inventory and ownership

| Package | Path | Responsibility |
| --- | --- | --- |
| `grafy_api` | `apps/api/src/grafy_api` | FastAPI app, `/v1` routes, plugin discovery, runtime composition, HTTP adapters. |
| `grafy_mcp` | `apps/mcp/src/grafy_mcp` | Stateless FastMCP Streamable HTTP tools; request-scoped PAT actor context. |
| `grafy_core` | `libs/core/src/grafy_core` | Domain aggregates, ports, application services, runtime, plugin contract, built-in operators. |
| `grafy_persistence` | `libs/persistence/src/grafy_persistence` | Async SQLAlchemy repositories, unit-of-work adapters, ORM mappings, Alembic schema. |
| `grafy_storage` | `libs/storage/src/grafy_storage` | Local and S3-compatible object stores. |
| `grafy_plugin_llm` | `plugins/llm` | OpenAI-compatible Chat Completions + legacy Mistral structured node. |
| `grafy_plugin_ocr` | `plugins/ocr` | OCR and table-extraction nodes; Mistral + Tesseract adapters. |
| `grafy_plugin_gis` | `plugins/gis` | WGS84 vector sources, georeferenced raster scans, OGC WFS/WMS integration, map-layer recipes. |
| `grafy_plugin_sql` | `plugins/sql` | Parameterized statement artifacts, PostgreSQL batch executor, DuckDB join executor. |

### Dependency direction (hexagonal)

The core is the architectural center. Concrete infrastructure (SQLAlchemy, S3,
Mistral SDKs) lives behind ports and is wired only at composition time.

```mermaid
flowchart TB
    subgraph EntryPoints["Entry points"]
        API["apps/api — HTTP routes + composition root"]
        MCP["apps/mcp — MCP tools"]
        Plugins["plugins/* — node extensions"]
    end
    subgraph Policy["Policy (high-level, depends on ports only)"]
        App["grafy_core/application\nSavedGraphService · CollaborationService\nIdentityService · ModuleLibraryService · TemplateService"]
        Runtime["grafy_core/runtime\nNodeRuntime · InputMaterializer · OutputPersister"]
    end
    subgraph Domain["Domain + ports"]
        DomainAgg["grafy_core/domain\nsaved_graphs · collaboration · identity · modules"]
        Ports["grafy_core/ports\nnarrow per-aggregate Protocol ports"]
    end
    subgraph Adapters["Infrastructure adapters (low-level)"]
        Persistence["grafy_persistence\nSql*Repository · SqlAlchemyUnitOfWork"]
        Storage["grafy_storage\nLocalFileObjectStore · S3ObjectStore"]
        SDKs["plugin SDK adapters\nMistral · OpenAI · Tesseract · GDAL · SQLAlchemy"]
    end

    API --> App
    API --> Runtime
    MCP --> App
    Plugins --> Ports
    Plugins --> DomainAgg
    App --> Ports
    App --> DomainAgg
    Runtime --> Ports
    Ports --> DomainAgg
    Persistence --> Ports
    Storage --> Ports
    SDKs --> Ports

    style Persistence fill:#f4f4f4,stroke:#999
    style Storage fill:#f4f4f4,stroke:#999
    style SDKs fill:#f4f4f4,stroke:#999
```

Rule: **dependency arrows point inward toward the domain.** No port, domain, or
application module imports a concrete infrastructure type (`SqlAlchemy*`,
`LocalFileObjectStore`, SDK clients). All concrete construction happens in
composition roots (`apps/api/src/grafy_api/main.py` and
`services/composition.py`).

---

## 3. Composition root and application assembly

The FastAPI app is built in `apps/api/src/grafy_api/main.py`. During lifespan
it constructs every service from concrete adapters, binds them to
`app.state`, and tears them down once.

```mermaid
flowchart TB
    Settings["Settings\n(env + .env)"] --> Database["create_database\ngrafy_persistence.database"]
    Settings --> StorageFactory["create_file_storage\ngrafy_storage.factory"]
    Settings --> OwnerLease["ApiOwnerLease\nsingle-owner lock"]

    Database --> Db["Database\nengine + async_sessionmaker"]
    Db --> SavedGraphs["SavedGraphService"]
    Db --> ModuleLib["ModuleLibraryService"]
    Db --> Templates["TemplateService"]
    Db --> Collab["CollaborationService"]
    Db --> NodeSecrets["NodeSecretService"]
    Db --> UoW["SqlAlchemyUnitOfWork"]

    builtin_plugins --> Registry["build_plugin_registry"]
    discover_plugins --> Registry
    Registry --> Components["build_workbench_components\nservices/composition.py"]

    StorageFactory --> Storage["FileStoragePort"]
    Components --> AppResources["AppResources\napp.state.resources"]

    Identity["IdentityService"] --> Auth["AuthService"]
    Auth --> AppIdentity["AppIdentity\napp.state.identity"]
```

**Key construction points** (all in `apps/api`):

- **`build_plugin_registry`** (`plugin_discovery.py`) installs builtin plugins,
  discovers external plugins via the `grafy.plugins` entry-point group, and
  `freeze()`s the registry (validating artifact/conversion/port contracts).
- **`build_workbench_components`** (`services/composition.py`) is the workbench
  composition root: it builds resolvers/writers from the registry, the
  `NodeRuntime`, the compiler, the in-process execution engine, and the
  run/execution/artifact/materialization services.
- **`app.state`** holds two dataclasses — `AppIdentity` (auth) and
  `AppResources` (workbench services) — fetched via `get_identity(app)` and
  `get_resources(app)`.

---

## 4. HTTP request lifecycle (REST)

A typical `/v1` request flows through middleware, the route handler, an
application service, and finally a persistence unit-of-work.

```mermaid
sequenceDiagram
    actor C as Client (Web / Agent)
    participant MW as Middleware (CORS, abuse cookie)
    participant R as Route handler (v1/routes/*/views.py)
    participant Svc as Application service (core/application)
    participant Port as Domain port (core/ports)
    participant UoW as SqlAlchemyUnitOfWork
    participant DB as SQLite / PostgreSQL

    C->>MW: HTTP request
    MW->>R: validated request (FastAPI routing)
    R->>Svc: call service method
    Svc->>Port: authorize / capability check
    Port->>UoW: open transaction (context manager)
    UoW->>DB: read/write via repository
    DB-->>UoW: rows
    UoW-->>Port: aggregate returned
    Port-->>Svc: result
    Svc-->>R: response model
    R-->>MW: JSON response
    MW-->>C: HTTP response
```

Exception handling is centralized: `NotFoundError` → 404,
`CapabilityDeniedError` → 403, `UserDisabledError` → 401,
`IdentityInvariantError` → 409, and `RequestValidationError` is special-cased for
the OIDC login/callback flows (abuse control + rate limiting).

---

## 5. MCP dependency flow

`apps/mcp` is deliberately independent: it imports no FastAPI routes,
persistence, storage, or plugin implementations. The API process mounts it at
`/mcp` and injects a **request-scoped** caller binding.

```mermaid
flowchart TB
    Agent["MCP client (PAT bearer)"] --> Mount["create_mounted_mcp_app\napps/api/mcp/mount.py"]
    Mount --> PatMW["_McpPatMiddleware\nrequire_mcp_access"]
    PatMW --> Auth["AuthService\nworkspace-bound PAT validation"]
    Auth --> Caller["McpCallerContext\n(user_id, workspace_id, scopes)"]
    Caller --> Bind["bind_mcp_request\nContextVar binding"]
    Bind --> MCP["grafy_mcp server\nFastMCP tools"]
    MCP --> Ops["ApiGraphWorkspaceOperations\napps/api/mcp/operations.py"]
    Ops --> App["app.state resources\nSavedGraphService · CollaborationService · registry"]
```

- **PAT scoping:** effective permission is `token scope ∩ current membership`.
- **Stateless:** each request binds a fresh caller context; the `Authorization`
  header is never retained in process-global state.
- **Boundary rule:** `grafy_mcp` never imports FastAPI routes, persistence,
  storage, or plugin implementations; it talks only through
  `GraphWorkspaceOperations` provided by the API package.

---

## 6. Plugin system dependency flow

Plugins are discovered from the `grafy.plugins` entry-point group and
installed into a frozen `PluginRegistry`. The registry is the single source of
truth for nodes, artifact types, conversions, resolvers, and writers.

```mermaid
flowchart TB
    subgraph Core["grafy_core"]
        Registry["PluginRegistry\nnodes · artifact_types · conversions"]
        Contract["Plugin contract\nNode · ArtifactTypeSpec · ArtifactConversion\nResolverFactory · WriterFactory"]
        Runtime["Runtime\nNodeRuntime · InputMaterializer · OutputPersister"]
    end
    subgraph Builtin["Builtin operators (core)"]
        Ops["operators/\nimage · sequence · arithmetic · text\nschema · prompt · table · modules"]
    end
    subgraph External["External plugins"]
        LLM["grafy_plugin_llm"]
        OCR["grafy_plugin_ocr"]
        GIS["grafy_plugin_gis"]
        SQL["grafy_plugin_sql"]
    end

    Ops --> Registry
    LLM --> Registry
    OCR --> Registry
    GIS --> Registry
    SQL --> Registry
    Registry --> Contract
    Contract --> Runtime
    Registry -. "entry point discovery" .-> EntryPoints["importlib.metadata\ngroup='grafy.plugins'"]
```

**Registration path** — each `Plugin` declares:
- `node(...)` / `function_node(...)` → `NodeRegistration`
- `register_artifact_type(...)` → `ArtifactTypeSpec`
- `register_artifact_conversion(...)` → `ArtifactConversion`
- `register_resolver(...)` / `register_writer(...)` → runtime factories

**Validation on `freeze()`:** the registry verifies every port references an
installed artifact type, conversions meet compatible runtime types, field
projections target installed types, scalar targets are unique, and conversion
chains are type-compatible. It then expands derived field projections for
JSON-Schema string/integer leaves.

---

## 7. Graph execution dependency flow

Execution is a pipeline: **preflight → compile → run → persist**. The
composition root wires one in-process engine behind the
`GraphExecutionEngine` protocol.

```mermaid
flowchart LR
    Run["RunGraph"] --> Preflight["GraphRunPreflight\nvalidate graph + registry"]
    Run --> Compiler["GraphCompiler\ncompile nodes/edges to plan"]
    Run --> Engine["GraphExecutionEngine\nin-process"]
    Engine --> Coordinator["GraphExecutionCoordinator"]
    Coordinator --> NodeExec["NodeExecutionService"]
    NodeExec --> Runtime["NodeRuntime"]
    Runtime --> Materializer["InputMaterializer\n(resolvers)"]
    Runtime --> Persister["OutputPersister\n(writers)"]
    Runtime --> Cache["PersistentInvocationCache"]
    NodeExec --> Secrets["NodeSecretResolverPort"]
    Compiler --> Catalog["GraphModuleCatalog\nmodule-library catalog"]
    Catalog --> Registry["PluginRegistry"]
    Coordinator --> History["ExecutionHistoryService"]
    Engine --> Manager["RunExecutionManager\ninterrupt / lifecycle"]
```

**Node execution steps:**
1. `InputMaterializer` resolves each input `ArtifactRef` to a Python value using
   the registered `Resolver` instances (builtin integer/text + plugin resolvers).
2. `NodeRuntime` runs the node `run(context, config, inputs)` and checks the
   invocation cache policy.
3. `OutputPersister` writes produced artifacts via registered `ArtifactOutputWriter`
   instances.
4. `PersistentInvocationCache` stores content-addressed invocation results.

**Execution engine** (satisfies `GraphExecutionEngine`):
- `InlineExecutionEngine` — runs in-process; MAP items run concurrently,
  bounded by `map_max_concurrency` (default 4).

---

## 8. Persistence architecture

`grafy_persistence` implements every core port with a `Sql*Repository` and
exposes a unit-of-work facade. Alembic (`infra/db/migrations`) is the **only**
schema authority; SQLAlchemy ORM mappings (`orm.py`) are derived from the
`schema.py` table definitions.

```mermaid
flowchart TB
    subgraph CorePorts["grafy_core/ports"]
        P1["SavedGraphRepositoryPort"]
        P2["CollaborationRepositoryPort"]
        P3["IdentityRepositoryPort"]
        P4["ExecutionHistoryRepositoryPort"]
        P5["InvocationCacheRepositoryPort"]
        P6["NodeSecretRepositoryPort"]
        P7["ModuleLibraryRepositoryPort"]
        P8["StagedUploadRepositoryPort"]
        P9["TemplateRepositoryPort"]
    end
    subgraph Adapters["grafy_persistence"]
        R["adapters/repositories.py\nSqlSavedGraphRepository · SqlCollaborationRepository\nSqlIdentityRepository · Sql* ... (12 classes)"]
        UOW["unit_of_work.py\nSqlAlchemyUnitOfWork\nSqlAlchemySavedGraphUnitOfWork"]
        DB["database.py\ncreate_database"]
        ORM["orm.py + schema.py\nstart_mappers"]
    end
    subgraph DBs["Databases"]
        SQLite["SQLite (sqlite+aiosqlite)"]
        PG["PostgreSQL"]
    end

    P1 --> R
    P2 --> R
    P3 --> R
    P4 --> R
    P5 --> R
    P6 --> R
    P7 --> R
    P8 --> R
    P9 --> R
    R --> UOW
    UOW --> DB
    DB --> SQLite
    DB --> PG
    DB --> ORM
```

**Unit-of-work pattern:** `SqlAlchemyUnitOfWork` is an async context manager that
opens an `AsyncSession`, exposes the repositories its caller needs, commits on
success, and rolls back on failure. `SqlAlchemySavedGraphUnitOfWork` additionally
exposes the saved-graph repositories required by `SavedGraphService`.

---

## 9. Object storage dependency flow

`grafy_storage` exposes the `FileStoragePort` from core and provides two
adapters selected by `storage_backend` in settings (`local` | `s3`).

```mermaid
flowchart LR
    Core["FileStoragePort\ngrafy_core/ports/storage.py"] --> Factory["create_file_storage\ngrafy_storage/factory.py"]
    Factory --> Local["LocalFileObjectStore\nlocal FS root"]
    Factory --> S3["S3ObjectStore\nS3-compatible (MinIO/AWS)"]
    Local --> FS["filesystem"]
    S3 --> Endpoint["S3 endpoint"]
```

Artifacts and uploads are written through `FileStoragePort`, never through a
concrete store — so swapping local for S3 changes only the factory call in
`main.py`.

---

## 10. Deployment / infrastructure

`infra/docker/compose.yaml` orchestrates the production topology: an nginx
gateway, the API container, the web container, and a shared volume. Keycloak
provides OIDC.

```mermaid
flowchart TB
    User["Browser"] --> Gateway["nginx gateway\ngateway/nginx.conf"]
    User --> Web["Next.js web\nweb.Dockerfile"]
    Gateway --> API["FastAPI API\napi.Dockerfile"]
    Web --> API
    API --> DB["SQLite / PostgreSQL\n/data/workbench"]
    API --> Keycloak["Keycloak\nOIDC issuer"]
    API --> StorageBackend["local objects | S3"]
```

**Collaboration assumption:** the API runs **one Uvicorn process with one worker**
(`WEB_CONCURRENCY: "1"`), acquires an exclusive workspace lock at startup
(`GRAFY_REQUIRE_SINGLE_API_OWNER=true`), and relies on the graph-room hub for
in-process WebSocket collaboration.

---

## 11. Structural diagram: application services

The application layer is one cohesive service per aggregate/actor.

```mermaid
classDiagram
    class SavedGraphService {
        create()
        get(graph_id)
        list_revisions()
        list_accessible(actor)
        create_folder()
        list_folders()
    }
    class CollaborationService {
        bootstrap_graph()
        accept_command()
        checkpoint()
        replace_complete_document()
        copy_exact_head()
        delete_graph()
    }
    class IdentityService {
        list_workspaces()
        create_personal_access_token()
        provision_oidc_identity()
        authorize()
    }
    class ModuleLibraryService {
        list_library()
        publish_release()
        deprecate()
        import_release()
        catalog_definitions()
    }
    class TemplateService {
    }
    class AuthService {
        require_mcp_access()
        allow_login_start()
        replace_login_transaction()
    }
    class NodeSecretService {
        resolve_secret()
    }
    class PluginRegistry {
        node_registration(operator_id, version)
        build_node(operator_id, version)
    }

    SavedGraphService --> CollaborationService
    CollaborationService --> SavedGraphService
    CollaborationService --> ModuleLibraryService
    SavedGraphService --> IdentityService
    ModuleLibraryService --> IdentityService
    AuthService --> IdentityService
    NodeSecretService --> PluginRegistry
```

---

## 12. Runtime structural diagram

```mermaid
classDiagram
    class NodeRuntime {
        run(context, config, inputs)
    }
    class InputMaterializer {
        resolve inputs
    }
    class OutputPersister {
        persist outputs
    }
    class ResolverRegistry {
        resolvers
    }
    class ArtifactWriterRegistry {
        writers
    }
    class PersistentInvocationCache {
        get/put content-addressed
    }

    NodeRuntime --> InputMaterializer
    NodeRuntime --> OutputPersister
    NodeRuntime --> PersistentInvocationCache
    InputMaterializer --> ResolverRegistry
    OutputPersister --> ArtifactWriterRegistry
```

---

## 13. Dependency rules (enforcement)

These are the import constraints the codebase is designed around; new code should
preserve them.

```mermaid
flowchart LR
    subgraph Allowed["Allowed: policy → ports/domain"]
        API["apps/api"] --> Core["core/application · core/ports · core/domain"]
        MCP["apps/mcp"] --> Ops["GraphWorkspaceOperations (API-provided)"]
    end
    subgraph Forbidden["Forbidden: policy → concrete infra"]
        App["application / runtime"] -. "✗" .-> Sql["SqlAlchemy*"]
        App -. "✗" .-> Storage["LocalFileObjectStore / S3ObjectStore"]
        App -. "✗" .-> SDK["Mistral / OpenAI SDK"]
        Core -. "✗" .-> FastAPI["FastAPI routes"]
    end
```

- **Core never imports FastAPI, SQLAlchemy, storage stores, or SDK clients.**
- **`grafy_mcp` never imports FastAPI routes, persistence, storage, or plugin
  implementations.**
- **Concrete construction lives only in composition roots** (`main.py`,
  `composition.py`, `factory.py`).
- **Ports are narrow and per-aggregate** (ISP); `UnitOfWorkPort` facades expose
  only the repositories the caller needs.
- **Plugins are the extension mechanism** (OCP): new nodes/artifact types are
  added by installing a plugin, not by modifying core.

---

## 14. Test coverage map

Tests live under `tests/` and are colocated with the units they protect.
See `pytest.ini` and `pyproject.toml` for the strict-basedpyright + ruff + pytest
toolchain.

| Area | Location | What it protects |
| --- | --- | --- |
| Unit | `tests/unit` | Domain aggregates, ports, runtime, plugin registry. |
| API | `apps/api` (pytest) | Route handlers, auth, collaboration, execution. |
| Integration | `tests/` | Repository/UoW adapters, storage, end-to-end HTTP. |

---

## 15. References

- `CONTEXT.md` — product vocabulary and active scope.
- `docs/workbench-interaction-plan.md` — interaction and runtime decisions.
- `docs/adr/0002-server-authoritative-workbench-collaboration.md` — collaboration authority.
- `docs/adr/0003-authenticate-users-and-scope-collaboration-to-workspaces.md` — auth/tenancy.
- `docs/design/authentication-and-workspace-tenancy.md` — identity & tenancy design.
- `SOLID_REVIEW.md` — SOLID audit of the backend scope.
