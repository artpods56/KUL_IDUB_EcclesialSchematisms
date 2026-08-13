# SOLID Review

## Scope
- Mode: repository (backend)
- Reviewed: the Notarius workbench backend — `apps/api` (FastAPI service), `apps/mcp` (MCP server), `libs/core` (ports, domain, application, runtime), `libs/persistence` (SQL adapters), `libs/storage`, and `plugins/` (llm, sql, ocr, gis). ~197 Python source files; ~46,260 lines.
- Excluded: generated/vendor code, `node_modules`, tests, and frontend/UI packages. None of the reviewed modules are generated.

## Summary
- Findings: 0
- By principle: SRP 0, OCP 0, LSP 0, ISP 0, DIP 0
- By severity: High 0, Medium 0, Low 0

## Findings

No supported SOLID violations were found in the reviewed scope.

## Coverage and limitations

**Inspected.** The review traced composition roots, port boundaries, subtype families, and every candidate extension point.

- **DIP / composition**: `apps/api/src/notarius_api/main.py` and `services/composition.py` are clean composition roots performing all concrete construction; the API surface and MCP server depend only on `notarius_core` ports and domain types. `Sql*Repository` classes in `libs/persistence/adapters/repositories.py` implement core ports structurally and are wired through `unit_of_work.py` (e.g. `SqlCollaborationRepository` provided as `CollaborationRepositoryPort`). High-level policy (application services) imports only ports and domain objects; no concrete infrastructure imports into policy code.
- **ISP / ports**: Core defines narrow, client-specific ports per aggregate — `IdentityRepositoryPort`, `SavedGraphRepositoryPort`, `CollaborationRepositoryPort`, `GraphExecutionHistoryRepositoryPort`, `ModuleLibraryRepositoryPort`, `InvocationCacheRepositoryPort`, `NodeSecretRepositoryPort`, `StagedUploadRepositoryPort`, `TemplateRepositoryPort`, plus `UnitOfWorkPort` facades exposing only the repositories their callers need. Runtime protocols (`NodeRuntime`, `ExecutionTaskRunner`, `GraphExecutionEngine`, `NodeSecretResolverPort`, `GraphModuleExecutorPort`) are each shaped by a single consuming policy.
- **LSP / subtype families**: LLM providers (`mistral.py`, `openai_compatible.py`) and OCR providers implement the same caller-visible protocol with substitutable contracts. Execution engines (`inline.py`, `prefect.py`) both satisfy `GraphExecutionEngine`; both task runners satisfy `ExecutionTaskRunner` with matching lifecycle semantics. Artifact writers/resolvers (`operators/tables.py`, `artifacts.py`) implement `ArtifactOutputWriter`/`Resolver` protocols uniformly. No subtype constrains inputs, relaxes outputs, or throws unsupported operations beyond its base contract.
- **OCP / extension**: `libs/core/plugins.py` is a genuine open-for-extension registry — plugins register `NodeRegistration`, `ArtifactType`, and `ArtifactConversion` objects without modification to core; resolvers and writers are registered by artifact type. Operator families are protocol-typed. The `apply_graph_command` dispatch in `domain/collaboration.py` (an `isinstance` switch over ~20 `GraphCommandKind` members) was reviewed as an OCP candidate; it is the idiomatic Python command-dispatch pattern over a deliberately bounded, domain-defined command journal with a single application-operation axis, and per the OCP reference's stated tradeoff (enum/switch favors adding operations) it is a contract-consistent procedural design — not a supported violation.
- **SRP / cohesion**: Application services are cohesive per aggregate/actor (`SavedGraphService`, `CollaborationService`, `IdentityService`, `ExecutionHistoryService`/`MaterializationService`/`RunResultPresenter`, `GraphModuleCatalog`). The large `AuthService` (~1,224 lines) spans OIDC protocol, credential issuance, and HTTP cookie/CSRF handling but delegates abuse control to a dedicated `AuthAbuseControl` and audit to domain events; `ArtifactService` (~1,314 lines) spans geo/raster/table content access but is cohesive around artifact content serving. Both were considered as SRP candidates and discarded as cohesive single-actor services. `repositories.py` (~1,986 lines) is a collection of 12 cohesive classes, one per port/aggregate.

**Limitations.** No runtime/historical evidence (tests, issue history) was available to confirm behavioral-substitution edge cases beyond static contract inspection. The frontend/UI scope was out of review scope. No supported violations were omitted; the borderline candidates above each retain a contract-consistent explanation and so are not reported as findings per the review discipline.
