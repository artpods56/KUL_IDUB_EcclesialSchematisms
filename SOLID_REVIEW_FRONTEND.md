# SOLID Review

## Scope
- Mode: repository (frontend)
- Reviewed: the Grafy workbench frontend — `apps/web` (Next.js 16 + React 19 + React Flow + Stylex). Source under `src/`: `app/` (route pages), `features/` (auth, graphs, templates, workspaces, workbench), `hooks/` (SWR data hooks), `lib/api/` (generated OpenAPI client + domain-specific modules), `components/` (shared UI), `lib/stylex/` (tokens). ~118 non-generated TypeScript/TSX source files; ~68,000 lines.
- Excluded: generated/vendor code — `src/lib/api/generated/grafy.ts` (openapi-typescript output, 6,035 lines), `node_modules`, `.next`, and test files (`*.test.ts`, `*.test.tsx`, `*.spec.ts`).

## Summary
- Findings: 1
- By principle: SRP 1, OCP 0, LSP 0, ISP 0, DIP 0
- By severity: High 0, Medium 1, Low 0

## Findings

### [SOLID-SRP-001] `WorkbenchBody` orchestrates graph authoring, artifact-viewer presentation, collaboration protocol, and execution in one function
- Severity: Medium
- Confidence: Medium
- Location: `apps/web/src/features/workbench/ui/Workbench.tsx:261`
- Related locations: `apps/web/src/features/workbench/ui/Workbench.tsx:1778` (`onNodesChange`), `Workbench.tsx:2444` (`onConnect`), `Workbench.tsx:1300` (`onCommandAccepted` room callback), `Workbench.tsx:3082` (`allCanvasNodes`), `Workbench.tsx:3159` (`allCanvasEdges`), `Workbench.tsx:3424` (return)
- Change attribution: Not applicable
- Principle: Single Responsibility Principle
- Evidence: `WorkbenchBody` is a single function spanning lines 261–3424 (~3,160 lines) that owns the orchestration of several distinct change drivers in interleaved callbacks:
  - Graph authoring: `onNodesChange` (line 1778) maps React Flow changes to authoring commands; `onConnect` (line 2444) creates workflow edges; `addCatalogNode`, `duplicateSelectedNodes`, `deleteSelectedNodes` mutate the authored document.
  - Artifact-viewer presentation: the same `onNodesChange` filters `artifactViewerChanges` and `annotationChanges` (lines ~1850–1880); `onConnect` routes to `ArtifactViewerEdge` and `ArtifactViewerInteractionEdge` bindings (lines 2444–2530); `onCommandAccepted` updates viewer positions/annotations.
  - Collaboration protocol: the `useGraphRoomSession` options block (lines ~1300–1500) includes `onCommandAccepted`, `onCommandRejected`, `onTerminalClose`, and `presentationRoomSyncRef`, which translate room commands into authoring commands, artifact-viewer state, and presentation sync.
  - Execution lifecycle: `useRunExecution` is invoked at line 1571 but its orchestration, error routing (`setRunError` at lines 520, 837, 878, 919, 942, 963, 996, 1047, 1287, 1398, 1443, 1471), and canvas assembly (`allCanvasNodes`/`allCanvasEdges`) remain inline.
  These callbacks cross domain boundaries: `onCommandAccepted` (room → authoring + presentation), `onNodesChange` (authoring → presence + presentation), and `onConnect` (three connection types) all live in the one function.
- Violation: The SRP reference holds that a unit mixes change drivers when more than one actor would request changes to it. `WorkbenchBody` is a single function that owns graph authoring, artifact-viewer presentation, annotation management, collaboration room protocol handling, and execution orchestration. A change to the collaboration protocol (e.g., a new `RoomGraphCommand` kind such as `duplicate_node`, already present in `toLocalGraphCommand`) must be edited in the same `onCommandAccepted` callback that also applies authoring commands and artifact-viewer presentation state. A change to artifact-viewer presentation touches `onNodesChange`, `onConnect`, `onCommandAccepted`, and the canvas assembly — all in one function. The concerns are demonstrably separable (the codebase itself extracts `model/`, `canvas/`, and `room/` into pure-function modules), yet their orchestration is merged here.
- Impact: Mixed ownership across authoring, presentation, collaboration, and execution makes an ordinary change to any one of these concerns force edits in the same ~3,160-line function's unrelated sections, coupling otherwise-independent change drivers.

## Coverage and limitations

**Inspected.** I traced composition roots, extension points, subtype families, and every candidate type-switch across the reviewed scope.

- **SRP / cohesion**: All `model/` modules (`graph-document.ts`, `graph-authoring.ts`, `execution-plan.ts`, `node-catalog.ts`, `connection-feeds.ts`) are cohesive pure-function modules per aggregate. `canvas/` modules (`types.ts`, `saved-graph.ts`, `handles.ts`, `input-plugs.ts`, `node-layout.ts`, `annotations.ts`, `artifact-viewer.ts`, `artifact-interactions.ts`, `config-schema.ts`, `schema-builder.ts`, `query-artifact-tables.ts`, `node-secrets.ts`) are cohesive per canvas concern. The room modules (`protocol.ts`, `graph-room-session.ts`, `room-command-bridge.ts`) are cohesive around the collaboration protocol. The large UI hooks (`useRunExecution.ts`, `useSavedGraphLifecycle.ts`, `useNodeSecrets.ts`) and renderers (`WorkflowNode.tsx`, `geo-map-artifact-renderer.tsx`, `NodeSelector.tsx`, `Workbench.tsx`) are decomposed into cohesive sub-components/handlers; each was considered as an SRP candidate and discarded as a single-actor unit, except `WorkbenchBody`.
- **OCP / extension**: `artifact-renderers.tsx` defines an open-for-extension registry (`ARTIFACT_RENDERERS` + structural `matches` predicates + `rendererFor` fallback); adding a renderer appends to the array without modifying existing renderers — the idiomatic OCP registry, not a violation. `applyGraphCommand` (`model/graph-document.ts:292`) and `executionInvalidatedNodeIds` are command-dispatch over a deliberately bounded, domain-defined command journal (the same pattern accepted in the backend review). `parseServerRoomMessage` (`room/protocol.ts`) switches over a bounded wire protocol, not a genuine extension axis. `toRoomGraphCommand`/`toLocalGraphCommand` (`room/room-command-bridge.ts`) bridge two bounded vocabularies. JSON-schema type switches (`config-schema.ts`, `type-inspector.tsx`) and React Flow `Position` switches (`edge-path.ts`) cover closed sets. None met the OCP bar for a real, repeated variation axis.
- **LSP / subtype families**: No inheritance subtype families exist beyond React/`Error`/`Node`/`Edge` base types (`SavedGraphHydrationError`, `ApiError`, `GraphRoomCommandError`, `MemberListRefreshError` are leaf error classes). The artifact-renderer family uses structural `matches` composition rather than a substitutable base-type contract, so there is no caller-visible base contract to violate. No `isinstance`-style type checks on domain subtypes were found.
- **ISP / interfaces**: Hooks expose narrow, client-specific result/options interfaces (`UseNodeSecretsResult`, `UseNodeSecretsOptions`, `UseRunExecutionOptions`, `UseSavedGraphLifecycleOptions`, `UseGraphRoomSessionResult`, `GraphRoomSessionListeners`, `GraphRoomSessionOptions`), each shaped by a single consuming client. The API layer (`lib/api/`) is segregated by domain area (`contract`, `auth`, `workbench`, `workspaces`, `modules`, `templates`) and the generated client is a single generated file (excluded as vendor). No client was found forced to depend on operations it does not use.
- **DIP / dependency direction**: High-level feature code imports only `@/lib/api` abstractions and domain modules; the concrete fetch/CSRF client (`lib/api/client.ts`) is infrastructure consumed through `request`. Hooks receive concrete infrastructure through options interfaces (dependency injection) at component boundaries, and concrete construction occurs at intentional composition points (`WorkbenchBody` wires hooks/sub-components; `WorkspaceLayout`, `AuthSessionBoundary` are composition roots). No policy code directly imports or instantiates replaceable infrastructure outside those composition boundaries.

**Limitations.** No runtime/historical evidence (issue history, runtime regressions) was available to confirm behavioral coupling beyond static inspection. The `WorkbenchBody` finding is Medium-confidence because the idiomatic React Flow container pattern (a single canvas that must merge all node families) provides a partial contract-consistent counter-explanation; the demonstrated cross-domain interleaving in its callbacks is what supports the finding. No supported violations were omitted; the borderline candidates above each retain a contract-consistent explanation and are not reported per the review discipline.


