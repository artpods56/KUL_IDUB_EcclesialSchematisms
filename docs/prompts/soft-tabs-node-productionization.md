# Task: Productionize the "Soft Tabs" node spike (type inspector, artifacts appendix, renderer registry)

## Objective

Transfer the node design and features prototyped under `apps/web/src/components/canvas/spikes/` into the production workbench canvas, then remove the spike. After this task the production `WorkflowNodeCard` should look and behave like the spike's `SoftTabsNode`, and the spike route/files should be gone.

Everything described below already works on the spike page (`/spikes/nodes`) — use it as the reference implementation and visual spec before deleting it.

## What the spike adds over production

Production node: `apps/web/src/components/canvas/nodes/WorkflowNode.tsx` (`WorkflowNodeCard`).
Spike node: `apps/web/src/components/canvas/spikes/SoftTabsNode.tsx`.

1. **Soft Tabs visual design.** Ports render as pill "tabs" hugging the card edge (inputs flush left with `border-radius: 0 9999px 9999px 0`, outputs flush right, mirrored), with the React Flow `Handle` overlapping the pill edge. The card is a soft 300px shell (`tokens.radiusLg`, `tokens.shadowNode`, no border). See `spikes/soft/primitives.tsx` (`soft.shell`, `SoftShell`, `SoftHeader`) and the `tab`/`tabIn`/`tabOut` styles in `SoftTabsNode.tsx`. Body content (upload UI / config fields) stays functionally identical to production's `ImageUploadBody`/`GenericBody`; the spike variants are `SoftUploadBody`/`SoftConfigBody` in `primitives.tsx`.

2. **Port type inspector.** Every port tab is a Base UI `Popover.Trigger`. The popover (`spikes/type-inspector.tsx`) shows:
   - the port contract in mono, colored by artifact type: `artifact_type@schema_version`, wrapped as `list[...]` when the effective shape is `many`;
   - the port description;
   - the artifact type's `payload_schema` (JSON schema) rendered as a nested field tree with Python-style type labels via `schemaTypeLabel` (`str`, `int`, `float`, `bool`, `list[str]`, `list[object]`…), required-field `*` markers, and indentation for nested objects / arrays-of-objects;
   - a "Projectable fields" section listing the type's `field_projections` (`.path → target_artifact_type_id`);
   - types with an empty `payload_schema` (e.g. `image.raster`) show "No declared payload schema — this artifact carries opaque content."

3. **Header.** "?" (description popover) and "x" (remove node) icon buttons in the **top-left corner**, then the title; operator id line below. No status pill. Production `WorkflowNodeCard` already has both buttons but places them on the right — adopt the spike's top-left placement. Node removal must keep using `data.onRemoveNode?.(id)` (wired in `apps/web/src/app/page.tsx`); drop the spike's `useReactFlow().deleteElements` fallback, it existed only because the spike page had no callback.

4. **Produced artifacts appendix** (replaces production's inline `ProducedArtifactsAppendix` footer). Reference: `spikes/soft/artifacts-appendix.tsx`.
   - A **detached card under the node** (same 300px width, own `tokens.shadowNode` surface, ~10px gap), rendered as a sibling after the shell inside the node component (React Flow wraps both; output handles stay on the shell).
   - One **section per `RunPortOutput`** with artifacts. Header row: port name (uppercase), kind badge (`single` or `sequence · N` from `output.kind` — never derive item identity from payload fields), and a mode toggle.
   - `single` outputs render one flat item, no pager.
   - `sequence` outputs get a numeric pager: windowed number chips (`pageWindow` — all pages when ≤ 7, else `1 2 … current±1 … n-1 n`) plus an integer input (`current / N`) that clamps to range.
   - **Field projection select** (sequences with JSON payloads only): options derived from top-level `payload_schema` properties of scalar or list-of-scalar type, labeled `map .field → list[type]`. When active it replaces the pager and shows the mapped list (pretty: indexed rows; raw: the projected JSON array) plus a `list[type] · N items` line. This is a *preview* of edge-level projection; keep the schema-derived options as in the spike (the API's `field_projections` are narrower — only paths with registered target artifact types — and remain the source of truth for actual edge projections).
   - The appendix root carries `className="nodrag nowheel"`.

5. **Modular artifact renderer registry.** Reference: `spikes/soft/artifact-renderers.tsx`. The appendix delegates rendering of the focused artifact to the first matching `ArtifactRendererSpec`:
   - interface: `{ id, modes: readonly string[], matches(artifact, payload?), Component({ artifact, payload, mode }) }`; first mode is the default; the appendix builds its mode toggle from the active renderer's `modes`;
   - `image` — matches `content_type.startsWith("image/")` with a `content_url`; modes `preview` (renders `<img>` via `artifactContentUrl`) / `meta`;
   - `json` — matches when a payload is available or `content_type === "application/json"`; modes `pretty` (the `PrettyValue` key/value tree: strings as text, numbers in accent mono, scalar lists as chips, nested object lists as indented bordered groups) / `raw` (formatted JSON);
   - `meta` — fallback; renders artifact metadata rows (type@version, content_type, byte_size, text, artifact_id).
   - The registry exists so future types (tables, CSV bundles…) are added by appending an entry, without touching the appendix.

## Production integration work (this is the part the spike faked)

- **Artifact type specs.** The spike threads an `artifactTypes: Record<string, ArtifactTypeSpec>` map through node data (`SpikeNodeData` in `spikes/shared.ts`). In production, don't thread it through node data: the registry is already fetched with SWR via `useNodeRegistry()` (`apps/web/src/hooks/use-api.ts`, `GET /v1/nodes` → `NodeRegistryResponse.artifact_types`). Call it (or a small `useArtifactType(id)` helper) directly from the inspector/appendix components — SWR dedupes.
- **JSON payloads.** The spike hardcodes payloads in `data.payloads`. In production, `ArtifactSummary` has no inline payload; for `application/json` artifacts fetch the payload from `artifactContentUrl(artifact.content_url)` (`apps/web/src/lib/api/workbench.ts`) lazily — only for the focused artifact, cached (SWR keyed by artifact id is fine). Handle missing/failed fetches by falling back to the `meta` renderer. If JSON artifacts ship no `content_url`, the appendix must degrade gracefully to metadata display; use an installed plugin compound artifact or a test fixture to cover that case rather than relying on a production arithmetic compound type.
- **Run data** is already wired: `apps/web/src/app/page.tsx` sets `node.data.run` from `runGraph` responses (`RunNodeResponse.outputs[]` with `kind: "single" | "sequence"`). The appendix consumes exactly this shape. `data.execution.error` display must survive the redesign (production shows it in the body).
- **Handles/connections must not regress.** Keep `encodeHandleId(portMetaForPort(port, shape))`, `effectivePortShape(data, port)`, `handleStyle(...)` (`canvas/handles.ts`, `canvas/types.ts`, `ARTIFACT_TYPE_COLOR` in `canvas/nodes.css.ts`), and the `useUpdateNodeInternals` effect keyed on `mappedInputPort`, port counts, config field count, and produced artifact count (the appendix changes node height). Saved graphs (`canvas/saved-graph.ts`) and edge editing (`canvas/edges/WorkflowEdge.tsx`) depend on handle ids staying stable.

## File plan

Promote out of `spikes/` (adjust imports; these become production modules):
- `spikes/type-inspector.tsx` → e.g. `components/canvas/nodes/type-inspector.tsx` (drop the `portColor` import from `spikes/shared.ts`; that helper is 3 lines — inline it or move it near `ARTIFACT_TYPE_COLOR`).
- `spikes/soft/artifact-renderers.tsx` → e.g. `components/canvas/nodes/artifact-renderers.tsx`.
- `spikes/soft/artifacts-appendix.tsx` → e.g. `components/canvas/nodes/ArtifactsAppendix.tsx` (rework `SpikeNodeData` → `WorkflowNodeData` + payload fetching as above).
- The Soft Tabs shell/header/body styles from `spikes/soft/primitives.tsx` and the tab layout from `spikes/SoftTabsNode.tsx` get merged **into** `nodes/WorkflowNode.tsx` (or split into siblings) — production keeps one node card component, registered as `workflowNode` in `WorkflowCanvas.tsx`. Do not keep two parallel node implementations.

Delete after porting:
- `apps/web/src/app/spikes/` (the whole route),
- `apps/web/src/components/canvas/spikes/SoftTabsNode.tsx`, `spikes/soft/primitives.tsx`, `spikes/shared.ts`, `spikes/registry.ts`, `spikes/type-inspector.tsx`, `spikes/soft/artifact-renderers.tsx`, `spikes/soft/artifacts-appendix.tsx`,
- `spikes/gazetteer.ts` — demo fixture (fake operator + Słownik Geograficzny sample payloads); do **not** port it anywhere,
- the stray `spikes/soft/bands/workbench-interaction-plan.md`.

Keep: `apps/web/src/components/canvas/spikes/archive/bands/` — explicitly parked for future exploration (see its README). If the `spikes/` folder would otherwise be empty, `archive/` can move to `components/canvas/archive/`.

Already shared with production (leave as is): `components/canvas/config-schema.ts` (`schemaFields`).

## Constraints

- StyleX only, all values from `src/lib/stylex/tokens.stylex.ts` tokens (`light-dark()` aware — no hardcoded colors).
- Base UI (`@base-ui/react/popover`) for popovers; `lucide-react` icons (`CircleHelp`, `X`).
- Every interactive element inside a node needs `className="nodrag"` (+ `nowheel` for scrollables) or React Flow will swallow events.
- Don't regenerate `src/lib/api/generated/grafy.ts`; all needed types exist: `ArtifactTypeSpec`, `ArtifactSummary`, `RunNodeResult`, `NodeSpec`, `Port`, `NodeRegistry` from `@/lib/api`.
- React Compiler is on — avoid manual `useMemo` on values derived from props it can't verify (it produced a lint error in the spike once; plain derivation is fine).

## Verification

1. `cd apps/web && npx tsc --noEmit && npx eslint src`.
2. Run the API (`uv run uvicorn grafy_api.main:app --port 8791` from repo root) and web (`GRAFY_API_UPSTREAM=http://127.0.0.1:8791 npm run dev` in `apps/web`), open the workbench, and check:
   - add nodes from the palette; port tabs render, connections still validate (type + shape) and edge pills/projection editing still work;
   - port tab click → type inspector with schema tree (try any installed plugin compound output; automated projection coverage uses `test.compound_result@1` rather than a production arithmetic tutorial node);
   - "?" and "x" in the top-left work; "x" removes the node;
   - upload images → run the graph → appendix appears under nodes: OCR output should page as a sequence, single outputs render flat; image artifacts show a preview; JSON-payload artifacts show pretty/raw (or degrade to meta if no content_url);
   - node dragging still works when grabbing the header/shell (no dead zones from stray `nodrag`).
3. `/spikes/nodes` returns 404 and no imports from `components/canvas/spikes/` remain (`rg "canvas/spikes" apps/web/src`).
