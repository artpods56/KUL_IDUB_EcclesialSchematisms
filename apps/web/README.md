# Notarius Workbench

A ComfyUI-like workbench for assembling typed artifact transformations over the
Notarius runtime. It is built with Next.js 16, StyleX, Base UI
primitives, React Flow, and SWR.

The default slice is a buildable arithmetic field-projection workflow. It opens
with four disconnected arithmetic nodes so the connection behavior can be
tested directly; **Wire example** installs the canonical four-edge graph in one
click. Two integer nodes feed an Add & subtract node whose single `result`
artifact contains `addition` and `subtraction`. Each nested field is routed over
a labeled edge into Multiply, without adding adapter nodes or
operation-specific integer artifact types. The OCR and table operators are
available in the node library when the OCR extra is installed.

Node configuration is rendered directly inside each node from the live
`config_schema`. Primitive JSON Schema fields use a compact preset: text,
number/integer (including bounds), boolean, or enum selection. Source files and
run artifacts are also managed from the node, so the canvas does not depend on
a separate inspector sidebar. The node header keeps only two local controls:
help from the registered description and removal from the canvas.

Connections own transport behavior. Every edge has an inline control for
choosing the whole output or a declared nested-field projection, showing
whether the target receives the value directly or maps each list item, and
removing the connection. Different outgoing edges from one output can therefore
carry different fields without changing the source node.

## Source of truth

The authoritative backend contract for this app is the workbench implementation:

- `libs/core/src/notarius_core/` owns node, port, artifact, resolver,
  persistence, and runtime behavior.
- `apps/api/src/notarius_api/schemas/workbench.py` owns the HTTP models.
- `apps/api/src/notarius_api/v1/routes/workbench.py` exposes the node
  catalog and execution routes.

The web workbench depends on no routes outside this contract.

## Generated transport types

The web transport contract is generated from the canonical FastAPI app:

```bash
cd apps/web
npm run generate:api
```

This updates:

- `openapi/notarius.json`
- `src/lib/api/generated/notarius.ts`

Refresh the OpenAPI JSON and check that the committed TypeScript types are
current:

```bash
npm run check:api
```

Do not hand-edit the generated TypeScript file. `contract.ts` supplies
short application-facing aliases derived from generated operations and schemas.

The node, port, request, and response envelopes are generated from the API.
Node-specific `config_schema`, `input_schema`, `output_schema`, artifact
`payload_schema`, and run-node `config` remain dynamic JSON objects. This lets
the backend advertise new node shapes without requiring a handwritten frontend
type for every node.

## Running locally

Requirements:

1. Node 20+ and npm.
2. Python 3.12.9 with the repository uv workspace synced.
3. The Notarius API running on `http://localhost:8000`.
4. `make install-ocr` and `make api-ocr` used from the repository root when OCR
   operators should be available.
5. `MISTRAL_API_KEY` configured in the API process environment only when using
   the Mistral OCR operator.

```bash
cd apps/web
npm ci
npm run dev
```

Open <http://localhost:3000>.

The API key stays on the server. It is never stored in node configuration or
sent to the browser.

Saved graphs have canonical browser URLs at
`/workspaces/local/graphs/{graph_uuid}`. A blank draft uses
`/workspaces/local/graphs/new`; the root route retains the arithmetic example.
`local` is intentionally the only accepted workspace slug today. It names the
single active workbench in the URL, but it is not yet a tenant or authorization
boundary.

## Current flow

1. `GET /v1/nodes` returns the catalog, typed ports, JSON schemas, and
   artifact type definitions.
2. The canvas opens with Number 9, Number 4, Add & subtract, and Multiply but no
   connections. The user may wire the ports manually or choose **Wire example**.
3. Add & subtract emits one `arithmetic.result@1` artifact containing
   `{addition: 13, subtraction: 5}`.
4. Two independent edges select `result.addition` and `result.subtraction`; the server
   materializes each as `scalar.integer@1` for Multiply, which produces `65`.
5. Dragging the compound result to a compatible integer input opens a picker
   populated from the artifact type's declared field projections. The created
   edge remains editable and owns that choice.
6. List-valued edges expose whether the target receives the whole list directly
   or maps the target operation over each item. Mapping is edge state, not node
   invocation configuration.
7. Drag-selecting nodes enables **Run selected**, which sends the selected nodes,
   their internal edges, and incoming edges crossing from unselected upstream
   nodes. Each crossing edge pins the exact `ArtifactRef` or
   `ArtifactRefSequence` from its source port's latest visible successful output;
   its projection and `direct`/`map` mode still apply. The upstream source is not
   re-executed, and its result remains untouched. If that current output is
   absent, the run is refused instead of asking the server for a fuzzy "latest"
   artifact. **Run all** executes the complete graph.
8. Editing a node's schema-derived controls invalidates stale run results before
   the next execution.
9. `POST /v1/runs` validates the graph, collection modes, and projection paths before
   executing it through the runtime.
10. Node results expose values and content URLs served by
   `GET /v1/artifacts/{artifact_id}/content`.

Run results and selected-run pins are transient client/API process state under
the current architecture. They are not restored after a page reload or API
restart, even when the graph itself has been saved.

## Project layout

```text
src/
  app/                         # workbench route and global styles
  components/
    canvas/                    # React Flow adapter, edge controls, and node rendering
    ui/dialog.tsx              # field-projection picker primitive
    providers.tsx              # registry SWR and theme providers
    theme.tsx                  # persisted light/dark/system preference
  hooks/use-api.ts             # node registry hook
  lib/
    api/
      generated/notarius.ts    # generated OpenAPI transport types
      contract.ts              # application-facing generated aliases
      workbench.ts             # workbench HTTP calls
    stylex/                    # design tokens
```

## Verification

```bash
npm run check:api
npm run lint
npm run typecheck
npm run build
```
