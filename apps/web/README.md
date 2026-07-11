# Notarius Studio — Prototype Workbench

A ComfyUI-like workbench for assembling typed artifact transformations over the
Notarius prototype runtime. It is built with Next.js 16, StyleX, Base UI
primitives, React Flow, and SWR.

The default slice is a buildable arithmetic field-projection workflow. It opens
with four disconnected arithmetic nodes so the connection behavior can be
tested directly; **Wire example** installs the canonical four-edge graph in one
click. Two integer nodes feed an Add & subtract node whose single `result`
artifact contains `addition` and `subtraction`. Each nested field is routed over
a labeled edge into Multiply, without adding adapter nodes or
operation-specific integer artifact types. The OCR and table operators remain
available in the node library.

Node configuration is rendered directly inside each node from the live
`config_schema`. Primitive JSON Schema fields use a compact preset: text,
number/integer (including bounds), boolean, or enum selection. Source files and
run artifacts are also managed from the node, so the canvas does not depend on
a separate inspector sidebar.

## Source of truth

The authoritative backend contract for this app is the prototype implementation:

- `libs/core/src/notarius_core/prototype/` owns node, port, artifact, resolver,
  persistence, and runtime behavior.
- `apps/api/src/notarius_api/schemas/prototype.py` owns the HTTP models.
- `apps/api/src/notarius_api/v1/routes/prototype.py` exposes the prototype node
  catalog and execution routes.

The older workflow, project, and artifact routes are not part of the web
workbench contract.

## Generated transport types

The web transport contract is generated from an isolated FastAPI app containing
only `/v1/prototype/*` routes:

```bash
cd apps/web
npm run generate:prototype-api
```

This updates:

- `openapi/prototype.json`
- `src/lib/api/generated/prototype.ts`

Refresh the OpenAPI JSON and check that the committed TypeScript types are
current:

```bash
npm run check:prototype-api
```

Do not hand-edit the generated TypeScript file. `prototype-contract.ts` supplies
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
4. `MISTRAL_API_KEY` configured in the API process environment only when using
   the Mistral OCR operator.

```bash
cd apps/web
cp .env.local.example .env.local  # optional
npm install
npm run dev
```

Open <http://localhost:3000>.

The API key stays on the server. It is never stored in node configuration or
sent to the browser.

## Current flow

1. `GET /v1/prototype/nodes` returns the catalog, typed ports, JSON schemas, and
   artifact type definitions.
2. The canvas opens with Number 9, Number 4, Add & subtract, and Multiply but no
   connections. The user may wire the ports manually or choose **Wire example**.
3. Add & subtract emits one `arithmetic.result@1` artifact containing
   `{addition: 13, subtraction: 5}`.
4. Edges select `result.addition` and `result.subtraction`; the server
   materializes each as `scalar.integer@1` for Multiply, which produces `65`.
5. Dragging the compound result to a compatible integer input opens a picker
   populated from the artifact type's declared field projections.
6. Editing a node's schema-derived controls invalidates stale run results before
   the next execution.
7. `POST /v1/prototype/run` validates the graph and projection paths before
   executing it through the prototype runtime.
8. Node results expose values and content URLs served by
   `GET /v1/prototype/artifacts/{artifact_id}/content`.

## Project layout

```text
src/
  app/                         # workbench route and global styles
  components/
    canvas/                    # React Flow adapter and node rendering
    ui/                        # shared StyleX UI primitives
    providers.tsx              # SWR, tooltip, and toast providers
  hooks/                       # data hooks
  lib/
    api/
      generated/prototype.ts   # generated OpenAPI transport types
      prototype-contract.ts    # application-facing generated aliases
      prototype.ts             # prototype HTTP calls
    stylex/                    # design tokens
```

## Verification

```bash
npm run check:prototype-api
npm run lint
npm run typecheck
npm run build
```
