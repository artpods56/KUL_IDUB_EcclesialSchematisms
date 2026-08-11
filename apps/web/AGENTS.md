<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->

<!-- BEGIN:notarius-workbench-rules -->
# React Flow canvas checks

- Do not let global `svg` sizing or media resets constrain React Flow's edge
  layers. Keep any required override scoped to `.react-flow__edges > svg`.
- After changing node dimensions, handles, edge SVG styles, or canvas layout,
  verify at least one real pointer-drag connection in the rendered workbench.
  Compilation and programmatic edge insertion do not prove that wiring works.
<!-- END:notarius-workbench-rules -->

<!-- BEGIN:web-app-conventions -->
# Web app conventions

## Routing

- Graph is the primary user object. `src/app/page.tsx` (`/`) and `/graphs` both
  render the canonical cross-location graph browser after authentication.
- `/workspaces` is the secondary **Teams & access** administration surface.
  A personal workspace is presented as **My graphs**; a shared workspace is
  presented by its Team name. Do not expose slugs, roles, capabilities, or user
  UUIDs on graph-authoring surfaces.
- `/workspaces/{slug}` remains the location administration overview and
  `/workspaces/{slug}/graphs/{id}` remains the workspace-scoped workbench route.
  These routes preserve tenancy but are not the primary discovery hierarchy.

## Shared components and call sites

- `WorkspaceRail` is used by `GraphBrowser`, the Teams & access page, and the
  workspace layout (`[workspaceSlug]/layout.tsx`). Any prop added outside the
  core interface must be optional.
- `graphAgeLabel`, `sortGraphsByRecency`, and `filterGraphsByQuery` live in
  `WorkspaceGraphPanel.tsx` — they are reused outside the panel and should
  remain importable.

## CSS

- All classes use the `.ns-` prefix with BEM-ish structure:
  `.ns-workspace-rail__item`, `.ns-home__section-header`, etc.
- All styles are in `src/app/globals.css`. There are no per-component CSS files.
- Light/dark theming uses `light-dark()` — do not use raw hex literals for
  component styling.
- `--ns-rail-width` is the single source of truth for sidebar width. Main
  content padding must reference it via `calc(var(--ns-rail-width, 200px) + …)`.

## Tests

- `@testing-library/react` is **not** installed. Tests either use
  `react-dom/server`'s `renderToStaticMarkup` or import pure functions directly.
- Test files need `.tsx` extension when they contain JSX.
- `vitest run` is the test command. There is no separate `npm test`.

## API hooks

- Cross-workspace data (e.g., graphs from all workspaces) requires an explicit,
  typed multi-request SWR key and fetcher. See `useAllWorkspacesGraphs` in
  `src/hooks/use-api.ts`; do not pass an array of URLs to the global string
  fetcher.
- Barrel exports: `src/lib/api/index.ts` re-exports everything.
  Features like `useAuthSession` must be imported from their full file path
  unless a barrel exists.
<!-- END:web-app-conventions -->
