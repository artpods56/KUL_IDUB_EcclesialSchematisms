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

- `src/app/page.tsx` is the **home screen** (`/`). After authentication, users
  land here with a greeting, recent graphs, and workspace quick links. It is
  **not** a redirect.
- `/workspaces` is the full workspace directory with search, sort, and
  create/join dialogs. Accessible from the home screen's "All workspaces" link
  and from the sidebar rail.
- `/workspaces/{slug}` is the workspace overview. `/workspaces/{slug}/graphs/{id}`
  is the workbench canvas.

## Shared components and call sites

- `WorkspaceRail` is used in **two** call sites: the home page (`page.tsx`) and
  the workspace layout (`[workspaceSlug]/layout.tsx`). Any prop added outside
  the core interface must be optional.
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

- Cross-workspace data (e.g., graphs from all workspaces) requires explicit
  multi-key SWR usage. See `useAllWorkspacesGraphs` in
  `src/hooks/use-api.ts` for the pattern.
- Barrel exports: `src/lib/api/index.ts` re-exports everything.
  Features like `useAuthSession` must be imported from their full file path
  unless a barrel exists.
<!-- END:web-app-conventions -->
