# Web sandbox

A **development-only** place to spike UI against real Grafy chrome (StyleX
tokens, overlay, catalog node preview, port marks) without touching workbench
feature code.

- URL: [http://127.0.0.1:3000/sandbox](http://127.0.0.1:3000/sandbox)
- Production: the route `notFound()`s. Do not link it from product navigation.
- Auth: still behind the normal session boundary (same as the rest of the app).
- Direction: sandbox may import `features/`, `components/`, `lib/`. **Never**
  import sandbox from those trees.

## Add a spike

1. Create `src/sandbox/spikes/<id>/<Name>Spike.tsx` (client component).
2. Register it in `catalog.ts`.
3. Reuse fixtures under `src/sandbox/fixtures/` instead of inventing a second
   node language. Import `CatalogNodePreview`, `overlay`, `tokens`,
   `portMarkStyle` — do not restyle a fake node.
4. Keep variants inside the spike. Do not patch `WorkflowNode`,
   `type-inspector`, or other product files “to make the spike work”.

## This folder vs Cursor canvases

Cursor `.canvas.tsx` files cannot import Grafy. Use this sandbox when the
question is shape, chrome, or type color. Use a Cursor canvas for tables,
copy decks, or anything that does not need to look like the workbench.
