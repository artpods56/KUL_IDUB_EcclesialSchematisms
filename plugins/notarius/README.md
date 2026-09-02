# Grafy Notarius plugin

The Notarius plugin extracts structured JSON from an ordered raster image
sequence. It also converts an extraction dataset to `table.data@1` for JSON or
CSV download.

## Nodes

### `notarius.dataset.extract_structured@1`

Connect these inputs:

- `images`: an ordered `image.raster@1` sequence.
- `system_prompt`: a `scalar.text@1` artifact with extraction rules.
- `instruction`: a `scalar.text@1` artifact with the per-image task.
- `json_schema`: a `json.schema@1` artifact for each structured response.

The node owns dataset iteration and context management. With
`sliding_window` or `full_history`, it processes images sequentially and keeps
prior user and assistant messages. With `independent`, it processes images
concurrently up to `max_concurrent`.

When `lookahead_images` is enabled, each request contains the current image and
the next image. The generated message identifies the next image as lookahead
and tells the model not to extract records that belong only to it. Historical
user messages keep their text but drop their images.

### `notarius.dataset.to_table@1`

Connect a `notarius.extraction.dataset@1` artifact. If the response schema contains one
top-level array of objects, the node emits one table row per array item. If the
schema contains several such arrays, set `rows_field`. If the schema contains
none, the node emits one row per source image.

The output uses `table.data@1`. Grafy can download the artifact as JSON or CSV.

## Verify the plugin

Run these commands from this directory:

```shell
uv sync --locked --no-sources --find-links wheels
uv run pytest -q
/opt/homebrew/bin/basedpyright src tests
```
