# Table Plugin

First-party `builtin.table` Plugin package. It owns table import, normalization,
fuzzy matching, and the ordinary `table.data@1` writer/resolver registrations.

The producer-neutral value and artifact contracts live in
`grafy_core.table_contracts`; portable storage and bundle handling live under
`grafy_core.runtime`.

## Independent development

The package resolves the exact vendored `grafy-core==0.1.0` wheel without local
workspace sources:

```shell
uv sync --project plugins/table --no-sources --find-links plugins/table/wheels
uv run --project plugins/table pytest
```
