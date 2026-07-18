import json

from notarius_api.schemas.workbench import TableCellPreviewResponse


def table_cell_preview(
    value: object,
    max_cell_characters: int,
) -> TableCellPreviewResponse:
    if isinstance(value, int) and not isinstance(value, bool):
        display: str | float | bool | None = str(value)
    elif value is None or isinstance(value, str | float | bool):
        display = value
    else:
        display = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    if not isinstance(display, str) or len(display) <= max_cell_characters:
        return TableCellPreviewResponse(
            display=display,
            truncated=False,
        )
    return TableCellPreviewResponse(
        display=display[:max_cell_characters] + "…",
        truncated=True,
        original_length=len(display),
    )


__all__ = ["table_cell_preview"]
