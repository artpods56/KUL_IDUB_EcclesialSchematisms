from __future__ import annotations

import wandb
from typing import Dict, Sequence, Any, Tuple
from PIL import Image
import json
from statistics import mean

__all__ = [
    "create_eval_table",
    "add_eval_row",
    "create_summary_table",
    "format_comparison_md",
]

_DEFAULT_FIELDS: Tuple[str, ...] = (
    "page_number",
    "deanery",
    "parish",
    "dedication",
    "building_material",
)


def _build_columns(fields: Sequence[str]) -> list[str]:
    """Return list of column names for the W&B table."""
    base_cols = [
        "id",
        "pil_image",
        "page_info_json",
        "raw_llm_response",
        "lmv3_response",
        "parsed_llm_response",
        "comparison_md",  # human-readable summary
    ]
    metric_cols = []
    for fld in fields:
        for metric in ("TP", "FP", "FN", "support", "precision", "recall", "f1", "accuracy"):
            metric_cols.append(f"{fld}_{metric}")
    return base_cols + metric_cols


def create_eval_table(fields: Sequence[str] = _DEFAULT_FIELDS) -> wandb.Table:
    """Create and return an empty W&B table with predefined columns.

    Parameters
    ----------
    fields
        Iterable of field names that appear in the metrics dictionary produced by
        `src.data.stats.evaluate_json_response`.
    """
    columns = _build_columns(fields)
    return wandb.Table(columns=columns)


def add_eval_row(
    table: wandb.Table,
    sample_id: str,
    pil_image: Image.Image,
    page_info_json: Dict[str, Any],
    lmv3_response: Dict[str, Any],
    raw_llm_response: Dict[str, Any],
    parsed_llm_response: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    fields: Sequence[str] = _DEFAULT_FIELDS,
) -> None:
    """Add a single evaluation result row to the provided W&B table.

    The function will automatically compute *support* (TP + FN) for every field
    before inserting the data.

    Notes
    -----
    The ``metrics`` argument must follow the structure returned by
    ``src.data.stats.evaluate_json_response``.
    """
    comparison_md = format_comparison_md(page_info_json, lmv3_response, raw_llm_response,  parsed_llm_response)

    row_data: list[Any] = [
        sample_id,
        wandb.Image(pil_image),
        json.dumps(page_info_json, ensure_ascii=False),
        json.dumps(raw_llm_response, ensure_ascii=False),
        json.dumps(lmv3_response, ensure_ascii=False),
        json.dumps(parsed_llm_response, ensure_ascii=False),
        comparison_md,
    ]

    for fld in fields:
        fld_metrics = metrics.get(fld, {})
        tp = int(fld_metrics.get("TP", 0))
        fp = int(fld_metrics.get("FP", 0))
        fn = int(fld_metrics.get("FN", 0))
        precision = float(fld_metrics.get("precision", 0.0))
        recall = float(fld_metrics.get("recall", 0.0))
        f1 = float(fld_metrics.get("f1", 0.0))
        accuracy = float(fld_metrics.get("accuracy", 0.0))
        support = tp + fn

        row_data.extend([tp, fp, fn, support, precision, recall, f1, accuracy])

    table.add_data(*row_data)


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------


def _numeric_summary(values: Sequence[float]) -> tuple[float, float, float]:
    """Return (mean, minimum, maximum) for a sequence of numbers.

    The result is (mean, min, max) rounded to 4 decimal places for compactness.
    """
    if not values:
        return 0.0, 0.0, 0.0
    return round(mean(values), 4), round(min(values), 4), round(max(values), 4)


def create_summary_table(
    eval_table: "wandb.Table",
    fields: Sequence[str] = _DEFAULT_FIELDS,
    metrics_to_summarise: Sequence[str] = ("precision", "recall", "f1", "accuracy"),
) -> "wandb.Table":
    """Build an aggregated table (mean/min/max) for each *field + metric* pair.

    The output columns are: ``field``, ``metric``, ``mean``, ``min``, ``max``.
    """
    summary = wandb.Table(columns=["field", "metric", "mean", "min", "max"])

    for fld in fields:
        for metric in metrics_to_summarise:
            col_name = f"{fld}_{metric}"
            try:
                col_values = [float(v) for v in eval_table.get_column(col_name)]
            except KeyError:
                continue  # column might be absent if never logged

            m, mn, mx = _numeric_summary(col_values)
            summary.add_data(fld, metric, m, mn, mx)

    return summary


# ---------------------------------------------------------------------------
# Markdown comparison helpers
# ---------------------------------------------------------------------------


def _entries_to_table(entries: Sequence[Dict[str, Any]]) -> str:
    if not entries:
        return "(no entries)"

    # sort entries by parish (case-insensitive, fallback to empty string)
    entries_sorted = sorted(entries, key=lambda e: e.get("parish", "").lower())

    header = "| parish | dedication | material |\n|---|---|---|"
    rows = [
        f"| {e.get('parish', '')} | {e.get('dedication', '')} | {e.get('building_material', '')} |"
        for e in entries_sorted
    ]
    return "\n".join([header, *rows])


def _section(title: str, data: Dict[str, Any]) -> str:
    parts = [f"### {title}", ""]
    if data.get("page_number"):
        parts.append(f"**Page**: {data.get('page_number')}")
    if data.get("deanery"):
        parts.append(f"**Deanery**: {data.get('deanery')}")
    parts.append("")
    parts.append(_entries_to_table(data.get("entries", [])))
    return "\n".join(parts)


def format_comparison_md(gt_json: Dict[str, Any], lmv3_json: Dict[str, Any], raw_llm_json: Dict[str, Any], parsed_llm_json: Dict[str, Any]) -> str:
    """Return a markdown string comparing ground-truth vs prediction."""
    return _section("Ground Truth", gt_json) + "\n\n" + _section("LMV3", lmv3_json) + "\n\n" + _section("Raw LLM", raw_llm_json) + "\n\n" + _section("Parsed LLM", parsed_llm_json) 