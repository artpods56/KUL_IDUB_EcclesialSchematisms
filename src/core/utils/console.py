from __future__ import annotations

from datetime import datetime
from typing import Mapping, Optional, Sequence, Tuple, Type

import os
import platform

from rich.box import ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from core.models.base import ConfigurableModel


def _short_path(path: Optional[str], max_len: int = 48) -> str:
    if not path:
        return "-"
    if len(path) <= max_len:
        return path
    head, tail = path[: max_len // 2 - 1], path[-(max_len // 2 - 2) :]
    return f"{head}…{tail}"


def render_run_header(
    *,
    run_id: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_summary: Optional[str] = None,
    model_configs: Optional[Mapping[Type[ConfigurableModel], object]] = None,
    cache_counts: Optional[Mapping[str, int]] = None,
    phases: Optional[Sequence[str]] = None,
    project_root: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    """Render a compact header box with basic run information.

    This function is side-effect-only: it prints a Rich `Panel` to the provided
    console or a global Console if none is given. Inputs are optional so callers
    can pass what they have without complex coupling.
    """

    console = console or Console()

    # Header title line
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys = f"{platform.system()} {platform.machine()}"
    title = f"AI_Osrodek Pipeline • {run_id or '-'} • {now_str} • {sys}"

    # Left table: data + phases
    left = Table.grid(padding=(0, 1))
    left.add_column(justify="right", style="dim")
    left.add_column(no_wrap=True)
    left.add_row("Dataset", dataset_name or "-")
    left.add_row("Summary", dataset_summary or "-")
    left.add_row("Phases", ", ".join(phases or []) or "-")

    # Right table: models + caches
    models_table = Table.grid(padding=(0, 1))
    models_table.add_column(justify="right", style="dim")
    models_table.add_column(no_wrap=True)

    if model_configs:
        for cls, cfg in model_configs.items():
            # Try to extract a compact model identifier from common config fields
            model_label = cls.__name__
            model_value = "-"
            try:
                # Heuristics for known models
                # LLM: config.interfaces.<api_type>.model
                interfaces = getattr(cfg, "interfaces", None)
                predictor = getattr(cfg, "predictor", None)
                if interfaces and predictor:
                    api_type = predictor.get("api_type", "")
                    icfg = interfaces.get(api_type, {})
                    model_value = icfg.get("model", model_value)
                # LMv3: config.inference.checkpoint
                inference = getattr(cfg, "inference", None)
                if inference and inference.get("checkpoint"):
                    model_value = inference.get("checkpoint")
                # OCR: config.language
                if hasattr(cfg, "get") and cfg.get("language"):
                    model_value = cfg.get("language")
            except Exception:
                pass

            models_table.add_row(model_label, _short_path(str(model_value)))
    else:
        models_table.add_row("Models", "-")

    if cache_counts:
        for name, count in cache_counts.items():
            models_table.add_row(name, str(count))

    # Bottom table: paths
    bottom = Table.grid(padding=(0, 1))
    bottom.add_column(justify="right", style="dim")
    bottom.add_column()
    cwd = project_root or os.getcwd()
    bottom.add_row("Project", _short_path(cwd))

    body = Table.grid(expand=True)
    body.add_column(ratio=1)
    body.add_column(ratio=1)
    body.add_row(left, models_table)
    body.add_row(bottom, Table.grid())

    console.print(Panel(body, title=title, box=ROUNDED))


