#!/usr/bin/env python3
"""
Script to update HuggingFace dataset with generated source (Latin) data.

Usage:
    python scripts/update_hf_with_source.py <path_to_json>
    python scripts/update_hf_with_source.py data/outputs/source_generation/20251211_165730_wloclawek_1872_source.json
    python scripts/update_hf_with_source.py <path> --dry-run  # Preview without pushing
"""

import asyncio
from pathlib import Path
from typing import Annotated, Optional

import typer
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from notarius.application.use_cases.export.update_hf_with_source import UpdateHFWithSourceRequest
from notarius.shared.logger import setup_logging, get_logger

setup_logging()

logger = get_logger(__name__)

envs = load_dotenv()
if not envs:
    logger.warning("No environment variables loaded from .env file")

# Default HuggingFace dataset path
DEFAULT_HF_REPO = "artpods56/KUL_IDUB_EcclessiaSchematisms"

console = Console()

app = typer.Typer(
    name="update-hf-source",
    help="Update HuggingFace dataset with generated source data",
    add_completion=False,
)


async def _run_update(
    source_json: Path,
    hf_repo: str,
    split: str,
    dry_run: bool,
    show_updated: bool,
    show_unchanged: bool,
    commit_message: Optional[str],
) -> None:
    """Execute the update workflow."""

    if not source_json.exists():
        logger.error("Source JSON file not found", path=str(source_json))
        raise typer.BadParameter(f"Source JSON file not found: {source_json}")

    logger.info(
        "Loading HuggingFace dataset",
        repo=hf_repo,
        split=split,
    )

    # Load the HuggingFace dataset
    dataset = load_dataset(hf_repo, split=split)

    if not isinstance(dataset, Dataset):
        raise ValueError(
            "Only standard HuggingFace dataset object format is supported",
        )

    logger.info(
        "Dataset loaded",
        total_samples=len(dataset),
        columns=dataset.column_names,
    )

    # Create and execute the use case
    use_case = UpdateHFDatasetWithSource()

    request = UpdateHFWithSourceRequest(
        source_json_path=source_json,
        hf_dataset=dataset,
        hf_repo_path=hf_repo,
        push_to_hub=not dry_run,
        commit_message=commit_message,
    )

    logger.info(
        "Executing update",
        source_json=str(source_json),
        dry_run=dry_run,
    )

    response = await use_case.execute(request)

    stats_table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Count", justify="right")
    stats_table.add_column("Description")

    stats_table.add_row(
        "Total Samples",
        str(response.stats["total_samples"]),
        "in dataset",
        style="bold",
    )
    stats_table.add_row("", "", "")  # spacer

    stats_table.add_row(
        "Matched",
        str(response.stats["samples_matched"]),
        "key found in source",
        style="cyan",
    )
    stats_table.add_row(
        "  Updated",
        str(response.stats["samples_updated"]),
        "content changed",
        style="green",
    )
    stats_table.add_row(
        "  Unchanged",
        str(response.stats["samples_unchanged"]),
        "content identical",
        style="yellow",
    )
    stats_table.add_row(
        "Not Found",
        str(response.stats["samples_not_found"]),
        "key not in source",
        style="dim",
    )

    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column("Key", style="dim")
    info_table.add_column("Value")

    info_table.add_row("Source JSON", str(source_json.name))
    info_table.add_row("HuggingFace Repo", hf_repo)
    info_table.add_row("Split", split)

    if response.pushed_to_hub:
        status = Text("Pushed to Hub", style="bold green")
    else:
        status = Text("Dry Run (not pushed)", style="bold yellow")

    info_table.add_row("Status", status)

    console.print(
        Panel(info_table, title="[bold]Configuration[/bold]", border_style="blue")
    )
    console.print(
        Panel(stats_table, title="[bold]Update Results[/bold]", border_style="green")
    )

    updated_keys = response.stats.get("updated_keys", None)
    if show_updated and updated_keys:
        id_table = Table(show_header=False, box=None, padding=(0, 1))
        id_table.add_column("Sample ID", style="green")
        for sample_id in sorted(updated_keys):
            id_table.add_row(sample_id)
        console.print(
            Panel(
                id_table,
                title=f"[bold green]Updated Samples ({len(updated_keys)})[/bold green]",
                border_style="green",
            )
        )

    unchanged_keys = response.stats.get("unchanged_keys", None)
    if show_unchanged and unchanged_keys:
        id_table = Table(show_header=False, box=None, padding=(0, 1))
        id_table.add_column("Sample ID", style="yellow")
        for sample_id in sorted(unchanged_keys):
            id_table.add_row(sample_id)
        console.print(
            Panel(
                id_table,
                title=f"[bold yellow]Unchanged Samples ({len(unchanged_keys)})[/bold yellow]",
                border_style="yellow",
            )
        )

    if dry_run:
        console.print()
        console.print(
            "[yellow]Dry run mode:[/yellow] No changes were pushed to HuggingFace Hub."
        )
        console.print("Run without [bold]--dry-run[/bold] to push changes.")


@app.command()
def main(
    source_json: Annotated[
        Path,
        typer.Argument(
            help="Path to the generated source JSON file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    hf_repo: Annotated[
        str,
        typer.Option(
            "--hf-repo",
            "-r",
            help="HuggingFace repository path",
        ),
    ] = DEFAULT_HF_REPO,
    split: Annotated[
        str,
        typer.Option(
            "--split",
            "-s",
            help="Dataset split to update",
        ),
    ] = "train",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-d",
            help="Preview changes without pushing to HuggingFace Hub",
        ),
    ] = False,
    show_updated: Annotated[
        bool,
        typer.Option(
            "--show-updated",
            "-u",
            help="Print list of updated sample IDs (content changed)",
        ),
    ] = False,
    show_unchanged: Annotated[
        bool,
        typer.Option(
            "--show-unchanged",
            help="Print list of unchanged sample IDs (content identical)",
        ),
    ] = False,
    commit_message: Annotated[
        Optional[str],
        typer.Option(
            "--commit-message",
            "-m",
            help="Commit message for pushed changes",
        ),
    ] = None,
) -> None:
    """
    Update HuggingFace dataset with generated source (Latin) data.

    Loads source records from a JSON file and updates the HuggingFace dataset
    by matching samples using schematism_name + filename as lookup key.

    Statistics are tracked for:
    - Samples matched (key found in source data)
    - Samples updated (content actually changed)
    - Samples unchanged (key matched but content identical)
    - Samples not found (key not in source data)
    """
    asyncio.run(
        _run_update(
            source_json=source_json,
            hf_repo=hf_repo,
            split=split,
            dry_run=dry_run,
            show_updated=show_updated,
            show_unchanged=show_unchanged,
            commit_message=commit_message,
        )
    )


if __name__ == "__main__":
    app()
