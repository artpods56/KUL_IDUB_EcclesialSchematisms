import os
from pathlib import Path

import pytest

from grafy_core.artifacts import NoConfig
from grafy_core.operators.tables import Table, TableColumn, TableValueType
from grafy_plugin.models import TableSummary
from grafy_plugin.nodes import (
    RenderSummaryInput,
    SummarizeTableInput,
    render_summary,
    summarize_table,
)


async def test_family_nodes_round_trip_a_core_table() -> None:
    table = Table(
        columns=[
            TableColumn(id="name", title="Name", value_type=TableValueType.TEXT),
        ],
        rows=[{"name": "ada"}, {"name": "grace"}],
    )
    summarized = await summarize_table(
        NoConfig(),
        SummarizeTableInput(table=table),
    )
    assert summarized.summary == TableSummary(
        row_count=2,
        column_count=1,
        column_ids=("name",),
    )
    rendered = await render_summary(
        NoConfig(),
        RenderSummaryInput(summary=summarized.summary),
    )
    assert rendered.text.value == "2 rows, 1 columns: name"


def test_publisher_environment_cannot_see_host_secrets() -> None:
    assert "GRAFY_PLUGIN_SENTINEL_SECRET" not in os.environ


@pytest.mark.skipif(
    os.environ.get("GRAFY_PLUGIN_PUBLISHING") != "1",
    reason="only meaningful inside the publisher's verification environment",
)
def test_working_copy_mutations_cannot_reach_the_freeze() -> None:
    init_path = Path("src/grafy_plugin/__init__.py")
    original = init_path.read_text(encoding="utf-8")
    init_path.write_text(
        original + "\nMUTATED_DURING_TESTS = True\n",
        encoding="utf-8",
    )
