from typing import Annotated

from pydantic import Field

from grafy_core.artifacts import NoConfig, NodeInput, NodeOutput
from grafy_core.artifact_contracts import TEXT_VALUE, TextValue
from grafy_core.nodes import InPort, OutPort
from grafy_core.table_contracts import TABLE_DATA, Table
from grafy_core.plugins import NodeCachePolicy

from grafy_plugin.artifacts import TABLE_SUMMARY
from grafy_plugin.declaration import PLUGIN
from grafy_plugin.models import TableSummary


class SummarizeTableInput(NodeInput):
    table: Annotated[
        Table,
        InPort(TABLE_DATA),
        Field(description="Builtin table.data@1; this Plugin does not own Table."),
    ]


class SummarizeTableOutput(NodeOutput):
    summary: Annotated[
        TableSummary,
        OutPort(TABLE_SUMMARY),
        Field(description="Plugin-owned notes.table_summary@1."),
    ]


class RenderSummaryInput(NodeInput):
    summary: Annotated[
        TableSummary,
        InPort(TABLE_SUMMARY),
        Field(description="The family contract shared by both nodes."),
    ]


class RenderSummaryOutput(NodeOutput):
    text: Annotated[
        TextValue,
        OutPort(TEXT_VALUE),
        Field(description="Builtin scalar.text@1."),
    ]


@PLUGIN.function_node(
    operator_id="notes.table.summarize",
    version=1,
    title="Summarize table",
    cache_policy=NodeCachePolicy.EXACT,
)
async def summarize_table(
    _config: NoConfig,
    inputs: SummarizeTableInput,
) -> SummarizeTableOutput:
    """Count rows and columns on a core Table."""

    table = inputs.table
    return SummarizeTableOutput(
        summary=TableSummary(
            row_count=len(table.rows),
            column_count=len(table.columns),
            column_ids=tuple(column.id for column in table.columns),
        )
    )


@PLUGIN.function_node(
    operator_id="notes.summary.render",
    version=1,
    title="Render table summary",
    cache_policy=NodeCachePolicy.EXACT,
)
async def render_summary(
    _config: NoConfig,
    inputs: RenderSummaryInput,
) -> RenderSummaryOutput:
    """Render the Plugin-owned summary as core text."""

    summary = inputs.summary
    columns = ", ".join(summary.column_ids) if summary.column_ids else "(none)"
    return RenderSummaryOutput(
        text=TextValue(
            value=(
                f"{summary.row_count} rows, {summary.column_count} columns: {columns}"
            )
        )
    )
