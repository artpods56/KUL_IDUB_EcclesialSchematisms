import re
from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from grafy_core.artifacts import (
    NoConfig,
    NodeInput,
    NodeOutput,
)
from grafy_core.nodes import InPort, OutPort

from grafy_plugin_ocr.artifacts import MISTRAL_OCR_RESPONSE, TABLE_FRAGMENT
from grafy_plugin_ocr.declaration import OCR
from grafy_plugin_ocr.mistral import MistralOcrResponsePayload


class TableFragment(BaseModel):
    source_image_artifact_id: UUID
    source_image: str
    provider_page_index: int = Field(ge=0)
    provider_table_index: int = Field(ge=1)
    rows: list[list[str]] = Field(min_length=1)


def split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in stripped:
        if escaped:
            current.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)

    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def is_markdown_separator(line: str) -> bool:
    cells = split_markdown_row(line)
    if not cells:
        return False
    return all(re.fullmatch(r":?-{2,}:?", cell.strip()) for cell in cells)


def normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows

    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def markdown_table_rows(markdown: str) -> list[list[str]]:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    if is_markdown_separator(lines[1]):
        body = [lines[0], *lines[2:]]
    else:
        body = lines

    rows = [split_markdown_row(line) for line in body if "|" in line]
    return normalize_rows(rows)


def markdown_tables_from_page(markdown: str) -> list[list[list[str]]]:
    lines = markdown.splitlines()
    tables: list[list[list[str]]] = []
    index = 0
    while index < len(lines) - 1:
        line = lines[index].strip()
        next_line = lines[index + 1].strip()
        if "|" not in line or not is_markdown_separator(next_line):
            index += 1
            continue

        block = [line, next_line]
        index += 2
        while index < len(lines) and "|" in lines[index]:
            block.append(lines[index].strip())
            index += 1

        rows = markdown_table_rows("\n".join(block))
        if rows:
            tables.append(rows)

    return tables


def extract_table_fragments(
    responses: Sequence[MistralOcrResponsePayload],
) -> list[TableFragment]:
    fragments: list[TableFragment] = []
    for response in responses:
        for page in response.pages:
            extracted_count = len(fragments)
            if page.tables:
                for table_index, table in enumerate(page.tables, start=1):
                    rows = markdown_table_rows(table.content)
                    if rows:
                        fragments.append(
                            TableFragment(
                                source_image_artifact_id=(
                                    response.source_image_artifact_id
                                ),
                                source_image=response.source_image,
                                provider_page_index=page.index,
                                provider_table_index=table_index,
                                rows=rows,
                            )
                        )
                if len(fragments) > extracted_count:
                    continue

            for table_index, rows in enumerate(
                markdown_tables_from_page(page.markdown), start=1
            ):
                fragments.append(
                    TableFragment(
                        source_image_artifact_id=response.source_image_artifact_id,
                        source_image=response.source_image,
                        provider_page_index=page.index,
                        provider_table_index=table_index,
                        rows=rows,
                    )
                )

    return fragments


class ExtractTableFragmentsInput(NodeInput):
    responses: Annotated[
        list[MistralOcrResponsePayload],
        InPort(MISTRAL_OCR_RESPONSE),
        Field(description="Mistral OCR responses containing table Markdown."),
    ]


class ExtractTableFragmentsOutput(NodeOutput):
    fragments: Annotated[
        list[TableFragment],
        OutPort(TABLE_FRAGMENT),
        Field(description="Table fragments extracted from the OCR responses."),
    ]


@OCR.function_node(
    operator_id="table.markdown.extract",
    version=1,
    title="Extract Markdown Tables",
)
async def extract_markdown_tables(
    _config: NoConfig,
    inputs: ExtractTableFragmentsInput,
) -> ExtractTableFragmentsOutput:
    """Extracts provider tables with page-Markdown fallback."""
    return ExtractTableFragmentsOutput(
        fragments=extract_table_fragments(inputs.responses)
    )
