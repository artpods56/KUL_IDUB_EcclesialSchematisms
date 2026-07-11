import csv
import re
from collections.abc import Sequence
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import Annotated, ClassVar, override
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from notarius_core.prototype.artifacts import (
    MISTRAL_OCR_RESPONSE,
    TABLE_CSV_BUNDLE,
    TABLE_FRAGMENT,
    TABLE_PAGE,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.prototype.mistral_ocr import MistralOcrResponsePayload
from notarius_core.prototype.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
)


class TableFragment(BaseModel):
    source_image_artifact_id: UUID
    source_image: str
    provider_page_index: int = Field(ge=0)
    provider_table_index: int = Field(ge=1)
    rows: list[list[str]] = Field(min_length=1)


class TablePage(BaseModel):
    source_image_artifact_id: UUID
    source_image: str
    provider_page_index: int = Field(ge=0)
    provider_table_indexes: list[int] = Field(min_length=1)
    rows: list[list[str]] = Field(min_length=1)


class CsvFile(BaseModel):
    path: str
    content: str


class TableCsvBundle(BaseModel):
    files: list[CsvFile]

    @model_validator(mode="after")
    def validate_unique_paths(self) -> "TableCsvBundle":
        paths: set[str] = set()
        for csv_file in self.files:
            if csv_file.path in paths:
                raise ValueError(f"Duplicate CSV bundle path {csv_file.path!r}")
            paths.add(csv_file.path)
        return self


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


def merge_table_fragments(fragments: Sequence[TableFragment]) -> list[TablePage]:
    pages: list[TablePage] = []
    page_indexes: dict[tuple[UUID, int], int] = {}
    for fragment in fragments:
        key = (fragment.source_image_artifact_id, fragment.provider_page_index)
        page_index = page_indexes.get(key)
        if page_index is None:
            page_indexes[key] = len(pages)
            pages.append(
                TablePage(
                    source_image_artifact_id=fragment.source_image_artifact_id,
                    source_image=fragment.source_image,
                    provider_page_index=fragment.provider_page_index,
                    provider_table_indexes=[fragment.provider_table_index],
                    rows=[row.copy() for row in fragment.rows],
                )
            )
            continue

        page = pages[page_index]
        page.provider_table_indexes.append(fragment.provider_table_index)
        page.rows.extend(row.copy() for row in fragment.rows)

    return pages


def safe_stem(source_image: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(source_image).stem).strip("_")
    if stem:
        return stem
    return "image"


def build_table_csv_bundle(
    fragments: Sequence[TableFragment],
    pages: Sequence[TablePage],
) -> TableCsvBundle:
    source_stems: dict[UUID, str] = {}
    used_stems: set[str] = set()
    for table in chain(fragments, pages):
        if table.source_image_artifact_id in source_stems:
            continue
        base_stem = safe_stem(table.source_image)
        source_stem = base_stem
        suffix = 2
        while source_stem in used_stems:
            source_stem = f"{base_stem}_{suffix}"
            suffix += 1
        source_stems[table.source_image_artifact_id] = source_stem
        used_stems.add(source_stem)

    files: list[CsvFile] = []
    for fragment in fragments:
        stem = source_stems[fragment.source_image_artifact_id]
        files.append(
            CsvFile(
                path=(
                    f"fragments/{stem}_page{fragment.provider_page_index}"
                    f"_table{fragment.provider_table_index}.csv"
                ),
                content=_csv_content(fragment.rows),
            )
        )

    for page in pages:
        stem = source_stems[page.source_image_artifact_id]
        files.append(
            CsvFile(
                path=f"pages/{stem}_page{page.provider_page_index}.csv",
                content=_csv_content(page.rows),
            )
        )

    files.extend(
        [
            CsvFile(
                path="all_fragments_long.csv",
                content=_long_csv_content(fragments),
            ),
            CsvFile(
                path="all_pages_long.csv",
                content=_long_csv_content(pages),
            ),
            CsvFile(
                path="all_pages_rows.csv",
                content=_rows_csv_content(pages),
            ),
        ]
    )
    combined_content = _combined_csv_content(pages)
    if combined_content is not None:
        files.append(
            CsvFile(
                path="all_pages_combined.csv",
                content=combined_content,
            )
        )

    return TableCsvBundle(files=files)


def _csv_content(rows: Sequence[Sequence[str]]) -> str:
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerows(rows)
    return stream.getvalue()


def _long_csv_content(
    tables: Sequence[TableFragment] | Sequence[TablePage],
) -> str:
    rows = [
        [
            "source_image",
            "page_index",
            "table_index",
            "row_index",
            "column_index",
            "value",
        ]
    ]
    for table in tables:
        table_index = (
            table.provider_table_index if isinstance(table, TableFragment) else 1
        )
        for row_index, row in enumerate(table.rows, start=1):
            for column_index, value in enumerate(row, start=1):
                rows.append(
                    [
                        table.source_image,
                        str(table.provider_page_index),
                        str(table_index),
                        str(row_index),
                        str(column_index),
                        value,
                    ]
                )

    return _csv_content(rows)


def _rows_csv_content(pages: Sequence[TablePage]) -> str:
    width = 0
    for page in pages:
        for row in page.rows:
            width = max(width, len(row))

    rows = [
        [
            "source_image",
            "page_index",
            "table_index",
            "row_index",
            *[f"column_{index}" for index in range(1, width + 1)],
        ]
    ]
    for page in pages:
        for row_index, row in enumerate(page.rows, start=1):
            rows.append(
                [
                    page.source_image,
                    str(page.provider_page_index),
                    "1",
                    str(row_index),
                    *row,
                    *[""] * (width - len(row)),
                ]
            )

    return _csv_content(rows)


def _combined_csv_content(pages: Sequence[TablePage]) -> str | None:
    pages_with_headers = [page for page in pages if len(page.rows) >= 2]
    if not pages_with_headers:
        return None

    first_header = pages_with_headers[0].rows[0]
    if any(page.rows[0] != first_header for page in pages_with_headers):
        return None

    rows = [["_source_image", "_page_index", "_table_index", *first_header]]
    for page in pages_with_headers:
        for row in page.rows[1:]:
            rows.append(
                [
                    page.source_image,
                    str(page.provider_page_index),
                    "1",
                    *row,
                ]
            )

    return _csv_content(rows)


class ExtractTableFragmentsInput(NodeInput):
    responses: Annotated[
        list[MistralOcrResponsePayload],
        InPort(MISTRAL_OCR_RESPONSE),
    ]


class ExtractTableFragmentsOutput(NodeOutput):
    fragments: Annotated[
        list[TableFragment],
        OutPort(TABLE_FRAGMENT),
    ]


class ExtractTableFragmentsNode(
    Node[NoConfig, ExtractTableFragmentsInput, ExtractTableFragmentsOutput]
):
    operator_id: ClassVar[str] = "table.markdown.extract"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: ExtractTableFragmentsInput,
        /,
    ) -> ExtractTableFragmentsOutput:
        return ExtractTableFragmentsOutput(
            fragments=extract_table_fragments(inputs.responses)
        )


class MergeTablePagesInput(NodeInput):
    fragments: Annotated[
        list[TableFragment],
        InPort(TABLE_FRAGMENT),
    ]


class MergeTablePagesOutput(NodeOutput):
    pages: Annotated[
        list[TablePage],
        OutPort(TABLE_PAGE),
    ]


class MergeTablePagesNode(
    Node[NoConfig, MergeTablePagesInput, MergeTablePagesOutput]
):
    operator_id: ClassVar[str] = "table.page.merge"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: MergeTablePagesInput,
        /,
    ) -> MergeTablePagesOutput:
        return MergeTablePagesOutput(pages=merge_table_fragments(inputs.fragments))


class BuildTableCsvBundleInput(NodeInput):
    fragments: Annotated[
        list[TableFragment],
        InPort(TABLE_FRAGMENT),
    ]
    pages: Annotated[
        list[TablePage],
        InPort(TABLE_PAGE),
    ]


class BuildTableCsvBundleOutput(NodeOutput):
    bundle: Annotated[
        TableCsvBundle,
        OutPort(TABLE_CSV_BUNDLE),
    ]


class BuildTableCsvBundleNode(
    Node[NoConfig, BuildTableCsvBundleInput, BuildTableCsvBundleOutput]
):
    operator_id: ClassVar[str] = "table.csv.export"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: BuildTableCsvBundleInput,
        /,
    ) -> BuildTableCsvBundleOutput:
        return BuildTableCsvBundleOutput(
            bundle=build_table_csv_bundle(inputs.fragments, inputs.pages)
        )
