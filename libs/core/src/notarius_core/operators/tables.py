import csv
import re
from collections.abc import Sequence
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import Annotated, override
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from notarius_core.artifacts import (
    TABLE_CSV_BUNDLE,
    TABLE_FRAGMENT,
    TABLE_PAGE,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
)
from notarius_core.plugins import Plugin


TABLES = Plugin(
    slug="builtin.tables",
    title="Tables",
)
TABLES.register_artifact_type(TABLE_FRAGMENT)
TABLES.register_artifact_type(TABLE_PAGE)
TABLES.register_artifact_type(TABLE_CSV_BUNDLE)


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


class MergeTablePagesInput(NodeInput):
    fragments: Annotated[
        list[TableFragment],
        InPort(TABLE_FRAGMENT),
        Field(description="Ordered table fragments to merge by source page."),
    ]


class MergeTablePagesOutput(NodeOutput):
    pages: Annotated[
        list[TablePage],
        OutPort(TABLE_PAGE),
        Field(description="Merged table content for each source page."),
    ]


@TABLES.node(
    operator_id="table.page.merge",
    version=1,
    title="Merge table pages",
)
class MergeTablePagesNode(Node[NoConfig, MergeTablePagesInput, MergeTablePagesOutput]):
    """Merges ordered table fragments that belong to the same source page."""

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
        Field(description="Original table fragments for per-table CSV files."),
    ]
    pages: Annotated[
        list[TablePage],
        InPort(TABLE_PAGE),
        Field(description="Merged pages for page and combined CSV files."),
    ]


class BuildTableCsvBundleOutput(NodeOutput):
    bundle: Annotated[
        TableCsvBundle,
        OutPort(TABLE_CSV_BUNDLE),
        Field(description="Generated CSV files bundled as one artifact."),
    ]


@TABLES.node(
    operator_id="table.csv.export",
    version=1,
    title="Build CSV bundle",
)
class BuildTableCsvBundleNode(
    Node[NoConfig, BuildTableCsvBundleInput, BuildTableCsvBundleOutput]
):
    """Builds per-table, per-page, and combined CSV files as one bundle."""

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
