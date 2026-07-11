from uuid import UUID, uuid4

import pytest

from notarius_core.prototype.mistral_ocr import (
    MistralOcrPagePayload,
    MistralOcrResponsePayload,
    MistralOcrTablePayload,
)
from notarius_core.prototype.tables import (
    CsvFile,
    TableCsvBundle,
    TableFragment,
    build_table_csv_bundle,
    extract_table_fragments,
    is_markdown_separator,
    markdown_table_rows,
    markdown_tables_from_page,
    merge_table_fragments,
    normalize_rows,
    safe_stem,
    split_markdown_row,
)


def response(
    *,
    source_artifact_id: UUID,
    source_image: str,
    pages: list[MistralOcrPagePayload],
) -> MistralOcrResponsePayload:
    return MistralOcrResponsePayload(
        source_image_artifact_id=source_artifact_id,
        source_image=source_image,
        sequence_index=0,
        model="mistral-ocr-latest",
        markdown="\n".join(item.markdown for item in pages),
        pages=pages,
        usage_info={},
        raw_response={},
    )


def table(content: str) -> MistralOcrTablePayload:
    return MistralOcrTablePayload(
        id="table.md",
        content=content,
    )


def page(
    *,
    index: int,
    markdown: str,
    tables: list[MistralOcrTablePayload] | None = None,
) -> MistralOcrPagePayload:
    return MistralOcrPagePayload(
        index=index,
        markdown=markdown,
        tables=tables or [],
        blocks=[],
        dimensions={},
    )


def test_markdown_row_parsing_preserves_escaped_pipes_and_normalizes_width() -> None:
    assert split_markdown_row(r"| name | left \| right | trailing\\ |") == [
        "name",
        "left | right",
        "trailing\\",
    ]
    assert is_markdown_separator("| :--- | ---: |")
    assert not is_markdown_separator("| -- | value |")
    assert normalize_rows([["a"], ["b", "c"]]) == [
        ["a", ""],
        ["b", "c"],
    ]

    assert markdown_table_rows(
        "| A | B |\n| --- | --- |\n| one | two | extra |"
    ) == [
        ["A", "B", ""],
        ["one", "two", "extra"],
    ]


def test_embedded_markdown_tables_are_found_as_separate_blocks() -> None:
    markdown = """Before
| A | B |
| --- | --- |
| one | two |

Between
| C | D |
| :--- | ---: |
| three | four |
After
"""

    assert markdown_tables_from_page(markdown) == [
        [["A", "B"], ["one", "two"]],
        [["C", "D"], ["three", "four"]],
    ]


def test_provider_tables_win_and_invalid_provider_tables_fall_back_to_page() -> None:
    artifact_id = uuid4()
    provider_table = "| Provider | Value |\n| --- | --- |\n| p | 1 |"
    embedded_table = "| Embedded | Value |\n| --- | --- |\n| e | 2 |"
    fallback_table = "| Fallback | Value |\n| --- | --- |\n| f | 3 |"

    fragments = extract_table_fragments(
        [
            response(
                source_artifact_id=artifact_id,
                source_image="scan.jpg",
                pages=[
                    page(
                        index=4,
                        markdown=embedded_table,
                        tables=[table(provider_table)],
                    ),
                    page(
                        index=8,
                        markdown=fallback_table,
                        tables=[table("not a markdown table")],
                    ),
                ],
            )
        ]
    )

    assert [fragment.model_dump() for fragment in fragments] == [
        {
            "source_image_artifact_id": artifact_id,
            "source_image": "scan.jpg",
            "provider_page_index": 4,
            "provider_table_index": 1,
            "rows": [["Provider", "Value"], ["p", "1"]],
        },
        {
            "source_image_artifact_id": artifact_id,
            "source_image": "scan.jpg",
            "provider_page_index": 8,
            "provider_table_index": 1,
            "rows": [["Fallback", "Value"], ["f", "3"]],
        },
    ]


def test_page_merge_uses_artifact_and_provider_page_as_identity() -> None:
    first_artifact = uuid4()
    second_artifact = uuid4()
    fragments = [
        TableFragment(
            source_image_artifact_id=first_artifact,
            source_image="same-name.jpg",
            provider_page_index=0,
            provider_table_index=1,
            rows=[["A"], ["one"]],
        ),
        TableFragment(
            source_image_artifact_id=first_artifact,
            source_image="same-name.jpg",
            provider_page_index=0,
            provider_table_index=2,
            rows=[["two"]],
        ),
        TableFragment(
            source_image_artifact_id=second_artifact,
            source_image="same-name.jpg",
            provider_page_index=0,
            provider_table_index=1,
            rows=[["other"]],
        ),
    ]

    pages = merge_table_fragments(fragments)

    assert len(pages) == 2
    assert pages[0].source_image_artifact_id == first_artifact
    assert pages[0].provider_table_indexes == [1, 2]
    assert pages[0].rows == [["A"], ["one"], ["two"]]
    assert pages[1].source_image_artifact_id == second_artifact
    assert pages[1].rows == [["other"]]


def test_csv_bundle_matches_script_paths_and_csv_shapes() -> None:
    first_artifact = uuid4()
    second_artifact = uuid4()
    fragments = [
        TableFragment(
            source_image_artifact_id=first_artifact,
            source_image="First page (scan).jpg",
            provider_page_index=0,
            provider_table_index=1,
            rows=[["Name", "Value"], ["A", "1"]],
        ),
        TableFragment(
            source_image_artifact_id=second_artifact,
            source_image="second.jpg",
            provider_page_index=2,
            provider_table_index=1,
            rows=[["Name", "Value"], ["B", "2"]],
        ),
    ]
    pages = merge_table_fragments(fragments)

    bundle = build_table_csv_bundle(fragments, pages)
    files = {file.path: file.content for file in bundle.files}

    assert safe_stem("First page (scan).jpg") == "First_page_scan"
    assert list(files) == [
        "fragments/First_page_scan_page0_table1.csv",
        "fragments/second_page2_table1.csv",
        "pages/First_page_scan_page0.csv",
        "pages/second_page2.csv",
        "all_fragments_long.csv",
        "all_pages_long.csv",
        "all_pages_rows.csv",
        "all_pages_combined.csv",
    ]
    assert files["fragments/First_page_scan_page0_table1.csv"] == (
        "Name,Value\r\nA,1\r\n"
    )
    assert files["pages/second_page2.csv"] == "Name,Value\r\nB,2\r\n"
    assert files["all_fragments_long.csv"] == (
        "source_image,page_index,table_index,row_index,column_index,value\r\n"
        "First page (scan).jpg,0,1,1,1,Name\r\n"
        "First page (scan).jpg,0,1,1,2,Value\r\n"
        "First page (scan).jpg,0,1,2,1,A\r\n"
        "First page (scan).jpg,0,1,2,2,1\r\n"
        "second.jpg,2,1,1,1,Name\r\n"
        "second.jpg,2,1,1,2,Value\r\n"
        "second.jpg,2,1,2,1,B\r\n"
        "second.jpg,2,1,2,2,2\r\n"
    )
    assert files["all_pages_long.csv"] == files["all_fragments_long.csv"]
    assert files["all_pages_rows.csv"] == (
        "source_image,page_index,table_index,row_index,column_1,column_2\r\n"
        "First page (scan).jpg,0,1,1,Name,Value\r\n"
        "First page (scan).jpg,0,1,2,A,1\r\n"
        "second.jpg,2,1,1,Name,Value\r\n"
        "second.jpg,2,1,2,B,2\r\n"
    )
    assert files["all_pages_combined.csv"] == (
        "_source_image,_page_index,_table_index,Name,Value\r\n"
        "First page (scan).jpg,0,1,A,1\r\n"
        "second.jpg,2,1,B,2\r\n"
    )


def test_csv_bundle_omits_combined_file_when_page_headers_differ() -> None:
    fragments = [
        TableFragment(
            source_image_artifact_id=uuid4(),
            source_image="first.jpg",
            provider_page_index=0,
            provider_table_index=1,
            rows=[["A"], ["one"]],
        ),
        TableFragment(
            source_image_artifact_id=uuid4(),
            source_image="second.jpg",
            provider_page_index=0,
            provider_table_index=1,
            rows=[["B"], ["two"]],
        ),
    ]

    bundle = build_table_csv_bundle(fragments, merge_table_fragments(fragments))

    assert "all_pages_combined.csv" not in {file.path for file in bundle.files}


def test_csv_bundle_disambiguates_colliding_source_stems() -> None:
    first_artifact_id = uuid4()
    second_artifact_id = uuid4()
    fragments = [
        TableFragment(
            source_image_artifact_id=first_artifact_id,
            source_image="same name.jpg",
            provider_page_index=0,
            provider_table_index=1,
            rows=[["A"], ["1"]],
        ),
        TableFragment(
            source_image_artifact_id=second_artifact_id,
            source_image="same@name.jpg",
            provider_page_index=0,
            provider_table_index=1,
            rows=[["A"], ["2"]],
        ),
    ]
    pages = merge_table_fragments(fragments)

    paths = [
        csv_file.path for csv_file in build_table_csv_bundle(fragments, pages).files
    ]

    assert "fragments/same_name_page0_table1.csv" in paths
    assert "fragments/same_name_2_page0_table1.csv" in paths
    assert "pages/same_name_page0.csv" in paths
    assert "pages/same_name_2_page0.csv" in paths
    assert len(paths) == len(set(paths))


def test_csv_bundle_rejects_duplicate_archive_paths() -> None:
    with pytest.raises(ValueError, match="Duplicate CSV bundle path"):
        TableCsvBundle(
            files=[
                CsvFile(path="duplicate.csv", content="A\r\n"),
                CsvFile(path="duplicate.csv", content="B\r\n"),
            ]
        )
