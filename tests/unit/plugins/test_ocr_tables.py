from uuid import UUID, uuid4

from grafy_plugin_ocr.mistral import (
    MistralOcrPagePayload,
    MistralOcrResponsePayload,
    MistralOcrTablePayload,
)
from grafy_plugin_ocr.tables import (
    extract_table_fragments,
    is_markdown_separator,
    markdown_table_rows,
    markdown_tables_from_page,
    normalize_rows,
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

    assert markdown_table_rows("| A | B |\n| --- | --- |\n| one | two | extra |") == [
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
