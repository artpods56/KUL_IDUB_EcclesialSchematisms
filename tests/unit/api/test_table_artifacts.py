import asyncio
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from notarius_api.services.composition import WorkbenchComponents
from notarius_api.v1.routes.artifacts.services import ArtifactService
from notarius_api.v1.routes.executions.services import RunResultPresenter
from notarius_core.artifacts import ArtifactObject, InMemoryUnitOfWork
from notarius_core.nodes import NodeExecutionContext
from notarius_core.operators.tables import (
    Table,
    TableArtifactWriter,
    TableColumn,
    TableValueType,
)
from notarius_core.runtime.materialization import MaterializationProvenance
from notarius_core.runtime.persistence import ArtifactWriteContext
from notarius_storage import LocalFileObjectStore


def test_table_page_bounds_cell_previews_and_full_download(
    table_artifact_client: tuple[
        TestClient,
        TableArtifactWriter,
        WorkbenchComponents,
    ],
) -> None:
    client, writer, components = table_artifact_client
    long_geometry = "MULTIPOLYGON (((" + "1 2, " * 300 + "1 2)))"
    table = Table(
        columns=[
            TableColumn(id="id", title="ID", value_type=TableValueType.INTEGER),
            TableColumn(
                id="geometry/wkt",
                title="Geometry",
                value_type=TableValueType.TEXT,
            ),
            TableColumn(
                id="large_id",
                title="Large ID",
                value_type=TableValueType.INTEGER,
            ),
            TableColumn(
                id="metadata",
                title="Metadata",
                value_type=TableValueType.JSON,
            ),
        ],
        rows=[
            {
                "id": index,
                "geometry/wkt": f"{long_geometry}-{index}",
                "large_id": 2**60 + index,
                "metadata": {"index": index, "tags": ["table", "preview"]},
            }
            for index in range(120)
        ],
    )
    ref = asyncio.run(
        writer.write(
            table,
            ArtifactWriteContext(
                node_context=NodeExecutionContext(node_id="table"),
                provenance=MaterializationProvenance(refs_by_input={}),
            ),
        )
    )

    page_response = client.get(
        f"/v1/artifacts/{ref.artifact_id}/table/page",
        params={"offset": 95, "limit": 10, "max_cell_characters": 32},
    )

    assert page_response.status_code == 200
    page = page_response.json()
    assert page["offset"] == 95
    assert page["limit"] == 10
    assert page["total_rows"] == 120
    assert len(page["rows"]) == 10
    assert page["rows"][0]["id"] == {
        "display": "95",
        "truncated": False,
        "original_length": None,
    }
    geometry_preview = page["rows"][0]["geometry/wkt"]
    assert geometry_preview["truncated"] is True
    assert len(geometry_preview["display"]) == 33
    geometry = table.rows[95]["geometry/wkt"]
    assert isinstance(geometry, str)
    assert geometry_preview["original_length"] == len(geometry)

    cell_response = client.get(
        f"/v1/artifacts/{ref.artifact_id}/table/cell",
        params={"row_index": 95, "column_id": "geometry/wkt"},
    )
    assert cell_response.status_code == 200
    assert cell_response.json() == {
        "row_index": 95,
        "column_id": "geometry/wkt",
        "value": table.rows[95]["geometry/wkt"],
        "encoding": "native",
    }

    large_integer_response = client.get(
        f"/v1/artifacts/{ref.artifact_id}/table/cell",
        params={"row_index": 95, "column_id": "large_id"},
    )
    assert large_integer_response.json() == {
        "row_index": 95,
        "column_id": "large_id",
        "value": str(2**60 + 95),
        "encoding": "integer",
    }
    nested_response = client.get(
        f"/v1/artifacts/{ref.artifact_id}/table/cell",
        params={"row_index": 95, "column_id": "metadata"},
    )
    assert nested_response.json() == {
        "row_index": 95,
        "column_id": "metadata",
        "value": '{"index": 95, "tags": ["table", "preview"]}',
        "encoding": "json",
    }
    assert (
        client.get(
            f"/v1/artifacts/{ref.artifact_id}/table/cell",
            params={"row_index": 999, "column_id": "id"},
        ).status_code
        == 400
    )
    assert (
        client.get(
            f"/v1/artifacts/{ref.artifact_id}/table/cell",
            params={"row_index": 0, "column_id": "missing"},
        ).status_code
        == 400
    )

    column_page_response = client.get(
        f"/v1/artifacts/{ref.artifact_id}/table/page",
        params={"column_offset": 2, "column_limit": 1},
    )
    assert column_page_response.status_code == 200
    column_page = column_page_response.json()
    assert [column["id"] for column in column_page["columns"]] == ["large_id"]
    assert set(column_page["rows"][0]) == {"large_id"}
    assert column_page["column_offset"] == 2
    assert column_page["column_limit"] == 1
    assert column_page["total_columns"] == 4

    selected_columns_response = client.get(
        f"/v1/artifacts/{ref.artifact_id}/table/page",
        params=[
            ("column_ids", "metadata"),
            ("column_ids", "id"),
        ],
    )
    assert selected_columns_response.status_code == 200
    selected_columns_page = selected_columns_response.json()
    assert [
        column["id"] for column in selected_columns_page["columns"]
    ] == ["metadata", "id"]
    assert list(selected_columns_page["rows"][0]) == ["metadata", "id"]
    assert selected_columns_page["column_offset"] == 0
    assert selected_columns_page["column_limit"] == 2
    assert selected_columns_page["total_columns"] == 4
    assert (
        client.get(
            f"/v1/artifacts/{ref.artifact_id}/table/page",
            params={"column_ids": "missing"},
        ).status_code
        == 400
    )

    schema_response = client.get(
        f"/v1/artifacts/{ref.artifact_id}/table/schema"
    )
    assert schema_response.status_code == 200
    assert schema_response.json()["total_rows"] == 120
    assert [
        column["id"]
        for column in schema_response.json()["columns"]
    ] == ["id", "geometry/wkt", "large_id", "metadata"]

    summary = asyncio.run(components.presenter.artifact_summary(ref))
    assert summary.byte_size is not None

    content_response = client.get(f"/v1/artifacts/{ref.artifact_id}/content")
    assert content_response.status_code == 200
    assert Table.model_validate(content_response.json()) == table
    assert len(content_response.content) == summary.byte_size
    assert sha256(content_response.content).hexdigest() == ref.content_hash

    assert summary.text is None
    assert summary.metadata["row_count"] == 120
    assert summary.metadata["column_count"] == 4


def test_table_page_rejects_non_table_and_invalid_limits(
    table_artifact_client: tuple[
        TestClient,
        TableArtifactWriter,
        WorkbenchComponents,
    ],
) -> None:
    client, _, _ = table_artifact_client

    run_response = client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "text",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "not a table"},
                }
            ]
        },
    )
    assert run_response.status_code == 200
    text_artifact_id = run_response.json()["node_runs"][0]["outputs"][0]["value"][
        "artifact_id"
    ]
    assert client.get(f"/v1/artifacts/{text_artifact_id}/table/page").status_code == 400

    assert (
        client.get(
            "/v1/artifacts/00000000-0000-0000-0000-000000000000/table/page"
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/v1/artifacts/00000000-0000-0000-0000-000000000000/table/page",
            params={"limit": 0},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/v1/artifacts/00000000-0000-0000-0000-000000000000/table/page",
            params={"limit": 100, "column_limit": 100, "max_cell_characters": 2_000},
        ).status_code
        == 400
    )


def test_table_query_filters_composite_keys_and_preserves_source_rows(
    table_artifact_client: tuple[
        TestClient,
        TableArtifactWriter,
        WorkbenchComponents,
    ],
) -> None:
    client, writer, _ = table_artifact_client
    table = Table(
        columns=[
            TableColumn(
                id="place",
                title="Place",
                value_type=TableValueType.TEXT,
            ),
            TableColumn(
                id="district",
                title="District",
                value_type=TableValueType.TEXT,
            ),
            TableColumn(
                id="status",
                title="Status",
                value_type=TableValueType.TEXT,
            ),
        ],
        rows=[
            {
                "place": "Belynichi",
                "district": "Mohilev",
                "status": "accepted",
            },
            {
                "place": "Belynichi",
                "district": "Orsha",
                "status": "accepted",
            },
            {
                "place": "Kniazhitsy",
                "district": "Mohilev",
                "status": "review",
            },
            {
                "place": "Gomel",
                "district": "Gomel",
                "status": "review",
            },
        ],
    )
    ref = asyncio.run(
        writer.write(
            table,
            ArtifactWriteContext(
                node_context=NodeExecutionContext(node_id="table-query"),
                provenance=MaterializationProvenance(refs_by_input={}),
            ),
        )
    )

    response = client.post(
        f"/v1/artifacts/{ref.artifact_id}/table/query",
        json={
            "filter_groups": [
                {
                    "rows": [
                        {"values": {"place": "Belynichi"}},
                        {"values": {"place": "Kniazhitsy"}},
                    ]
                },
                {"rows": [{"values": {"district": "Mohilev"}}]},
            ],
            "highlight_groups": [
                {"rows": [{"values": {"status": "review"}}]}
            ],
            "offset": 0,
            "limit": 50,
            "column_offset": 0,
            "column_limit": 25,
            "max_cell_characters": 256,
        },
    )

    assert response.status_code == 200
    page = response.json()
    assert page["total_rows"] == 2
    assert page["row_indices"] == [0, 2]
    assert page["highlighted_row_indices"] == [2]
    assert [
        row["place"]["display"]
        for row in page["rows"]
    ] == ["Belynichi", "Kniazhitsy"]

    missing_field = client.post(
        f"/v1/artifacts/{ref.artifact_id}/table/query",
        json={
            "filter_groups": [
                {"rows": [{"values": {"missing": "Belynichi"}}]}
            ]
        },
    )
    assert missing_field.status_code == 400
    assert "missing" in missing_field.json()["detail"]


@pytest.mark.asyncio
async def test_artifact_summaries_never_embed_unbounded_or_table_json(
    tmp_path: Path,
) -> None:
    unit_of_work = InMemoryUnitOfWork()
    unknown_size = ArtifactObject(
        artifact_type="sql.result",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"table": "unbounded"},
    )
    large = ArtifactObject(
        artifact_type="sql.result",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"table": "x" * 70_000},
        byte_size=70_000,
    )
    legacy_table = Table(
        columns=[
            TableColumn(id="value", title="Value", value_type=TableValueType.TEXT)
        ],
        rows=[{"value": "small but paged"}],
    )
    table_artifact = ArtifactObject(
        artifact_type="table.data",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload=legacy_table.model_dump(mode="json"),
        byte_size=10,
    )
    small = ArtifactObject(
        artifact_type="scalar.text",
        schema_version=1,
        content_type="application/json",
        storage_backend="inline",
        inline_payload={"text": "bounded"},
        byte_size=18,
    )
    async with unit_of_work as uow:
        for artifact in (unknown_size, large, table_artifact, small):
            await uow.artifacts.add(artifact)
        await uow.commit()
    presenter = RunResultPresenter(
        ArtifactService(
            unit_of_work,
            LocalFileObjectStore(tmp_path / "objects"),
        )
    )

    assert (await presenter.artifact_summary(unknown_size.ref())).text is None
    assert (await presenter.artifact_summary(large.ref())).text is None
    assert (await presenter.artifact_summary(table_artifact.ref())).text is None
    assert (await presenter.artifact_summary(small.ref())).text == "bounded"
