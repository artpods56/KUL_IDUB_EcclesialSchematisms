from io import BytesIO
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from notarius_api.main import app
from notarius_api.schemas.prototype import (
    PrototypeNodeRegistryResponse,
    PrototypeRunResponse,
)
from notarius_api.services.prototype_workbench import WorkbenchService
from notarius_api.v1.routes import prototype as prototype_routes
from notarius_core.prototype import (
    EncodedPageImage,
    MistralOcrConfig,
    MistralOcrProviderResponse,
)


class FakeMistralProvider:
    def __init__(self) -> None:
        self.images: list[EncodedPageImage] = []

    async def process(
        self,
        image: EncodedPageImage,
        config: MistralOcrConfig,
        /,
    ) -> MistralOcrProviderResponse:
        self.images.append(image)
        table = (
            "| Name | Value |\n"
            "| --- | --- |\n"
            f"| {image.filename} | {len(image.content)} |"
        )
        return MistralOcrProviderResponse.model_validate(
            {
                "model": "mistral-ocr-4-0",
                "usage_info": {"pages_processed": 1},
                "pages": [
                    {
                        "index": 0,
                        "markdown": table,
                        "tables": [
                            {
                                "id": f"table-{len(self.images)}",
                                "content": table,
                                "format": "markdown",
                            }
                        ],
                        "blocks": [
                            {
                                "type": "table",
                                "text": image.filename,
                                "bbox": [0, 0, 100, 100],
                            }
                        ],
                    }
                ],
            }
        )


def test_prototype_node_registry_exposes_runtime_materialization_contract() -> None:
    client = TestClient(app)

    response = client.get("/v1/prototype/nodes")

    assert response.status_code == 200
    registry = PrototypeNodeRegistryResponse.model_validate(response.json())
    nodes = {node.operator_id: node for node in registry.nodes}
    assert set(nodes) == {
        "source.local_upload.images",
        "source.image_sequence.merge",
        "ocr.tesseract.pages",
        "ocr.mistral.tables",
        "table.markdown.extract",
        "table.page.merge",
        "table.csv.export",
        "arithmetic.number",
        "arithmetic.add_subtract",
        "arithmetic.multiply",
    }
    assert [
        (artifact_type.key.id, artifact_type.key.schema_version, artifact_type.title)
        for artifact_type in registry.artifact_types
    ] == [
        ("source.page_image", 1, "Source page image"),
        ("ocr.page_result", 1, "OCR page result"),
        ("ocr.mistral_response", 1, "Mistral OCR response"),
        ("table.fragment", 1, "Extracted table fragment"),
        ("table.page", 1, "Merged page table"),
        ("tabular.csv_bundle", 1, "CSV export bundle"),
        ("scalar.integer", 1, "Integer value"),
        ("arithmetic.result", 1, "Arithmetic result"),
    ]

    source_output = nodes["source.local_upload.images"].outputs[0]
    assert source_output.artifact_type.id == "source.page_image"
    assert source_output.artifact_type.schema_version == 1
    assert source_output.shape == "many"
    assert source_output.variadic is False

    merge_input = nodes["source.image_sequence.merge"].inputs[0]
    assert merge_input.shape == "many"
    assert merge_input.variadic is True

    ocr_input = nodes["ocr.tesseract.pages"].inputs[0]
    assert ocr_input.artifact_type.id == "source.page_image"
    assert ocr_input.artifact_type.schema_version == 1
    assert ocr_input.shape == "many"
    assert ocr_input.variadic is False

    source_config_schema = nodes["source.local_upload.images"].config_schema
    source_config_properties = cast(
        dict[str, object],
        source_config_schema["properties"],
    )
    assert set(source_config_properties) == {"connector_id", "selection"}
    assert nodes["source.image_sequence.merge"].config_schema["properties"] == {}
    ocr_config_schema = nodes["ocr.tesseract.pages"].config_schema
    assert ocr_config_schema["properties"] == {}
    assert "x-schema-error" not in ocr_config_schema

    mistral = nodes["ocr.mistral.tables"]
    assert set(cast(dict[str, object], mistral.config_schema["properties"])) == {
        "model",
        "timeout_ms",
    }
    assert mistral.inputs[0].artifact_type.id == "source.page_image"
    assert mistral.inputs[0].shape == "many"
    assert mistral.outputs[0].artifact_type.id == "ocr.mistral_response"
    assert mistral.outputs[0].shape == "many"

    extract = nodes["table.markdown.extract"]
    assert extract.inputs[0].artifact_type.id == "ocr.mistral_response"
    assert extract.outputs[0].artifact_type.id == "table.fragment"
    merge = nodes["table.page.merge"]
    assert merge.inputs[0].artifact_type.id == "table.fragment"
    assert merge.outputs[0].artifact_type.id == "table.page"
    export = nodes["table.csv.export"]
    assert [port.artifact_type.id for port in export.inputs] == [
        "table.fragment",
        "table.page",
    ]
    assert export.outputs[0].artifact_type.id == "tabular.csv_bundle"


def test_prototype_run_accepts_empty_graph() -> None:
    client = TestClient(app)

    response = client.post("/v1/prototype/run", json={"nodes": [], "edges": []})

    assert response.status_code == 200
    result = PrototypeRunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    assert result.node_runs == []


def test_prototype_runs_mistral_table_pipeline_and_downloads_csv_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = FakeMistralProvider()
    service = WorkbenchService(
        workspace=tmp_path / "workbench",
        mistral_provider=provider,
    )
    monkeypatch.setattr(
        prototype_routes,
        "get_workbench_service",
        lambda: service,
    )
    client = TestClient(app)
    sample_response = client.post("/v1/prototype/samples", json={"count": 2})
    assert sample_response.status_code == 200
    selection = [
        {**item, "order_index": index}
        for index, item in enumerate(sample_response.json())
    ]

    run_response = client.post(
        "/v1/prototype/run",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "source.local_upload.images",
                    "config": {
                        "connector_id": "local_upload",
                        "selection": selection,
                    },
                },
                {
                    "id": "ocr",
                    "operator_id": "ocr.mistral.tables",
                    "config": {
                        "model": "mistral-ocr-latest",
                        "timeout_ms": 300000,
                    },
                },
                {
                    "id": "extract",
                    "operator_id": "table.markdown.extract",
                    "config": {},
                },
                {
                    "id": "merge",
                    "operator_id": "table.page.merge",
                    "config": {},
                },
                {
                    "id": "export",
                    "operator_id": "table.csv.export",
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "pages",
                    "to_node": "ocr",
                    "to_port": "pages",
                },
                {
                    "from_node": "ocr",
                    "from_port": "responses",
                    "to_node": "extract",
                    "to_port": "responses",
                },
                {
                    "from_node": "extract",
                    "from_port": "fragments",
                    "to_node": "merge",
                    "to_port": "fragments",
                },
                {
                    "from_node": "extract",
                    "from_port": "fragments",
                    "to_node": "export",
                    "to_port": "fragments",
                },
                {
                    "from_node": "merge",
                    "from_port": "pages",
                    "to_node": "export",
                    "to_port": "pages",
                },
            ],
        },
    )

    assert run_response.status_code == 200
    result = PrototypeRunResponse.model_validate(run_response.json())
    assert result.status == "succeeded"
    runs = {run.node_id: run for run in result.node_runs}
    assert list(runs) == ["source", "ocr", "extract", "merge", "export"]
    assert all(run.status == "succeeded" for run in runs.values())
    assert len(provider.images) == 2
    assert all(image.content.startswith(b"\x89PNG") for image in provider.images)

    ocr_artifacts = runs["ocr"].outputs[0].artifacts
    assert len(ocr_artifacts) == 2
    raw_response = client.get(
        f"/v1/prototype/artifacts/{ocr_artifacts[0].artifact_id}/content"
    )
    assert raw_response.status_code == 200
    assert raw_response.json()["model"] == "mistral-ocr-4-0"
    assert raw_response.json()["raw_response"]["pages"][0]["blocks"][0][
        "type"
    ] == "table"

    assert len(runs["extract"].outputs[0].artifacts) == 2
    assert len(runs["merge"].outputs[0].artifacts) == 2
    bundle = runs["export"].outputs[0].artifacts[0]
    assert bundle.content_type == "application/zip"
    assert bundle.metadata["file_count"] == 8
    assert bundle.metadata["combined_written"] is True

    bundle_response = client.get(
        f"/v1/prototype/artifacts/{bundle.artifact_id}/content"
    )
    assert bundle_response.status_code == 200
    assert bundle_response.headers["content-disposition"] == (
        'attachment; filename="notarius-table-csv-export.zip"'
    )
    with ZipFile(BytesIO(bundle_response.content)) as archive:
        assert archive.namelist() == [
            "fragments/sample-page-1_page0_table1.csv",
            "fragments/sample-page-2_page0_table1.csv",
            "pages/sample-page-1_page0.csv",
            "pages/sample-page-2_page0.csv",
            "all_fragments_long.csv",
            "all_pages_long.csv",
            "all_pages_rows.csv",
            "all_pages_combined.csv",
        ]
        rows_csv = archive.read("all_pages_rows.csv").decode("utf-8")
    assert "source_image,page_index,table_index,row_index" in rows_csv
    assert "sample-page-1.png" in rows_csv
