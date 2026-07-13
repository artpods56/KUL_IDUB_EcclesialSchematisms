from typing import cast

from fastapi.testclient import TestClient

from notarius_api.main import create_app
from notarius_api.schemas.workbench import NodeRegistryResponse, RunResponse
from notarius_api.services.workbench import WorkbenchService


def test_application_lifespan_builds_and_releases_workbench_service() -> None:
    application = create_app()
    assert not hasattr(application.state, "workbench")

    with TestClient(application) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert isinstance(application.state.workbench, WorkbenchService)
        assert application.state.workbench.plugin_registry.plugins

    assert not hasattr(application.state, "workbench")


def test_node_registry_exposes_builtin_plugins_and_runtime_contracts(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.get("/v1/nodes")

    assert response.status_code == 200
    registry = NodeRegistryResponse.model_validate(response.json())
    assert [(plugin.slug, plugin.title) for plugin in registry.plugins] == [
        ("builtin.sources", "Sources"),
        ("builtin.arithmetic", "Arithmetic"),
        ("builtin.text", "Text"),
        ("builtin.tables", "Tables"),
    ]
    nodes = {node.operator_id: node for node in registry.nodes}
    assert set(nodes) == {
        "source.local_upload.images",
        "source.image_sequence.merge",
        "arithmetic.number",
        "arithmetic.integer_sequence",
        "arithmetic.add_subtract",
        "arithmetic.multiply",
        "arithmetic.sum",
        "text.input",
        "text.split",
        "text.replace",
        "text.join",
        "table.page.merge",
        "table.csv.export",
    }
    assert {
        (artifact_type.key.id, artifact_type.key.schema_version)
        for artifact_type in registry.artifact_types
    } == {
        ("source.page_image", 1),
        ("scalar.integer", 1),
        ("arithmetic.result", 1),
        ("scalar.text", 1),
        ("table.fragment", 1),
        ("table.page", 1),
        ("tabular.csv_bundle", 1),
    }

    source = nodes["source.local_upload.images"]
    assert source.plugin_slug == "builtin.sources"
    assert source.description == (
        "Imports staged local images as an ordered artifact sequence."
    )
    assert source.outputs[0].artifact_type.id == "source.page_image"
    assert source.outputs[0].shape == "many"
    assert source.outputs[0].description == (
        "Ordered images imported from the selected source."
    )

    merge_tables = nodes["table.page.merge"]
    assert merge_tables.plugin_slug == "builtin.tables"
    assert merge_tables.inputs[0].artifact_type.id == "table.fragment"
    assert merge_tables.inputs[0].shape == "many"
    assert merge_tables.outputs[0].artifact_type.id == "table.page"
    export_tables = nodes["table.csv.export"]
    assert [port.artifact_type.id for port in export_tables.inputs] == [
        "table.fragment",
        "table.page",
    ]
    assert export_tables.outputs[0].artifact_type.id == "tabular.csv_bundle"

    text_input_properties = cast(
        dict[str, object],
        nodes["text.input"].config_schema["properties"],
    )
    assert text_input_properties["text"] == {
        "description": "Multiline text emitted by the node.",
        "format": "textarea",
        "title": "Text",
        "type": "string",
    }

    add_subtract = nodes["arithmetic.add_subtract"]
    assert add_subtract.inputs[0].title == "Left"
    assert add_subtract.inputs[0].description == "Left-hand integer operand."


def test_run_accepts_empty_graph(builtin_client: TestClient) -> None:
    response = builtin_client.post("/v1/runs", json={"nodes": [], "edges": []})

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    assert result.node_runs == []


def test_source_nodes_materialize_and_merge_sample_images(
    builtin_client: TestClient,
) -> None:
    sample_response = builtin_client.post("/v1/samples", json={"count": 2})
    assert sample_response.status_code == 200
    selection = [
        {**item, "order_index": index}
        for index, item in enumerate(sample_response.json())
    ]

    run_response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "source.local_upload.images",
                    "operator_version": 1,
                    "config": {
                        "connector_id": "local_upload",
                        "selection": selection,
                    },
                },
                {
                    "id": "merge",
                    "operator_id": "source.image_sequence.merge",
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "pages",
                    "to_node": "merge",
                    "to_port": "sequences",
                }
            ],
        },
    )

    assert run_response.status_code == 200
    result = RunResponse.model_validate(run_response.json())
    assert result.status == "succeeded"
    source_run, merge_run = result.node_runs
    assert source_run.status == "succeeded"
    assert merge_run.status == "succeeded"
    assert len(source_run.outputs[0].artifacts) == 2
    assert [artifact.artifact_id for artifact in merge_run.outputs[0].artifacts] == [
        artifact.artifact_id for artifact in source_run.outputs[0].artifacts
    ]

    content_response = builtin_client.get(
        f"/v1/artifacts/{source_run.outputs[0].artifacts[0].artifact_id}/content"
    )
    assert content_response.status_code == 200
    assert content_response.headers["content-type"] == "image/png"
    assert content_response.content.startswith(b"\x89PNG")
