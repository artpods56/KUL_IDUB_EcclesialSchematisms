import base64
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from notarius_api.builtins import builtin_plugins
from notarius_api.main import create_app
from notarius_api.plugin_discovery import build_plugin_registry
from notarius_api.schemas.workbench import NodeRegistryResponse, RunResponse
from notarius_api.services.workbench import WorkbenchService
from notarius_core.plugins import PluginOrigin


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
        ("builtin.image", "Image"),
        ("builtin.module", "Module"),
        ("builtin.sequence", "Sequence"),
        ("builtin.arithmetic", "Arithmetic"),
        ("builtin.text", "Text"),
        ("builtin.schema", "Schema"),
        ("builtin.prompt", "Prompt"),
        ("graph.module", "Modules"),
    ]
    assert {plugin.origin for plugin in registry.plugins} == {
        PluginOrigin.BUILTIN,
        PluginOrigin.MODULE,
    }
    nodes = {node.operator_id: node for node in registry.nodes}
    assert set(nodes) == {
        "image.upload",
        "module.input",
        "module.output",
        "sequence.collect",
        "sequence.count",
        "sequence.item_at",
        "sequence.slice",
        "arithmetic.number",
        "arithmetic.integer_sequence",
        "arithmetic.add",
        "arithmetic.subtract",
        "arithmetic.multiply",
        "arithmetic.sum",
        "text.input",
        "text.split",
        "text.replace",
        "text.join",
        "schema.builder",
        "prompt.message.create",
    }
    assert {
        (artifact_type.key.id, artifact_type.key.schema_version)
        for artifact_type in registry.artifact_types
    } == {
        ("image.raster", 1),
        ("scalar.integer", 1),
        ("scalar.text", 1),
        ("json.schema", 1),
        ("prompt.message", 2),
    }
    assert [
        conversion.model_dump() for conversion in registry.artifact_conversions
    ] == [
        {
            "key": {
                "id": "builtin.scalar.integer_to_text",
                "version": 1,
            },
            "source_artifact_type": {
                "id": "scalar.integer",
                "schema_version": 1,
            },
            "target_artifact_type": {
                "id": "scalar.text",
                "schema_version": 1,
            },
            "title": "As text",
        }
    ]

    upload = nodes["image.upload"]
    assert upload.plugin_slug == "builtin.image"
    assert upload.title == "Upload images"
    assert upload.description == (
        "Imports staged image uploads as an ordered raster image sequence."
    )
    assert upload.outputs[0].name == "images"
    assert upload.outputs[0].artifact_type is not None
    assert upload.outputs[0].artifact_type.id == "image.raster"
    assert upload.outputs[0].shape == "many"
    assert upload.outputs[0].description == (
        "Ordered raster images imported from staged uploads."
    )

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

    schema_builder = nodes["schema.builder"]
    assert schema_builder.plugin_slug == "builtin.schema"
    assert schema_builder.title == "Schema Builder"
    assert schema_builder.inputs[0].name == "schemas"
    assert schema_builder.inputs[0].artifact_type is not None
    assert schema_builder.inputs[0].artifact_type.id == "json.schema"
    assert schema_builder.inputs[0].accepted_shapes == ["one"]
    assert schema_builder.inputs[0].instance_plugs is True
    assert schema_builder.inputs[0].required is False
    assert schema_builder.outputs[0].artifact_type is not None
    assert schema_builder.outputs[0].artifact_type.id == "json.schema"
    assert schema_builder.outputs[0].name == "json_schema"
    assert schema_builder.outputs[0].title == "JSON Schema"

    schema_builder_properties = cast(
        dict[str, object],
        schema_builder.config_schema["properties"],
    )
    fields_schema = cast(dict[str, object], schema_builder_properties["fields"])
    assert fields_schema["type"] == "array"

    prompt_message = nodes["prompt.message.create"]
    prompt_message_definitions = cast(
        dict[str, object],
        prompt_message.config_schema["$defs"],
    )
    role_definition = cast(
        dict[str, object],
        prompt_message_definitions["PromptMessageRole"],
    )
    assert role_definition["enum"] == ["system", "user"]
    image_input = next(port for port in prompt_message.inputs if port.name == "images")
    assert image_input.artifact_type is not None
    assert image_input.artifact_type.id == "image.raster"
    assert image_input.shape == "many"
    assert image_input.required is False

    add = nodes["arithmetic.add"]
    assert add.inputs[0].title == "Left"
    assert add.inputs[0].description == "Left-hand integer operand."

    collect = nodes["sequence.collect"]
    assert collect.inputs[0].name == "items"
    assert collect.inputs[0].shape == "one"
    assert collect.inputs[0].accepted_shapes == ["one", "many"]
    assert collect.inputs[0].instance_plugs is True
    assert collect.outputs[0].accepted_shapes == ["many"]
    assert collect.outputs[0].instance_plugs is False
    assert collect.inputs[0].artifact_type is None
    assert collect.inputs[0].artifact_type_variable == "T"
    assert collect.outputs[0].artifact_type is None
    assert collect.outputs[0].artifact_type_variable == "T"


def test_run_accepts_empty_graph(builtin_client: TestClient) -> None:
    response = builtin_client.post("/v1/runs", json={"nodes": [], "edges": []})

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    assert result.node_runs == []


@pytest.mark.asyncio
async def test_upload_from_relative_workspace_returns_opaque_upload_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    service = WorkbenchService(
        plugin_registry=build_plugin_registry(
            builtin_plugins(),
            external_plugins=(),
        ),
        workspace=Path("relative-workbench"),
    )

    item = await service.save_image_upload(
        "page.png",
        base64.b64encode(b"image-bytes").decode("ascii"),
    )

    assert "/" not in item.upload_key
    assert "\\" not in item.upload_key
    assert item.upload_key.endswith("-page.png")
    assert item.filename == "page.png"
    assert item.byte_size == len(b"image-bytes")


def test_image_upload_materializes_sample_images(
    builtin_client: TestClient,
) -> None:
    sample_response = builtin_client.post("/v1/samples", json={"count": 2})
    assert sample_response.status_code == 200
    uploads = sample_response.json()

    run_response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "upload",
                    "operator_id": "image.upload",
                    "operator_version": 1,
                    "config": {"uploads": uploads},
                },
            ],
            "edges": [],
        },
    )

    assert run_response.status_code == 200
    result = RunResponse.model_validate(run_response.json())
    assert result.status == "succeeded"
    upload_run = result.node_runs[0]
    assert upload_run.status == "succeeded"
    assert upload_run.outputs[0].port == "images"
    assert len(upload_run.outputs[0].artifacts) == 2

    content_response = builtin_client.get(
        f"/v1/artifacts/{upload_run.outputs[0].artifacts[0].artifact_id}/content"
    )
    assert content_response.status_code == 200
    assert content_response.headers["content-type"] == "image/png"
    assert content_response.content.startswith(b"\x89PNG")
