from typing import cast

from fastapi.testclient import TestClient

from notarius_api.v1.routes.catalog.models import NodeRegistryResponse
from notarius_api.v1.routes.executions.models import RunResponse
from notarius_core.artifacts import ArtifactRefSequence
from notarius_core.operators.text import (
    MarkdownValue,
    TextInputConfig,
    TextValuePayload,
)


def _collect_run_payload() -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": "sequence-input",
                "operator_id": "text.input",
                "operator_version": 1,
                "config": {"text": "first|second"},
            },
            {
                "id": "split",
                "operator_id": "text.split",
                "operator_version": 1,
                "config": {"separator": "|"},
            },
            {
                "id": "single-input",
                "operator_id": "text.input",
                "operator_version": 1,
                "config": {"text": "third"},
            },
            {
                "id": "collect",
                "operator_id": "sequence.collect",
                "operator_version": 1,
                "config": {},
                "input_plugs": [
                    {"id": "sequence", "port": "items"},
                    {"id": "single", "port": "items"},
                ],
                "artifact_type_bindings": [
                    {
                        "variable": "T",
                        "artifact_type": {
                            "id": "scalar.text",
                            "schema_version": 1,
                        },
                    }
                ],
            },
        ],
        "edges": [
            {
                "from_node": "sequence-input",
                "from_port": "text",
                "to_node": "split",
                "to_port": "text",
            },
            {
                "from_node": "single-input",
                "from_port": "text",
                "to_node": "collect",
                "to_port": "items",
                "to_plug": "single",
            },
            {
                "from_node": "split",
                "from_port": "parts",
                "to_node": "collect",
                "to_port": "items",
                "to_plug": "sequence",
            },
        ],
    }


def _collect_node(payload: dict[str, object]) -> dict[str, object]:
    nodes = cast(list[dict[str, object]], payload["nodes"])
    return next(node for node in nodes if node["id"] == "collect")


def _collect_input_plugs(payload: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], _collect_node(payload)["input_plugs"])


def _collect_edges(payload: dict[str, object]) -> list[dict[str, object]]:
    edges = cast(list[dict[str, object]], payload["edges"])
    return [edge for edge in edges if edge["to_node"] == "collect"]


def test_registry_declares_text_artifact_and_operator_contracts(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.get("/v1/workspaces/00000000-0000-0000-0000-000000000007/nodes")

    assert response.status_code == 200
    registry = NodeRegistryResponse.model_validate(response.json())
    artifact_types = {
        artifact_type.key.id: artifact_type for artifact_type in registry.artifact_types
    }
    assert (
        artifact_types["scalar.text"].payload_schema
        == TextValuePayload.model_json_schema()
    )
    assert (
        artifact_types["text.markdown"].payload_schema
        == MarkdownValue.model_json_schema()
    )
    assert artifact_types["text.markdown"].field_projections[0].path == ["markdown"]
    assert (
        artifact_types["text.markdown"].field_projections[0].target_artifact_type.id
        == "scalar.text"
    )

    nodes = {node.operator_id: node for node in registry.nodes}
    assert nodes["text.input"].config_schema == TextInputConfig.model_json_schema()
    assert nodes["text.input"].outputs[0].name == "text"
    assert nodes["text.input"].outputs[0].shape == "one"

    assert nodes["text.as_markdown"].inputs[0].name == "text"
    assert nodes["text.as_markdown"].inputs[0].artifact_type is not None
    assert nodes["text.as_markdown"].inputs[0].artifact_type.id == "scalar.text"
    assert nodes["text.as_markdown"].outputs[0].name == "markdown"
    assert nodes["text.as_markdown"].outputs[0].artifact_type is not None
    assert nodes["text.as_markdown"].outputs[0].artifact_type.id == "text.markdown"

    assert nodes["text.split"].inputs[0].name == "text"
    assert nodes["text.split"].inputs[0].shape == "one"
    assert nodes["text.split"].outputs[0].name == "parts"
    assert nodes["text.split"].outputs[0].shape == "many"

    assert nodes["text.join"].inputs[0].name == "parts"
    assert nodes["text.join"].inputs[0].shape == "many"

    collect_input = nodes["sequence.collect"].inputs[0]
    assert collect_input.name == "items"
    assert collect_input.shape == "one"
    assert collect_input.accepted_shapes == ["one", "many"]
    assert collect_input.instance_plugs is True
    assert collect_input.artifact_type is None
    assert collect_input.artifact_type_variable == "T"


def test_as_markdown_graph_persists_exact_source(
    builtin_client: TestClient,
) -> None:
    source = "# Café\r\n\r\n- first\n- **second**\n\n"
    response = builtin_client.post(
        "/v1/workspaces/00000000-0000-0000-0000-000000000007/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": source},
                },
                {
                    "id": "markdown",
                    "operator_id": "text.as_markdown",
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "text",
                    "to_node": "markdown",
                    "to_port": "text",
                }
            ],
        },
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    markdown_run = next(
        node_run for node_run in result.node_runs if node_run.node_id == "markdown"
    )
    markdown_ref = markdown_run.outputs[0].artifacts[0]
    assert markdown_ref.artifact_type == "text.markdown"
    assert markdown_ref.text == source
    assert builtin_client.get(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{markdown_ref.artifact_id}/content"
    ).json() == {"markdown": source}

    # The artifact summary advertises its download formats.
    assert [entry.format for entry in markdown_ref.download_formats] == [
        "json",
        "txt",
    ]

    # Downloading as txt yields the bare markdown, not the JSON envelope.
    download = builtin_client.get(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{markdown_ref.artifact_id}/download",
        params={"format": "txt"},
    )
    assert download.status_code == 200
    assert download.content.decode("utf-8") == source
    assert download.headers["content-disposition"].startswith("attachment")
    assert "text/plain" in download.headers["content-type"]

    # Downloading as json returns the canonical payload envelope.
    json_download = builtin_client.get(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{markdown_ref.artifact_id}/download",
        params={"format": "json"},
    )
    assert json_download.status_code == 200
    assert json_download.json() == {"markdown": source}

    # An unsupported format is rejected with 400.
    bad = builtin_client.get(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{markdown_ref.artifact_id}/download",
        params={"format": "csv"},
    )
    assert bad.status_code == 400


def test_text_graph_splits_maps_replacement_and_joins(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/workspaces/00000000-0000-0000-0000-000000000007/runs",
        json={
            "nodes": [
                {
                    "id": "input",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "alpha||beta||||gamma||"},
                },
                {
                    "id": "split",
                    "operator_id": "text.split",
                    "operator_version": 1,
                    "config": {"separator": "||"},
                },
                {
                    "id": "replace",
                    "operator_id": "text.replace",
                    "operator_version": 1,
                    "config": {"search": "a", "replacement": "A"},
                },
                {
                    "id": "join",
                    "operator_id": "text.join",
                    "operator_version": 1,
                    "config": {"separator": "|"},
                },
            ],
            "edges": [
                {
                    "from_node": "input",
                    "from_port": "text",
                    "to_node": "split",
                    "to_port": "text",
                },
                {
                    "from_node": "split",
                    "from_port": "parts",
                    "to_node": "replace",
                    "to_port": "text",
                    "collection_mode": "map",
                },
                {
                    "from_node": "replace",
                    "from_port": "text",
                    "to_node": "join",
                    "to_port": "parts",
                },
            ],
        },
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    runs = {run.node_id: run for run in result.node_runs}
    assert [
        builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{artifact.artifact_id}/content").json()[
            "value"
        ]
        for artifact in runs["split"].outputs[0].artifacts
    ] == ["alpha", "beta", "", "gamma", ""]
    assert [
        builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{artifact.artifact_id}/content").json()[
            "value"
        ]
        for artifact in runs["replace"].outputs[0].artifacts
    ] == ["AlphA", "betA", "", "gAmmA", ""]

    joined = runs["join"].outputs[0].artifacts[0]
    assert builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{joined.artifact_id}/content").json() == {
        "value": "AlphA|betA||gAmmA|"
    }


def test_sequence_collect_accepts_text_shapes_in_declared_plug_order(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/runs", json=_collect_run_payload())

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    collect_run = next(run for run in result.node_runs if run.node_id == "collect")
    output = collect_run.outputs[0]
    assert isinstance(output.value, ArtifactRefSequence)
    assert [
        builtin_client.get(f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{artifact.artifact_id}/content").json()[
            "value"
        ]
        for artifact in output.artifacts
    ] == ["first", "second", "third"]
    assert output.value.metadata == {
        "collect_segments": [
            {
                "input_index": 0,
                "start_index": 0,
                "item_count": 2,
                "source_kind": "sequence",
            },
            {
                "input_index": 1,
                "start_index": 2,
                "item_count": 1,
                "source_kind": "single",
            },
        ]
    }


def test_sequence_collect_rejects_invalid_executable_plug_structures(
    builtin_client: TestClient,
) -> None:
    invalid_payloads: list[tuple[dict[str, object], str]] = []

    missing_plugs = _collect_run_payload()
    _collect_node(missing_plugs)["input_plugs"] = []
    invalid_payloads.append((missing_plugs, "has no submitted plugs"))

    duplicate_plugs = _collect_run_payload()
    duplicate_plug_values = _collect_input_plugs(duplicate_plugs)
    duplicate_plug_values[1]["id"] = duplicate_plug_values[0]["id"]
    invalid_payloads.append((duplicate_plugs, "duplicate input plug id"))

    unknown_port = _collect_run_payload()
    unknown_port_plugs = _collect_input_plugs(unknown_port)
    unknown_port_plugs[0]["port"] = "absent"
    invalid_payloads.append((unknown_port, "unknown input port"))

    normal_port_plug = _collect_run_payload()
    normal_nodes = cast(
        list[dict[str, object]],
        normal_port_plug["nodes"],
    )
    split_node = next(node for node in normal_nodes if node["id"] == "split")
    split_node["input_plugs"] = [{"id": "normal", "port": "text"}]
    invalid_payloads.append((normal_port_plug, "does not accept instance plugs"))

    missing_edge_plug = _collect_run_payload()
    _collect_edges(missing_edge_plug)[0].pop("to_plug")
    invalid_payloads.append((missing_edge_plug, "must target an input plug"))

    unknown_edge_plug = _collect_run_payload()
    _collect_edges(unknown_edge_plug)[0]["to_plug"] = "absent"
    invalid_payloads.append((unknown_edge_plug, "unknown input plug"))

    duplicate_edge_plug = _collect_run_payload()
    duplicate_edges = _collect_edges(duplicate_edge_plug)
    duplicate_edges[1]["to_plug"] = duplicate_edges[0]["to_plug"]
    invalid_payloads.append((duplicate_edge_plug, "exactly one incoming edge"))

    mapped_plug = _collect_run_payload()
    _collect_edges(mapped_plug)[0]["collection_mode"] = "map"
    invalid_payloads.append((mapped_plug, "cannot use collection mode 'map'"))

    normal_edge_plug = _collect_run_payload()
    all_edges = cast(list[dict[str, object]], normal_edge_plug["edges"])
    all_edges[0]["to_plug"] = "normal"
    invalid_payloads.append((normal_edge_plug, "does not accept instance plugs"))

    unconnected_plug = _collect_run_payload()
    unconnected_values = _collect_input_plugs(unconnected_plug)
    unconnected_values.append({"id": "unconnected", "port": "items"})
    invalid_payloads.append((unconnected_plug, "requires exactly one incoming edge"))

    for payload, expected_detail in invalid_payloads:
        response = builtin_client.post("/v1/workspaces/00000000-0000-0000-0000-000000000007/runs", json=payload)

        assert response.status_code == 422
        assert expected_detail in response.json()["detail"]
