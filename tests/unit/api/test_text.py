from fastapi.testclient import TestClient

from notarius_api.schemas.workbench import NodeRegistryResponse, RunResponse
from notarius_core.operators.text import TextInputConfig, TextValue


def test_registry_declares_text_artifact_and_operator_contracts(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.get("/v1/nodes")

    assert response.status_code == 200
    registry = NodeRegistryResponse.model_validate(response.json())
    artifact_types = {
        artifact_type.key.id: artifact_type for artifact_type in registry.artifact_types
    }
    assert artifact_types["scalar.text"].payload_schema == TextValue.model_json_schema()

    nodes = {node.operator_id: node for node in registry.nodes}
    assert nodes["text.input"].config_schema == TextInputConfig.model_json_schema()
    assert nodes["text.input"].outputs[0].name == "text"
    assert nodes["text.input"].outputs[0].shape == "one"

    assert nodes["text.split"].inputs[0].name == "text"
    assert nodes["text.split"].inputs[0].shape == "one"
    assert nodes["text.split"].outputs[0].name == "parts"
    assert nodes["text.split"].outputs[0].shape == "many"

    assert nodes["text.join"].inputs[0].name == "parts"
    assert nodes["text.join"].inputs[0].shape == "many"


def test_text_graph_splits_maps_replacement_and_joins(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
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
        builtin_client.get(f"/v1/artifacts/{artifact.artifact_id}/content").json()[
            "value"
        ]
        for artifact in runs["split"].outputs[0].artifacts
    ] == ["alpha", "beta", "", "gamma", ""]
    assert [
        builtin_client.get(f"/v1/artifacts/{artifact.artifact_id}/content").json()[
            "value"
        ]
        for artifact in runs["replace"].outputs[0].artifacts
    ] == ["AlphA", "betA", "", "gAmmA", ""]

    joined = runs["join"].outputs[0].artifacts[0]
    assert builtin_client.get(f"/v1/artifacts/{joined.artifact_id}/content").json() == {
        "value": "AlphA|betA||gAmmA|"
    }
