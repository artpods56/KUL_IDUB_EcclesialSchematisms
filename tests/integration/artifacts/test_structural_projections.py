from fastapi.testclient import TestClient

from grafy_api.v1.routes.catalog.models import NodeRegistryResponse
from grafy_api.v1.routes.executions.models import RunResponse


def test_registry_derives_nested_json_scalar_projections(
    structural_projection_client: TestClient,
) -> None:
    response = structural_projection_client.get("/v1/workspaces/00000000-0000-0000-0000-000000000007/nodes")

    assert response.status_code == 200
    registry = NodeRegistryResponse.model_validate(response.json())
    api_response = next(
        artifact_type
        for artifact_type in registry.artifact_types
        if artifact_type.key.id == "test.api_response"
    )
    assert [
        projection.model_dump() for projection in api_response.field_projections
    ] == [
        {
            "path": ["customer", "display_name"],
            "target_artifact_type": {
                "id": "scalar.text",
                "schema_version": 1,
            },
            "title": "Customer · Display name",
        },
        {
            "path": ["customer", "retry_count"],
            "target_artifact_type": {
                "id": "scalar.integer",
                "schema_version": 1,
            },
            "title": "Customer · Retry count",
        },
    ]


def test_nested_json_string_projects_directly_and_integer_converts_to_text(
    structural_projection_client: TestClient,
) -> None:
    response = structural_projection_client.post(
        "/v1/workspaces/00000000-0000-0000-0000-000000000007/runs",
        json={
            "nodes": [
                {
                    "id": "api",
                    "operator_id": "test.api_response",
                    "operator_version": 1,
                    "config": {},
                },
                {
                    "id": "name",
                    "operator_id": "text.replace",
                    "operator_version": 1,
                    "config": {"search": "b", "replacement": "B"},
                },
                {
                    "id": "retries",
                    "operator_id": "text.replace",
                    "operator_version": 1,
                    "config": {"search": "4", "replacement": "four-"},
                },
            ],
            "edges": [
                {
                    "from_node": "api",
                    "from_port": "response",
                    "to_node": "name",
                    "to_port": "text",
                    "projection": {"path": ["customer", "display_name"]},
                },
                {
                    "from_node": "api",
                    "from_port": "response",
                    "to_node": "retries",
                    "to_port": "text",
                    "projection": {"path": ["customer", "retry_count"]},
                    "conversion_path": [
                        {
                            "id": "builtin.scalar.integer_to_text",
                            "version": 1,
                        }
                    ],
                },
            ],
        },
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    outputs_by_node = {
        node_run.node_id: node_run.outputs[0].artifacts[0]
        for node_run in result.node_runs
        if node_run.node_id in {"name", "retries"}
    }
    name = outputs_by_node["name"]
    retries = outputs_by_node["retries"]
    assert structural_projection_client.get(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{name.artifact_id}/content"
    ).json() == {"value": "aBc"}
    assert structural_projection_client.get(
        f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{retries.artifact_id}/content"
    ).json() == {"value": "four-2"}


def test_projected_values_feed_generic_collect_with_optional_conversion(
    structural_projection_client: TestClient,
) -> None:
    response = structural_projection_client.post(
        "/v1/workspaces/00000000-0000-0000-0000-000000000007/runs",
        json={
            "nodes": [
                {
                    "id": "api",
                    "operator_id": "test.api_response",
                    "operator_version": 1,
                    "config": {},
                },
                {
                    "id": "collect",
                    "operator_id": "sequence.collect",
                    "operator_version": 1,
                    "config": {},
                    "input_plugs": [
                        {"id": "name", "port": "items"},
                        {"id": "retries", "port": "items"},
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
                    "from_node": "api",
                    "from_port": "response",
                    "to_node": "collect",
                    "to_port": "items",
                    "to_plug": "name",
                    "projection": {"path": ["customer", "display_name"]},
                },
                {
                    "from_node": "api",
                    "from_port": "response",
                    "to_node": "collect",
                    "to_port": "items",
                    "to_plug": "retries",
                    "projection": {"path": ["customer", "retry_count"]},
                    "conversion_path": [
                        {
                            "id": "builtin.scalar.integer_to_text",
                            "version": 1,
                        }
                    ],
                },
            ],
        },
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    collect_output = next(
        run.outputs[0] for run in result.node_runs if run.node_id == "collect"
    )
    assert [
        structural_projection_client.get(
            f"/v1/workspaces/00000000-0000-0000-0000-000000000007/artifacts/{artifact.artifact_id}/content"
        ).json()["value"]
        for artifact in collect_output.artifacts
    ] == ["abc", "42"]
