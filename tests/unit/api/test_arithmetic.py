import json
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from notarius_api.schemas.workbench import (
    NodeRegistryResponse,
    RunEdgeRequest,
    RunResponse,
)
from notarius_core.operators.arithmetic import (
    AddSubtractInput,
    ArithmeticResult,
    IntegerSequenceConfig,
    IntegerSequenceOutput,
    IntegerValuePayload,
    MultiplyInput,
    MultiplyOutput,
    NumberConfig,
    NumberOutput,
    SumIntegersInput,
    SumIntegersOutput,
)


def _arithmetic_run_request(
    *,
    left_projection: dict[str, list[str]] | None,
    right_projection: dict[str, list[str]] | None,
    number_values: tuple[object, object] = (9, 4),
) -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": "nine",
                "operator_id": "arithmetic.number",
                "operator_version": 1,
                "config": {"value": number_values[0]},
            },
            {
                "id": "four",
                "operator_id": "arithmetic.number",
                "operator_version": 1,
                "config": {"value": number_values[1]},
            },
            {
                "id": "add-subtract",
                "operator_id": "arithmetic.add_subtract",
                "operator_version": 1,
                "config": {},
            },
            {
                "id": "multiply",
                "operator_id": "arithmetic.multiply",
                "operator_version": 1,
                "config": {},
            },
        ],
        "edges": [
            {
                "from_node": "nine",
                "from_port": "value",
                "to_node": "add-subtract",
                "to_port": "left",
            },
            {
                "from_node": "four",
                "from_port": "value",
                "to_node": "add-subtract",
                "to_port": "right",
            },
            {
                "from_node": "add-subtract",
                "from_port": "result",
                "to_node": "multiply",
                "to_port": "left",
                "projection": left_projection,
            },
            {
                "from_node": "add-subtract",
                "from_port": "result",
                "to_node": "multiply",
                "to_port": "right",
                "projection": right_projection,
            },
        ],
    }


def _mapped_sum_run_request(
    *,
    collection_mode: str,
) -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": "sequence",
                "operator_id": "arithmetic.integer_sequence",
                "operator_version": 1,
                "config": {"start": 1, "count": 3, "step": 1},
            },
            {
                "id": "ten",
                "operator_id": "arithmetic.number",
                "operator_version": 1,
                "config": {"value": 10},
            },
            {
                "id": "multiply",
                "operator_id": "arithmetic.multiply",
                "operator_version": 1,
                "config": {},
            },
            {
                "id": "sum",
                "operator_id": "arithmetic.sum",
                "operator_version": 1,
                "config": {},
            },
        ],
        "edges": [
            {
                "from_node": "sequence",
                "from_port": "values",
                "to_node": "multiply",
                "to_port": "left",
                "collection_mode": collection_mode,
            },
            {
                "from_node": "ten",
                "from_port": "value",
                "to_node": "multiply",
                "to_port": "right",
            },
            {
                "from_node": "multiply",
                "from_port": "result",
                "to_node": "sum",
                "to_port": "values",
            },
        ],
    }


def test_registry_declares_generic_integer_and_arithmetic_result_projections(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.get("/v1/nodes")

    assert response.status_code == 200
    registry = NodeRegistryResponse.model_validate(response.json())
    artifact_types = {
        artifact_type.key.id: artifact_type for artifact_type in registry.artifact_types
    }
    assert "arithmetic.addition" not in artifact_types
    assert "arithmetic.subtraction" not in artifact_types
    assert artifact_types["scalar.integer"].field_projections == []

    result_type = artifact_types["arithmetic.result"]
    assert result_type.payload_schema == ArithmeticResult.model_json_schema()
    assert [
        projection.model_dump() for projection in result_type.field_projections
    ] == [
        {
            "path": ["addition"],
            "target_artifact_type": {
                "id": "scalar.integer",
                "schema_version": 1,
            },
            "title": "Addition",
        },
        {
            "path": ["subtraction"],
            "target_artifact_type": {
                "id": "scalar.integer",
                "schema_version": 1,
            },
            "title": "Subtraction",
        },
    ]

    nodes = {node.operator_id: node for node in registry.nodes}
    assert [
        (nodes[operator_id].title, nodes[operator_id].plugin_slug)
        for operator_id in (
            "arithmetic.number",
            "arithmetic.add_subtract",
            "arithmetic.multiply",
        )
    ] == [
        ("Number", "builtin.arithmetic"),
        ("Add & subtract", "builtin.arithmetic"),
        ("Multiply", "builtin.arithmetic"),
    ]


def test_arithmetic_graph_projects_both_result_fields_into_multiply(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json=_arithmetic_run_request(
            left_projection={"path": ["addition"]},
            right_projection={"path": ["subtraction"]},
        ),
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    runs = {run.node_id: run for run in result.node_runs}
    assert all(run.status == "succeeded" for run in runs.values())
    assert runs["nine"].outputs[0].artifacts[0].text == "9"
    assert runs["four"].outputs[0].artifacts[0].text == "4"

    compound = runs["add-subtract"].outputs[0].artifacts[0]
    assert compound.artifact_type == "arithmetic.result"
    assert json.loads(compound.text or "") == {
        "addition": 13,
        "subtraction": 5,
    }
    compound_content = builtin_client.get(
        f"/v1/artifacts/{compound.artifact_id}/content"
    )
    assert compound_content.status_code == 200
    assert compound_content.json() == {"addition": 13, "subtraction": 5}

    product = runs["multiply"].outputs[0].artifacts[0]
    assert product.artifact_type == "scalar.integer"
    assert product.text == "65"
    product_content = builtin_client.get(f"/v1/artifacts/{product.artifact_id}/content")
    assert product_content.status_code == 200
    assert product_content.json() == {"value": 65}


def test_integer_sequence_maps_multiply_and_sums_once(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json=_mapped_sum_run_request(collection_mode="map"),
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "succeeded"
    runs = {run.node_id: run for run in result.node_runs}
    assert [artifact.text for artifact in runs["sequence"].outputs[0].artifacts] == [
        "1",
        "2",
        "3",
    ]
    assert runs["multiply"].outputs[0].kind == "sequence"
    assert [artifact.text for artifact in runs["multiply"].outputs[0].artifacts] == [
        "10",
        "20",
        "30",
    ]
    assert runs["sum"].outputs[0].kind == "single"
    assert runs["sum"].outputs[0].artifacts[0].text == "60"


def test_many_output_cannot_feed_once_item_input(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json=_mapped_sum_run_request(collection_mode="direct"),
    )

    assert response.status_code == 422
    assert "incompatible shapes" in response.json()["detail"]
    assert "source is 'many', target expects 'one'" in response.json()["detail"]


def test_invalid_map_edge_target_is_rejected_before_execution(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "sequence",
                    "operator_id": "arithmetic.integer_sequence",
                    "operator_version": 1,
                    "config": {"start": 1, "count": 3, "step": 1},
                },
                {
                    "id": "multiply",
                    "operator_id": "arithmetic.multiply",
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "sequence",
                    "from_port": "values",
                    "to_node": "multiply",
                    "to_port": "missing",
                    "collection_mode": "map",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert "cannot drive mapped execution" in response.json()["detail"]
    assert "missing" in response.json()["detail"]


def test_node_rejects_more_than_one_map_edge(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "left-sequence",
                    "operator_id": "arithmetic.integer_sequence",
                    "operator_version": 1,
                    "config": {"start": 1, "count": 2, "step": 1},
                },
                {
                    "id": "right-sequence",
                    "operator_id": "arithmetic.integer_sequence",
                    "operator_version": 1,
                    "config": {"start": 3, "count": 2, "step": 1},
                },
                {
                    "id": "multiply",
                    "operator_id": "arithmetic.multiply",
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "left-sequence",
                    "from_port": "values",
                    "to_node": "multiply",
                    "to_port": "left",
                    "collection_mode": "map",
                },
                {
                    "from_node": "right-sequence",
                    "from_port": "values",
                    "to_node": "multiply",
                    "to_port": "right",
                    "collection_mode": "map",
                },
            ],
        },
    )

    assert response.status_code == 422
    assert "more than one map edge" in response.json()["detail"]
    assert "exactly one edge may drive mapped execution" in response.json()["detail"]


def test_unknown_operator_version_is_rejected_before_execution(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "number",
                    "operator_id": "arithmetic.number",
                    "operator_version": 99,
                    "config": {"value": "not-an-integer"},
                }
            ],
            "edges": [],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Unknown operator 'arithmetic.number' at version 99"
    )


@pytest.mark.parametrize(
    ("projection", "error_fragment"),
    [
        (None, "without a declared field projection"),
        ({"path": ["missing"]}, "requests undeclared projection 'missing'"),
    ],
)
def test_invalid_arithmetic_projection_is_422_before_node_execution(
    builtin_client: TestClient,
    projection: dict[str, list[str]] | None,
    error_fragment: str,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json=_arithmetic_run_request(
            left_projection=projection,
            right_projection={"path": ["subtraction"]},
            number_values=("not-an-integer", "also-not-an-integer"),
        ),
    )

    assert response.status_code == 422
    assert error_fragment in response.json()["detail"]


def test_missing_required_arithmetic_input_is_422_before_node_execution(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "invalid-number",
                    "operator_id": "arithmetic.number",
                    "operator_version": 1,
                    "config": {"value": "not-an-integer"},
                },
                {
                    "id": "add-subtract",
                    "operator_id": "arithmetic.add_subtract",
                    "operator_version": 1,
                    "config": {},
                },
            ],
            "edges": [
                {
                    "from_node": "invalid-number",
                    "from_port": "value",
                    "to_node": "add-subtract",
                    "to_port": "left",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Node 'add-subtract' (arithmetic.add_subtract@1) required input "
        "'right' has no incoming edge"
    )


@pytest.mark.parametrize("value", [True, "9"])
def test_number_node_config_does_not_coerce_non_integer_values(
    builtin_client: TestClient,
    value: object,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "number",
                    "operator_id": "arithmetic.number",
                    "operator_version": 1,
                    "config": {"value": value},
                }
            ],
            "edges": [],
        },
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    assert result.status == "failed"
    assert len(result.node_runs) == 1
    node_run = result.node_runs[0]
    assert node_run.status == "failed"
    assert node_run.outputs == []
    assert node_run.error is not None
    assert "Input should be a valid integer" in node_run.error


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (NumberConfig, {"value": True}),
        (NumberConfig, {"value": "9"}),
        (NumberOutput, {"value": True}),
        (NumberOutput, {"value": "9"}),
        (IntegerValuePayload, {"value": True}),
        (IntegerValuePayload, {"value": "9"}),
        (ArithmeticResult, {"addition": "13", "subtraction": 5}),
        (ArithmeticResult, {"addition": 13, "subtraction": False}),
        (AddSubtractInput, {"left": True, "right": 4}),
        (AddSubtractInput, {"left": 9, "right": "4"}),
        (MultiplyInput, {"left": True, "right": 4}),
        (MultiplyInput, {"left": 9, "right": "4"}),
        (MultiplyOutput, {"result": True}),
        (MultiplyOutput, {"result": "36"}),
        (IntegerSequenceConfig, {"start": True, "count": 3, "step": 1}),
        (IntegerSequenceConfig, {"start": 0, "count": "3", "step": 1}),
        (IntegerSequenceConfig, {"start": 0, "count": 3, "step": False}),
        (IntegerSequenceOutput, {"values": [1, "2", 3]}),
        (SumIntegersInput, {"values": [1, False, 3]}),
        (SumIntegersOutput, {"result": "6"}),
    ],
)
def test_arithmetic_models_reject_bool_and_string_integers(
    model: type[BaseModel],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="Input should be a valid integer"):
        model.model_validate(payload)


@pytest.mark.parametrize("count", [0, 10_001])
def test_integer_sequence_count_is_bounded(count: int) -> None:
    with pytest.raises(ValidationError):
        IntegerSequenceConfig(count=count)


def test_sum_integers_requires_at_least_one_value() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        SumIntegersInput(values=[])


def test_edge_collection_mode_is_narrow() -> None:
    with pytest.raises(ValidationError):
        RunEdgeRequest.model_validate(
            {
                "from_node": "source",
                "from_port": "values",
                "to_node": "target",
                "to_port": "value",
                "collection_mode": "broadcast",
            }
        )
