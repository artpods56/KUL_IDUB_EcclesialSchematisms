import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from notarius_api.schemas.workbench import (
    NodeRegistryResponse,
    RunEdgeRequest,
    RunPortOutputResponse,
    RunResponse,
)
from notarius_core.artifacts import ArtifactRef, ArtifactRefSequence
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


def _run_output(
    result: RunResponse,
    node_id: str,
    port: str,
) -> RunPortOutputResponse:
    node_run = next(run for run in result.node_runs if run.node_id == node_id)
    return next(output for output in node_run.outputs if output.port == port)


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


def test_selected_target_projects_two_edges_from_one_pinned_compound_output(
    builtin_client: TestClient,
) -> None:
    upstream_response = builtin_client.post(
        "/v1/runs",
        json=_arithmetic_run_request(
            left_projection={"path": ["addition"]},
            right_projection={"path": ["subtraction"]},
        ),
    )
    assert upstream_response.status_code == 200
    upstream_result = RunResponse.model_validate(upstream_response.json())
    compound_output = _run_output(upstream_result, "add-subtract", "result")
    assert isinstance(compound_output.value, ArtifactRef)
    assert compound_output.value.artifact_id == compound_output.artifacts[0].artifact_id
    assert compound_output.value.content_hash == compound_output.artifacts[0].sha256

    selected_response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "multiply",
                    "operator_id": "arithmetic.multiply",
                    "operator_version": 1,
                    "config": {},
                }
            ],
            "edges": [
                {
                    "from_node": "add-subtract",
                    "from_port": "result",
                    "to_node": "multiply",
                    "to_port": "left",
                    "projection": {"path": ["addition"]},
                },
                {
                    "from_node": "add-subtract",
                    "from_port": "result",
                    "to_node": "multiply",
                    "to_port": "right",
                    "projection": {"path": ["subtraction"]},
                },
            ],
            "pinned_outputs": [
                {
                    "from_node": "add-subtract",
                    "from_port": "result",
                    "value": compound_output.value.model_dump(mode="json"),
                }
            ],
        },
    )

    assert selected_response.status_code == 200
    selected_result = RunResponse.model_validate(selected_response.json())
    assert selected_result.status == "succeeded"
    assert [run.node_id for run in selected_result.node_runs] == ["multiply"]
    product_output = _run_output(selected_result, "multiply", "result")
    assert isinstance(product_output.value, ArtifactRef)
    assert product_output.value.artifact_id == product_output.artifacts[0].artifact_id
    assert product_output.value.content_hash == product_output.artifacts[0].sha256
    assert product_output.artifacts[0].text == "65"


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


def test_selected_mapped_run_uses_exact_pinned_sequence_envelope_in_order(
    builtin_client: TestClient,
) -> None:
    upstream_response = builtin_client.post(
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
                    "id": "ten",
                    "operator_id": "arithmetic.number",
                    "operator_version": 1,
                    "config": {"value": 10},
                },
            ],
            "edges": [],
        },
    )
    assert upstream_response.status_code == 200
    upstream_result = RunResponse.model_validate(upstream_response.json())
    sequence_output = _run_output(upstream_result, "sequence", "values")
    ten_output = _run_output(upstream_result, "ten", "value")
    assert isinstance(sequence_output.value, ArtifactRefSequence)
    assert isinstance(ten_output.value, ArtifactRef)
    assert [ref.artifact_id for ref in sequence_output.value.item_refs] == [
        artifact.artifact_id for artifact in sequence_output.artifacts
    ]

    selected_response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "multiply",
                    "operator_id": "arithmetic.multiply",
                    "operator_version": 1,
                    "config": {},
                }
            ],
            "edges": [
                {
                    "from_node": "sequence",
                    "from_port": "values",
                    "to_node": "multiply",
                    "to_port": "left",
                    "collection_mode": "map",
                },
                {
                    "from_node": "ten",
                    "from_port": "value",
                    "to_node": "multiply",
                    "to_port": "right",
                },
            ],
            "pinned_outputs": [
                {
                    "from_node": "sequence",
                    "from_port": "values",
                    "value": sequence_output.value.model_dump(mode="json"),
                },
                {
                    "from_node": "ten",
                    "from_port": "value",
                    "value": ten_output.value.model_dump(mode="json"),
                },
            ],
        },
    )

    assert selected_response.status_code == 200
    selected_result = RunResponse.model_validate(selected_response.json())
    multiplied_output = _run_output(selected_result, "multiply", "result")
    assert isinstance(multiplied_output.value, ArtifactRefSequence)
    assert [artifact.text for artifact in multiplied_output.artifacts] == [
        "10",
        "20",
        "30",
    ]
    assert multiplied_output.value.ordered is sequence_output.value.ordered
    assert multiplied_output.value.index_key == sequence_output.value.index_key
    assert multiplied_output.value.metadata["source_sequence_id"] == str(
        sequence_output.value.sequence_id
    )


def test_selected_run_uses_submitted_older_or_newer_pin_without_latest_lookup(
    builtin_client: TestClient,
) -> None:
    pinned_values: list[ArtifactRef] = []
    for value in (3, 7):
        response = builtin_client.post(
            "/v1/runs",
            json={
                "nodes": [
                    {
                        "id": "source",
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
        output_value = _run_output(result, "source", "value").value
        assert isinstance(output_value, ArtifactRef)
        pinned_values.append(output_value)
    assert pinned_values[0].artifact_id != pinned_values[1].artifact_id

    products: list[str | None] = []
    for pinned_value in pinned_values:
        response = builtin_client.post(
            "/v1/runs",
            json={
                "nodes": [
                    {
                        "id": "multiply",
                        "operator_id": "arithmetic.multiply",
                        "operator_version": 1,
                        "config": {},
                    }
                ],
                "edges": [
                    {
                        "from_node": "source",
                        "from_port": "value",
                        "to_node": "multiply",
                        "to_port": "left",
                    },
                    {
                        "from_node": "source",
                        "from_port": "value",
                        "to_node": "multiply",
                        "to_port": "right",
                    },
                ],
                "pinned_outputs": [
                    {
                        "from_node": "source",
                        "from_port": "value",
                        "value": pinned_value.model_dump(mode="json"),
                    }
                ],
            },
        )
        assert response.status_code == 200
        result = RunResponse.model_validate(response.json())
        products.append(_run_output(result, "multiply", "result").artifacts[0].text)

    assert products == ["9", "49"]


@pytest.mark.parametrize(
    ("case", "error_fragment"),
    [
        ("missing-artifact", "references missing artifact"),
        ("mismatched-ref", "does not match the repository ref for artifact"),
        ("missing-pin", "requires a pinned output"),
        ("duplicate-pin", "Duplicate pinned output"),
        ("unused-pin", "is not used by any incoming edge"),
        (
            "wrong-artifact-type",
            "cannot connect arithmetic.result@1 to scalar.integer@1",
        ),
    ],
)
def test_invalid_selected_run_pins_are_rejected_before_target_execution(
    builtin_client: TestClient,
    case: str,
    error_fragment: str,
) -> None:
    upstream_response = builtin_client.post(
        "/v1/runs",
        json=_arithmetic_run_request(
            left_projection={"path": ["addition"]},
            right_projection={"path": ["subtraction"]},
        ),
    )
    assert upstream_response.status_code == 200
    upstream_result = RunResponse.model_validate(upstream_response.json())
    integer_value = _run_output(upstream_result, "nine", "value").value
    compound_value = _run_output(upstream_result, "add-subtract", "result").value
    assert isinstance(integer_value, ArtifactRef)
    assert isinstance(compound_value, ArtifactRef)

    edges: list[dict[str, object]] = [
        {
            "from_node": "source",
            "from_port": "value",
            "to_node": "multiply",
            "to_port": "left",
        },
        {
            "from_node": "source",
            "from_port": "value",
            "to_node": "multiply",
            "to_port": "right",
        },
    ]
    pinned_outputs: list[dict[str, object]] = [
        {
            "from_node": "source",
            "from_port": "value",
            "value": integer_value.model_dump(mode="json"),
        }
    ]
    if case == "missing-artifact":
        missing_value = integer_value.model_copy(
            update={"artifact_id": uuid4()},
        )
        pinned_outputs[0]["value"] = missing_value.model_dump(mode="json")
    elif case == "mismatched-ref":
        mismatched_value = integer_value.model_copy(
            update={"content_hash": "not-the-repository-hash"},
        )
        pinned_outputs[0]["value"] = mismatched_value.model_dump(mode="json")
    elif case == "missing-pin":
        pinned_outputs = []
    elif case == "duplicate-pin":
        pinned_outputs.append(dict(pinned_outputs[0]))
    elif case == "unused-pin":
        pinned_outputs.append(
            {
                "from_node": "unused",
                "from_port": "value",
                "value": integer_value.model_dump(mode="json"),
            }
        )
    elif case == "wrong-artifact-type":
        edges = [
            {
                "from_node": "source",
                "from_port": "result",
                "to_node": "multiply",
                "to_port": "left",
            },
            {
                "from_node": "source",
                "from_port": "result",
                "to_node": "multiply",
                "to_port": "right",
            },
        ]
        pinned_outputs = [
            {
                "from_node": "source",
                "from_port": "result",
                "value": compound_value.model_dump(mode="json"),
            }
        ]

    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "multiply",
                    "operator_id": "arithmetic.multiply",
                    "operator_version": 1,
                    "config": {},
                }
            ],
            "edges": edges,
            "pinned_outputs": pinned_outputs,
        },
    )

    assert response.status_code == 422
    assert error_fragment in response.json()["detail"]


def test_pin_for_executing_source_is_rejected_before_source_config_runs(
    builtin_client: TestClient,
) -> None:
    upstream_response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "arithmetic.number",
                    "operator_version": 1,
                    "config": {"value": 3},
                }
            ],
            "edges": [],
        },
    )
    assert upstream_response.status_code == 200
    upstream_result = RunResponse.model_validate(upstream_response.json())
    pinned_value = _run_output(upstream_result, "source", "value").value
    assert isinstance(pinned_value, ArtifactRef)

    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "source",
                    "operator_id": "arithmetic.number",
                    "operator_version": 1,
                    "config": {"value": "invalid"},
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
                    "from_node": "source",
                    "from_port": "value",
                    "to_node": "multiply",
                    "to_port": "left",
                },
                {
                    "from_node": "source",
                    "from_port": "value",
                    "to_node": "multiply",
                    "to_port": "right",
                },
            ],
            "pinned_outputs": [
                {
                    "from_node": "source",
                    "from_port": "value",
                    "value": pinned_value.model_dump(mode="json"),
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Pinned output 'source'.'value' is invalid because source node "
        "'source' is also being executed"
    )


def test_crossing_edge_to_unknown_target_is_rejected(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [],
            "edges": [
                {
                    "from_node": "source",
                    "from_port": "value",
                    "to_node": "missing-target",
                    "to_port": "left",
                }
            ],
            "pinned_outputs": [],
        },
    )

    assert response.status_code == 422
    assert (
        "references unknown target node 'missing-target'" in response.json()["detail"]
    )


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
