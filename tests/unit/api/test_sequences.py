import asyncio

import pytest
from fastapi.testclient import TestClient

from notarius_core.artifacts import (
    SOURCE_PAGE_IMAGE,
    ArtifactObject,
    ArtifactRefSequence,
    InMemoryUnitOfWork,
)

from notarius_api.schemas.workbench import NodeRegistryResponse, RunResponse


async def _store_artifacts(
    uow: InMemoryUnitOfWork,
    artifacts: list[ArtifactObject],
) -> None:
    async with uow as entered:
        for artifact in artifacts:
            await entered.artifacts.add(artifact)
        await entered.commit()


def _binding(artifact_type: str, schema_version: int = 1) -> list[dict[str, object]]:
    return [
        {
            "variable": "T",
            "artifact_type": {
                "id": artifact_type,
                "schema_version": schema_version,
            },
        }
    ]


def test_registry_declares_generic_collect_contract(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.get("/v1/nodes")

    assert response.status_code == 200
    registry = NodeRegistryResponse.model_validate(response.json())
    collect = next(
        node for node in registry.nodes if node.operator_id == "sequence.collect"
    )
    assert collect.plugin_slug == "builtin.sequence"
    assert collect.title == "Collect"
    assert collect.inputs[0].artifact_type is None
    assert collect.inputs[0].artifact_type_variable == "T"
    assert collect.inputs[0].accepted_shapes == ["one", "many"]
    assert collect.inputs[0].instance_plugs is True
    assert collect.outputs[0].artifact_type is None
    assert collect.outputs[0].artifact_type_variable == "T"
    assert collect.outputs[0].shape == "many"


def test_collect_flattens_image_scalar_and_sequence_in_plug_order(
    conversion_path_client: tuple[TestClient, InMemoryUnitOfWork],
) -> None:
    client, uow = conversion_path_client
    images = [
        ArtifactObject(
            artifact_type=SOURCE_PAGE_IMAGE.key.id,
            schema_version=SOURCE_PAGE_IMAGE.key.schema_version,
            content_type="image/png",
            inline_payload={"index": index},
        )
        for index in range(4)
    ]
    asyncio.run(_store_artifacts(uow, images))
    image_sequence = ArtifactRefSequence.from_key(
        key=SOURCE_PAGE_IMAGE.key,
        item_refs=[image.ref() for image in images[1:]],
    )

    response = client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "collect",
                    "operator_id": "sequence.collect",
                    "operator_version": 1,
                    "config": {},
                    "input_plugs": [
                        {"id": "sequence", "port": "items"},
                        {"id": "single", "port": "items"},
                    ],
                    "artifact_type_bindings": _binding("source.page_image"),
                }
            ],
            "edges": [
                {
                    "from_node": "external-sequence",
                    "from_port": "images",
                    "to_node": "collect",
                    "to_port": "items",
                    "to_plug": "sequence",
                },
                {
                    "from_node": "external-single",
                    "from_port": "image",
                    "to_node": "collect",
                    "to_port": "items",
                    "to_plug": "single",
                },
            ],
            "pinned_outputs": [
                {
                    "from_node": "external-sequence",
                    "from_port": "images",
                    "value": image_sequence.model_dump(mode="json"),
                },
                {
                    "from_node": "external-single",
                    "from_port": "image",
                    "value": images[0].ref().model_dump(mode="json"),
                },
            ],
        },
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    output = result.node_runs[0].outputs[0]
    assert isinstance(output.value, ArtifactRefSequence)
    assert output.value.artifact_type == "source.page_image"
    assert [ref.artifact_id for ref in output.value.item_refs] == [
        images[1].id,
        images[2].id,
        images[3].id,
        images[0].id,
    ]
    assert output.value.metadata == {
        "collect_segments": [
            {
                "input_index": 0,
                "start_index": 0,
                "item_count": 3,
                "source_kind": "sequence",
            },
            {
                "input_index": 1,
                "start_index": 3,
                "item_count": 1,
                "source_kind": "single",
            },
        ]
    }


def test_collect_converts_each_input_to_its_bound_text_type(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "number",
                    "operator_id": "arithmetic.number",
                    "operator_version": 1,
                    "config": {"value": 42},
                },
                {
                    "id": "text",
                    "operator_id": "text.input",
                    "operator_version": 1,
                    "config": {"text": "answer"},
                },
                {
                    "id": "collect",
                    "operator_id": "sequence.collect",
                    "operator_version": 1,
                    "config": {},
                    "input_plugs": [
                        {"id": "number", "port": "items"},
                        {"id": "text", "port": "items"},
                    ],
                    "artifact_type_bindings": _binding("scalar.text"),
                },
            ],
            "edges": [
                {
                    "from_node": "number",
                    "from_port": "value",
                    "to_node": "collect",
                    "to_port": "items",
                    "to_plug": "number",
                    "conversion_path": [
                        {
                            "id": "builtin.scalar.integer_to_text",
                            "version": 1,
                        }
                    ],
                },
                {
                    "from_node": "text",
                    "from_port": "text",
                    "to_node": "collect",
                    "to_port": "items",
                    "to_plug": "text",
                },
            ],
        },
    )

    assert response.status_code == 200
    result = RunResponse.model_validate(response.json())
    output = next(
        run.outputs[0] for run in result.node_runs if run.node_id == "collect"
    )
    assert [
        builtin_client.get(f"/v1/artifacts/{artifact.artifact_id}/content").json()[
            "value"
        ]
        for artifact in output.artifacts
    ] == ["42", "answer"]


@pytest.mark.parametrize(
    ("bindings", "error_fragment"),
    [
        ([], "missing artifact type bindings: T"),
        (_binding("source.page_image", schema_version=99), "unavailable artifact type"),
        (
            [
                {
                    "variable": "U",
                    "artifact_type": {
                        "id": "source.page_image",
                        "schema_version": 1,
                    },
                }
            ],
            "unknown artifact type bindings: U",
        ),
        (
            [
                {
                    "variable": "U",
                    "artifact_type": {
                        "id": "unavailable.type",
                        "schema_version": 1,
                    },
                }
            ],
            "unknown artifact type bindings: U",
        ),
    ],
)
def test_collect_rejects_invalid_type_bindings(
    builtin_client: TestClient,
    bindings: list[dict[str, object]],
    error_fragment: str,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "collect",
                    "operator_id": "sequence.collect",
                    "operator_version": 1,
                    "config": {},
                    "input_plugs": [{"id": "value", "port": "items"}],
                    "artifact_type_bindings": bindings,
                }
            ]
        },
    )

    assert response.status_code == 422
    assert error_fragment in response.json()["detail"]


@pytest.mark.parametrize("schema_version", [True, 1.5])
def test_collect_rejects_non_integer_artifact_type_schema_versions(
    builtin_client: TestClient,
    schema_version: object,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "collect",
                    "operator_id": "sequence.collect",
                    "operator_version": 1,
                    "config": {},
                    "input_plugs": [{"id": "value", "port": "items"}],
                    "artifact_type_bindings": [
                        {
                            "variable": "T",
                            "artifact_type": {
                                "id": "scalar.text",
                                "schema_version": schema_version,
                            },
                        }
                    ],
                }
            ]
        },
    )

    assert response.status_code == 422
    assert "schema_version" in str(response.json())


def test_collect_rejects_an_input_with_a_different_artifact_type(
    builtin_client: TestClient,
) -> None:
    response = builtin_client.post(
        "/v1/runs",
        json={
            "nodes": [
                {
                    "id": "number",
                    "operator_id": "arithmetic.number",
                    "operator_version": 1,
                    "config": {"value": 7},
                },
                {
                    "id": "collect",
                    "operator_id": "sequence.collect",
                    "operator_version": 1,
                    "config": {},
                    "input_plugs": [{"id": "number", "port": "items"}],
                    "artifact_type_bindings": _binding("source.page_image"),
                },
            ],
            "edges": [
                {
                    "from_node": "number",
                    "from_port": "value",
                    "to_node": "collect",
                    "to_port": "items",
                    "to_plug": "number",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert "cannot connect scalar.integer@1 to source.page_image@1" in response.json()[
        "detail"
    ]
