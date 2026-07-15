from uuid import uuid4

import pytest

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
)
from notarius_core.nodes import (
    ArtifactTypeVariable,
    NodeContractResolutionError,
    NodeExecutionContext,
    PortShape,
)
from notarius_core.operators.sequences import CollectNode, SEQUENCES
from notarius_core.plugins import PluginRegistry
from notarius_core.runtime.execution import NodeRuntime
from notarius_core.runtime.materialization import (
    InputMaterializer,
    MaterializationError,
)
from notarius_core.runtime.persistence import (
    ArtifactWriterRegistry,
    OutputPersister,
    PersistedNodeOutput,
)
from notarius_core.runtime.resolvers import ResolverRegistry


VALUE_TYPE = ArtifactTypeKey("example.value", 1)
OTHER_TYPE = ArtifactTypeKey("example.other", 1)


def _runtime() -> NodeRuntime:
    return NodeRuntime(
        materializer=InputMaterializer(ResolverRegistry()),
        persister=OutputPersister(ArtifactWriterRegistry()),
    )


def test_collect_declares_one_homogeneous_generic_contract() -> None:
    input_port = CollectNode.input_contract.ports["items"]
    output_port = CollectNode.output_contract.ports["items"]

    assert isinstance(input_port.accepts, ArtifactTypeVariable)
    assert input_port.accepts.name == "T"
    assert output_port.produces is input_port.accepts
    assert input_port.accepted_shapes == (PortShape.ONE, PortShape.MANY)
    assert input_port.variadic is True
    assert input_port.instance_plugs is True
    assert output_port.shape is PortShape.MANY


def test_sequence_plugin_registers_generic_collect_without_artifact_types() -> None:
    registry = PluginRegistry()
    registry.install(SEQUENCES)

    registry.freeze()

    assert SEQUENCES.slug == "builtin.sequence"
    assert SEQUENCES.title == "Sequence"
    assert SEQUENCES.artifact_types == ()
    assert tuple(registration.key for registration in registry.nodes) == (
        ("sequence.collect", 1),
    )
    assert registry.nodes[0].title == "Collect"


@pytest.mark.asyncio
async def test_collect_flattens_one_level_in_plug_order_and_preserves_refs() -> None:
    first_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=VALUE_TYPE)
    second_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=VALUE_TYPE)
    third_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=VALUE_TYPE)
    sequence = ArtifactRefSequence(
        artifact_type=VALUE_TYPE.id,
        schema_version=VALUE_TYPE.schema_version,
        item_refs=[second_ref, third_ref],
        ordered=False,
        index_key="source_index",
    )
    empty_sequence = ArtifactRefSequence.from_key(
        key=VALUE_TYPE,
        item_refs=[],
    )

    result = await _runtime().run_node(
        CollectNode(),
        NodeExecutionContext(node_id="collect"),
        {"items": [first_ref, sequence, empty_sequence]},
        artifact_type_bindings={"T": VALUE_TYPE},
    )

    assert isinstance(result, PersistedNodeOutput)
    output = result["items"]
    assert isinstance(output, ArtifactRefSequence)
    assert output.artifact_type == VALUE_TYPE.id
    assert output.schema_version == VALUE_TYPE.schema_version
    assert output.item_refs == [first_ref, second_ref, third_ref]
    assert output.ordered is False
    assert output.metadata == {
        "collect_segments": [
            {
                "input_index": 0,
                "start_index": 0,
                "item_count": 1,
                "source_kind": "single",
            },
            {
                "input_index": 1,
                "start_index": 1,
                "item_count": 2,
                "source_kind": "sequence",
            },
            {
                "input_index": 2,
                "start_index": 3,
                "item_count": 0,
                "source_kind": "sequence",
            },
        ]
    }


@pytest.mark.asyncio
async def test_collect_derives_type_from_empty_sequence_container() -> None:
    result = await _runtime().bind(
        CollectNode(),
        artifact_type_bindings={"T": VALUE_TYPE},
    )(
        {
            "items": [
                ArtifactRefSequence.from_key(
                    key=VALUE_TYPE,
                    item_refs=[],
                )
            ]
        }
    )

    assert isinstance(result, PersistedNodeOutput)
    output = result["items"]
    assert isinstance(output, ArtifactRefSequence)
    assert output.item_refs == []
    assert output.artifact_type == VALUE_TYPE.id
    assert output.schema_version == VALUE_TYPE.schema_version
    assert output.metadata["collect_segments"] == [
        {
            "input_index": 0,
            "start_index": 0,
            "item_count": 0,
            "source_kind": "sequence",
        }
    ]


@pytest.mark.asyncio
async def test_collect_requires_binding_and_validates_input_against_it() -> None:
    ref = ArtifactRef.from_key(artifact_id=uuid4(), key=VALUE_TYPE)
    runtime = _runtime()

    with pytest.raises(
        NodeContractResolutionError,
        match="missing artifact type bindings: T",
    ):
        await runtime.run_node(
            CollectNode(),
            NodeExecutionContext(node_id="collect"),
            {"items": [ref]},
        )

    with pytest.raises(
        MaterializationError,
        match="expected example.other@1, got example.value@1",
    ):
        await runtime.run_node(
            CollectNode(),
            NodeExecutionContext(node_id="collect"),
            {"items": [ref]},
            artifact_type_bindings={"T": OTHER_TYPE},
        )
