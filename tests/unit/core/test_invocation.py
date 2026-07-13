from collections.abc import Callable
from typing import Annotated, ClassVar, cast, override
from uuid import UUID, uuid4

import pytest

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    ArtifactTypeSpec,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
    PortShape,
    derive_output_contract,
)
from notarius_core.runtime.execution import NodeRuntime
from notarius_core.runtime.invocation import (
    InvocationError,
    InvocationMode,
    NodeInvocation,
    effective_input_shape,
    effective_output_shape,
    map_input_candidates,
    supported_invocation_modes,
    validate_invocation,
)
from notarius_core.runtime.materialization import (
    InputMaterializer,
    MaterializationError,
    MaterializationProvenance,
)
from notarius_core.runtime.persistence import (
    ArtifactOutputWriter,
    ArtifactWriteContext,
    ArtifactWriterRegistry,
    OutputPersister,
    PersistedNodeOutput,
)
from notarius_core.runtime.resolvers import Resolver, ResolverRegistry


INPUT_VALUE = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.input_value", 1),
    title="Test input value",
)
OUTPUT_VALUE = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.output_value", 1),
    title="Test output value",
)
OTHER_VALUE = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.other_value", 1),
    title="Other test value",
)


class IntResolver:
    source = INPUT_VALUE.key
    target = int

    def __init__(self, values: dict[UUID, int]) -> None:
        self._values = values

    async def resolve(self, ref: ArtifactRef) -> int:
        return self._values[ref.artifact_id]


class RecordingWriter:
    artifact_type = OUTPUT_VALUE.key

    def __init__(self, completed_count: Callable[[], int] = lambda: 0) -> None:
        self._completed_count = completed_count
        self.values: list[int] = []
        self.item_indexes: list[int | None] = []
        self.completed_counts: list[int] = []
        self.refs: list[ArtifactRef] = []

    async def write(
        self,
        value: object,
        context: ArtifactWriteContext,
    ) -> ArtifactRef:
        assert isinstance(value, int)
        self.values.append(value)
        self.item_indexes.append(context.item_index)
        self.completed_counts.append(self._completed_count())
        ref = ArtifactRef.from_key(
            artifact_id=uuid4(),
            key=self.artifact_type,
        )
        self.refs.append(ref)
        return ref


class CollectionInput(NodeInput):
    items: Annotated[list[int], InPort(INPUT_VALUE)]


class CollectionOutput(NodeOutput):
    total: Annotated[int, OutPort(OUTPUT_VALUE)]


class CollectionNode(Node[NoConfig, CollectionInput, CollectionOutput]):
    operator_id: ClassVar[str] = "test.collection"
    operator_version: ClassVar[int] = 1

    def __init__(self) -> None:
        self.calls: list[tuple[int | None, list[int]]] = []

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        inputs: CollectionInput,
        /,
    ) -> CollectionOutput:
        self.calls.append((context.invocation_index, list(inputs.items)))
        return CollectionOutput(total=sum(inputs.items))


class ScalarInput(NodeInput):
    item: Annotated[int, InPort(INPUT_VALUE)]
    broadcast: Annotated[int, InPort(INPUT_VALUE)]


class ScalarOutput(NodeOutput):
    value: Annotated[int, OutPort(OUTPUT_VALUE)]


class ScalarNode(Node[NoConfig, ScalarInput, ScalarOutput]):
    operator_id: ClassVar[str] = "test.scalar"
    operator_version: ClassVar[int] = 1

    def __init__(self) -> None:
        self.calls: list[tuple[int | None, int, int]] = []

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        inputs: ScalarInput,
        /,
    ) -> ScalarOutput:
        self.calls.append((context.invocation_index, inputs.item, inputs.broadcast))
        return ScalarOutput(value=inputs.item + inputs.broadcast)


class OptionalDriverInput(NodeInput):
    item: Annotated[int | None, InPort(INPUT_VALUE)] = None


class OptionalDriverNode(Node[NoConfig, OptionalDriverInput, ScalarOutput]):
    operator_id: ClassVar[str] = "test.optional_driver"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: OptionalDriverInput,
        /,
    ) -> ScalarOutput:
        return ScalarOutput(value=0)


class VariadicDriverInput(NodeInput):
    items: Annotated[list[int], InPort(INPUT_VALUE, variadic=True)]


class VariadicDriverNode(Node[NoConfig, VariadicDriverInput, ScalarOutput]):
    operator_id: ClassVar[str] = "test.variadic_driver"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: VariadicDriverInput,
        /,
    ) -> ScalarOutput:
        return ScalarOutput(value=0)


class ManyOutput(NodeOutput):
    values: Annotated[list[int], OutPort(OUTPUT_VALUE)]


class ManyOutputNode(Node[NoConfig, ScalarInput, ManyOutput]):
    operator_id: ClassVar[str] = "test.many_output"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: ScalarInput,
        /,
    ) -> ManyOutput:
        return ManyOutput(values=[inputs.item])


class ExtraOutput(NodeOutput):
    value: Annotated[int, OutPort(OUTPUT_VALUE)]
    diagnostic: str


class ExtraOutputNode(Node[NoConfig, ScalarInput, ExtraOutput]):
    operator_id: ClassVar[str] = "test.extra_output"
    operator_version: ClassVar[int] = 1

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: ScalarInput,
        /,
    ) -> ExtraOutput:
        return ExtraOutput(value=inputs.item, diagnostic="unused")


class PassthroughOutput(NodeOutput):
    value: Annotated[object, OutPort(OUTPUT_VALUE)]


def runtime_with(
    resolver: IntResolver,
    writer: RecordingWriter,
) -> NodeRuntime:
    resolvers = ResolverRegistry()
    resolvers.register(cast(Resolver[object], resolver))
    writers = ArtifactWriterRegistry()
    writers.register(cast(ArtifactOutputWriter, writer))
    return NodeRuntime(
        materializer=InputMaterializer(resolvers),
        persister=OutputPersister(writers),
    )


@pytest.mark.asyncio
async def test_once_invokes_collection_node_once_with_the_whole_sequence() -> None:
    first_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=INPUT_VALUE.key)
    second_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=INPUT_VALUE.key)
    resolver = IntResolver({first_ref.artifact_id: 2, second_ref.artifact_id: 4})
    writer = RecordingWriter()
    runtime = runtime_with(resolver, writer)
    node = CollectionNode()

    result = await runtime.run_node(
        node,
        NodeExecutionContext(node_id="collection"),
        {
            "items": ArtifactRefSequence.from_key(
                key=INPUT_VALUE.key,
                item_refs=[first_ref, second_ref],
            )
        },
    )

    assert node.calls == [(None, [2, 4])]
    assert writer.values == [6]
    assert writer.item_indexes == [None]
    assert isinstance(result, PersistedNodeOutput)
    assert result["total"] == writer.refs[0]


@pytest.mark.asyncio
async def test_map_invokes_in_order_broadcasts_and_aggregates_outputs() -> None:
    first_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=INPUT_VALUE.key)
    second_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=INPUT_VALUE.key)
    broadcast_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=INPUT_VALUE.key)
    resolver = IntResolver(
        {
            first_ref.artifact_id: 2,
            second_ref.artifact_id: 4,
            broadcast_ref.artifact_id: 10,
        }
    )
    node = ScalarNode()
    writer = RecordingWriter(lambda: len(node.calls))
    runtime = runtime_with(resolver, writer)
    source_sequence = ArtifactRefSequence(
        artifact_type=INPUT_VALUE.key.id,
        schema_version=INPUT_VALUE.key.schema_version,
        item_refs=[first_ref, second_ref],
        ordered=False,
        index_key="source_position",
    )
    invocation = NodeInvocation(mode=InvocationMode.MAP, map_input="item")

    result = await runtime.run_node(
        node,
        NodeExecutionContext(node_id="scalar"),
        {"item": source_sequence, "broadcast": broadcast_ref},
        invocation=invocation,
    )

    assert node.calls == [(0, 2, 10), (1, 4, 10)]
    assert writer.values == [12, 14]
    assert writer.item_indexes == [0, 1]
    assert writer.completed_counts == [2, 2]
    assert isinstance(result, PersistedNodeOutput)
    output_sequence = result["value"]
    assert isinstance(output_sequence, ArtifactRefSequence)
    assert output_sequence.item_refs == writer.refs
    assert output_sequence.ordered is False
    assert output_sequence.index_key == "source_position"
    assert output_sequence.metadata == {
        "invocation_mode": "map",
        "map_input": "item",
        "source_sequence_id": str(source_sequence.sequence_id),
    }


def test_invocation_capabilities_and_effective_shapes() -> None:
    invocation = NodeInvocation(mode=InvocationMode.MAP, map_input="item")

    assert map_input_candidates(ScalarNode) == ("item", "broadcast")
    assert supported_invocation_modes(ScalarNode) == (
        InvocationMode.ONCE,
        InvocationMode.MAP,
    )
    assert effective_input_shape(ScalarNode, invocation, "item") is PortShape.MANY
    assert effective_input_shape(ScalarNode, invocation, "broadcast") is PortShape.ONE
    assert effective_output_shape(ScalarNode, invocation, "value") is PortShape.MANY


@pytest.mark.parametrize(
    ("node", "map_input", "message"),
    [
        (ScalarNode, "missing", "does not exist"),
        (CollectionNode, "items", "must have shape 'one'"),
        (OptionalDriverNode, "item", "must be required"),
        (VariadicDriverNode, "items", "cannot be variadic"),
        (ManyOutputNode, "item", "MAP output 'values' must have shape 'one'"),
        (ExtraOutputNode, "item", "non-port fields: diagnostic"),
    ],
)
def test_invalid_map_drivers_and_output_contracts_are_rejected(
    node: object,
    map_input: str,
    message: str,
) -> None:
    with pytest.raises(InvocationError, match=message):
        validate_invocation(
            cast(type[ScalarNode], node),
            NodeInvocation(mode=InvocationMode.MAP, map_input=map_input),
        )


@pytest.mark.asyncio
async def test_map_rejects_empty_sequences_before_invocation() -> None:
    broadcast_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=INPUT_VALUE.key)
    resolver = IntResolver({broadcast_ref.artifact_id: 10})
    writer = RecordingWriter()
    runtime = runtime_with(resolver, writer)
    node = ScalarNode()

    with pytest.raises(InvocationError, match="MAP input 'item' must not be empty"):
        await runtime.run_node(
            node,
            NodeExecutionContext(node_id="scalar"),
            {
                "item": ArtifactRefSequence.from_key(
                    key=INPUT_VALUE.key,
                    item_refs=[],
                ),
                "broadcast": broadcast_ref,
            },
            invocation=NodeInvocation(
                mode=InvocationMode.MAP,
                map_input="item",
            ),
        )

    assert node.calls == []
    assert writer.values == []


@pytest.mark.asyncio
async def test_map_error_preserves_item_context_and_original_cause() -> None:
    missing_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=INPUT_VALUE.key)
    broadcast_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=INPUT_VALUE.key)
    resolver = IntResolver({broadcast_ref.artifact_id: 10})
    runtime = runtime_with(resolver, RecordingWriter())

    with pytest.raises(
        InvocationError,
        match=f"item 0 \\({missing_ref.artifact_id}\\)",
    ) as raised:
        await runtime.run_node(
            ScalarNode(),
            NodeExecutionContext(node_id="scalar"),
            {
                "item": ArtifactRefSequence.from_key(
                    key=INPUT_VALUE.key,
                    item_refs=[missing_ref],
                ),
                "broadcast": broadcast_ref,
            },
            invocation=NodeInvocation(
                mode=InvocationMode.MAP,
                map_input="item",
            ),
        )

    assert isinstance(raised.value.__cause__, KeyError)


@pytest.mark.asyncio
async def test_materializer_rejects_wrong_key_empty_sequence() -> None:
    materializer = InputMaterializer(ResolverRegistry())

    with pytest.raises(
        MaterializationError,
        match="expected test.input_value@1, got test.other_value@1",
    ):
        await materializer.materialize(
            CollectionNode.input_contract,
            {
                "items": ArtifactRefSequence.from_key(
                    key=OTHER_VALUE.key,
                    item_refs=[],
                )
            },
        )


def test_registries_reject_duplicate_contracts() -> None:
    resolver = cast(Resolver[object], IntResolver({}))
    resolvers = ResolverRegistry([resolver])
    with pytest.raises(ValueError, match="Resolver already registered"):
        resolvers.register(resolver)

    writer = cast(ArtifactOutputWriter, RecordingWriter())
    writers = ArtifactWriterRegistry([writer])
    with pytest.raises(ValueError, match="Output writer already registered"):
        writers.register(writer)


@pytest.mark.asyncio
async def test_output_persister_rejects_passthrough_key_and_shape_mismatches() -> None:
    persister = OutputPersister(ArtifactWriterRegistry())
    contract = derive_output_contract(PassthroughOutput)
    context = NodeExecutionContext(node_id="passthrough")
    provenance = MaterializationProvenance(refs_by_input={})
    wrong_key_ref = ArtifactRef.from_key(
        artifact_id=uuid4(),
        key=OTHER_VALUE.key,
    )

    with pytest.raises(
        RuntimeError,
        match="expected test.output_value@1, got test.other_value@1",
    ):
        await persister.persist(
            contract,
            context,
            PassthroughOutput(value=wrong_key_ref),
            provenance,
        )

    correct_ref = ArtifactRef.from_key(
        artifact_id=uuid4(),
        key=OUTPUT_VALUE.key,
    )
    with pytest.raises(
        RuntimeError,
        match="expected an ArtifactRef, got ArtifactRefSequence",
    ):
        await persister.persist(
            contract,
            context,
            PassthroughOutput(
                value=ArtifactRefSequence.from_key(
                    key=OUTPUT_VALUE.key,
                    item_refs=[correct_ref],
                )
            ),
            provenance,
        )
