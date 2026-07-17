"""Behavioral contracts for direct in-process graph execution."""

from collections.abc import Mapping, Sequence
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
from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
    resolve_node_contracts,
)
from notarius_core.plugins import NodeCachePolicy, NodeRegistration
from notarius_core.ports.node_secrets import UnavailableNodeSecretResolver
from notarius_core.runtime.execution import NodeRuntime
from notarius_core.runtime.invocation import InvocationMode, NodeInvocation
from notarius_core.runtime.invocation_cache import InvocationCachePort
from notarius_core.runtime.materialization import InputMaterializer
from notarius_core.runtime.persistence import (
    ArtifactOutputWriter,
    ArtifactWriteContext,
    ArtifactWriterRegistry,
    OutputPersister,
)
from notarius_core.runtime.resolvers import Resolver, ResolverRegistry

from notarius_api.schemas.workbench import RunEdgeRequest, RunNodeRequest
from notarius_api.services.execution.coordinator import GraphExecutionCoordinator
from notarius_api.services.execution.edge_values import EdgeValueResolver
from notarius_api.services.execution.engine import PreparedGraphExecution
from notarius_api.services.execution.errors import GraphExecutionError
from notarius_api.services.execution.inline import InlineExecutionEngine
from notarius_api.services.execution.models import (
    CompiledEdge,
    CompiledGraph,
    CompiledNode,
)
from notarius_api.services.execution.node_execution import NodeExecutionService


VALUE = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.local_execution.value", 1),
    title="Local execution value",
)


class IntegerResolver:
    source = VALUE.key
    target = int

    def __init__(self, values: Mapping[UUID, int]) -> None:
        self._values = dict(values)

    async def resolve(self, ref: ArtifactRef) -> int:
        return self._values[ref.artifact_id]


class RecordingWriter:
    artifact_type = VALUE.key

    def __init__(self) -> None:
        self.values: list[int] = []
        self.item_indexes: list[int | None] = []
        self.refs: list[ArtifactRef] = []

    async def write(
        self,
        value: object,
        context: ArtifactWriteContext,
    ) -> ArtifactRef:
        assert isinstance(value, int)
        ref = ArtifactRef.from_key(
            artifact_id=uuid4(),
            key=self.artifact_type,
            content_hash=f"{value:064x}",
        )
        self.values.append(value)
        self.item_indexes.append(context.item_index)
        self.refs.append(ref)
        return ref


class AddInput(NodeInput):
    item: Annotated[int, InPort(VALUE)]
    broadcast: Annotated[int, InPort(VALUE)]


class AddOutput(NodeOutput):
    value: Annotated[int, OutPort(VALUE)]


class AddNode(Node[NoConfig, AddInput, AddOutput]):
    operator_id: ClassVar[str] = "test.local_execution.add"
    operator_version: ClassVar[int] = 1

    def __init__(self) -> None:
        self.calls: list[tuple[int | None, int, int]] = []

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        inputs: AddInput,
        /,
    ) -> AddOutput:
        self.calls.append((context.invocation_index, inputs.item, inputs.broadcast))
        return AddOutput(value=inputs.item + inputs.broadcast)


class MemoryInvocationCache(InvocationCachePort):
    def __init__(self) -> None:
        self.entries: dict[str, InvocationCacheEntry] = {}

    async def get(self, key_sha256: str) -> InvocationCacheEntry | None:
        return self.entries.get(key_sha256)

    async def put_if_absent(self, entry: InvocationCacheEntry) -> bool:
        if entry.key_sha256 in self.entries:
            return False
        self.entries[entry.key_sha256] = entry
        return True

    async def remove_if_current(
        self,
        key_sha256: str,
        generation: UUID,
    ) -> bool:
        entry = self.entries.get(key_sha256)
        if entry is None or entry.generation != generation:
            return False
        del self.entries[key_sha256]
        return True


class StubEdgeValueResolver:
    def __init__(self, inputs_by_node: Mapping[str, Mapping[str, object]]) -> None:
        self.inputs_by_node = {
            node_id: dict(node_inputs)
            for node_id, node_inputs in inputs_by_node.items()
        }
        self.calls: list[str] = []

    async def assemble_inputs(
        self,
        compiled_node: CompiledNode,
        incoming_edges: Sequence[CompiledEdge],
        outputs: dict[str, dict[str, ArtifactOutputValue]],
        workflow_run_id: UUID,
    ) -> dict[str, object]:
        del incoming_edges, outputs, workflow_run_id
        node_id = compiled_node.request.id
        self.calls.append(node_id)
        return dict(self.inputs_by_node[node_id])


def _runtime(
    resolver: IntegerResolver,
    writer: RecordingWriter,
    cache: InvocationCachePort | None = None,
) -> NodeRuntime:
    resolvers = ResolverRegistry([cast(Resolver[object], resolver)])
    writers = ArtifactWriterRegistry([cast(ArtifactOutputWriter, writer)])
    return NodeRuntime(
        materializer=InputMaterializer(resolvers),
        persister=OutputPersister(writers),
        invocation_cache=cache,
    )


def _compiled_add(
    *,
    node_id: str,
    node: AddNode,
    invocation: NodeInvocation,
    cache_policy: NodeCachePolicy = NodeCachePolicy.NEVER,
) -> CompiledNode:
    return CompiledNode(
        request=RunNodeRequest(
            id=node_id,
            operator_id=node.operator_id,
            operator_version=node.operator_version,
        ),
        node=node,
        registration=NodeRegistration(
            node_class=AddNode,
            factory=None,
            cache_policy=cache_policy,
        ),
        resolved_contracts=resolve_node_contracts(node, {}),
        invocation=invocation,
        artifact_type_bindings={},
    )


@pytest.mark.asyncio
async def test_inline_map_reuses_cache_and_preserves_sequence_envelope() -> None:
    first_ref = ArtifactRef.from_key(
        artifact_id=uuid4(),
        key=VALUE.key,
        content_hash="1" * 64,
    )
    second_ref = ArtifactRef.from_key(
        artifact_id=uuid4(),
        key=VALUE.key,
        content_hash="2" * 64,
    )
    broadcast_ref = ArtifactRef.from_key(
        artifact_id=uuid4(),
        key=VALUE.key,
        content_hash="3" * 64,
    )
    resolver = IntegerResolver(
        {
            first_ref.artifact_id: 2,
            second_ref.artifact_id: 4,
            broadcast_ref.artifact_id: 10,
        }
    )
    writer = RecordingWriter()
    node = AddNode()
    compiled_node = _compiled_add(
        node_id="mapped",
        node=node,
        invocation=NodeInvocation(mode=InvocationMode.MAP, map_input="item"),
        cache_policy=NodeCachePolicy.EXACT,
    )
    plan = CompiledGraph(nodes=(compiled_node,), edges=(), pinned_outputs={})
    edge_values = StubEdgeValueResolver(
        {
            "mapped": {
                "item": ArtifactRefSequence.from_key(
                    key=VALUE.key,
                    item_refs=[first_ref],
                ),
                "broadcast": broadcast_ref,
            }
        }
    )
    engine = InlineExecutionEngine(
        coordinator=GraphExecutionCoordinator(
            node_execution=NodeExecutionService(
                runtime=_runtime(resolver, writer, MemoryInvocationCache()),
                edge_values=cast(EdgeValueResolver, edge_values),
                node_secrets=UnavailableNodeSecretResolver(),
            )
        )
    )

    first = await engine.execute(
        PreparedGraphExecution(
            plan=plan,
            initial_outputs={},
            graph_id=None,
            graph_revision=None,
            secret_graph_id=None,
            secret_graph_revision=None,
            secret_node_ids=frozenset(),
            module_path=(),
            raise_node_errors=False,
        )
    )
    second_source = ArtifactRefSequence(
        artifact_type=VALUE.key.id,
        schema_version=VALUE.key.schema_version,
        item_refs=[first_ref, second_ref],
        ordered=False,
        index_key="source_position",
    )
    edge_values.inputs_by_node["mapped"]["item"] = second_source
    second = await engine.execute(
        PreparedGraphExecution(
            plan=plan,
            initial_outputs={},
            graph_id=None,
            graph_revision=None,
            secret_graph_id=None,
            secret_graph_revision=None,
            secret_node_ids=frozenset(),
            module_path=(),
            raise_node_errors=False,
        )
    )

    assert first.status == "succeeded"
    assert second.status == "succeeded"
    first_output = first.node_results[0].outputs["value"]
    second_output = second.node_results[0].outputs["value"]
    assert isinstance(first_output, ArtifactRefSequence)
    assert isinstance(second_output, ArtifactRefSequence)
    assert second_output.item_refs == [first_output.item_refs[0], writer.refs[1]]
    assert second_output.ordered is False
    assert second_output.index_key == "source_position"
    assert second_output.metadata == {
        "invocation_mode": "map",
        "map_input": "item",
        "source_sequence_id": str(second_source.sequence_id),
    }
    assert node.calls == [(0, 2, 10), (1, 4, 10)]
    assert writer.values == [12, 14]
    assert writer.item_indexes == [0, 1]


@pytest.mark.asyncio
async def test_inline_map_failure_skips_dependents_and_preserves_cause() -> None:
    missing_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=VALUE.key)
    broadcast_ref = ArtifactRef.from_key(artifact_id=uuid4(), key=VALUE.key)
    resolver = IntegerResolver({broadcast_ref.artifact_id: 10})
    writer = RecordingWriter()
    failed_node = _compiled_add(
        node_id="failed",
        node=AddNode(),
        invocation=NodeInvocation(mode=InvocationMode.MAP, map_input="item"),
    )
    downstream_node = _compiled_add(
        node_id="downstream",
        node=AddNode(),
        invocation=NodeInvocation(),
    )
    dependency = CompiledEdge(
        request=RunEdgeRequest(
            from_node="failed",
            from_port="value",
            to_node="downstream",
            to_port="item",
        ),
        projection=None,
        conversion_path=(),
    )
    plan = CompiledGraph(
        nodes=(failed_node, downstream_node),
        edges=(dependency,),
        pinned_outputs={},
    )
    edge_values = StubEdgeValueResolver(
        {
            "failed": {
                "item": ArtifactRefSequence.from_key(
                    key=VALUE.key,
                    item_refs=[missing_ref],
                ),
                "broadcast": broadcast_ref,
            },
            "downstream": {
                "item": broadcast_ref,
                "broadcast": broadcast_ref,
            },
        }
    )
    engine = InlineExecutionEngine(
        coordinator=GraphExecutionCoordinator(
            node_execution=NodeExecutionService(
                runtime=_runtime(resolver, writer),
                edge_values=cast(EdgeValueResolver, edge_values),
                node_secrets=UnavailableNodeSecretResolver(),
            )
        )
    )

    result = await engine.execute(
        PreparedGraphExecution(
            plan=plan,
            initial_outputs={},
            graph_id=None,
            graph_revision=None,
            secret_graph_id=None,
            secret_graph_revision=None,
            secret_node_ids=frozenset(),
            module_path=(),
            raise_node_errors=False,
        )
    )

    assert result.status == "failed"
    assert [node_result.status for node_result in result.node_results] == [
        "failed",
        "skipped",
    ]
    error = result.node_results[0].error
    assert error is not None
    assert (
        f"InvocationError: Node 'test.local_execution.add' MAP input 'item' "
        f"failed at item 0 ({missing_ref.artifact_id})" in error
    )
    assert "caused by KeyError" in error
    assert edge_values.calls == ["failed"]

    with pytest.raises(
        GraphExecutionError,
        match="nested graph node 'failed' \\(test.local_execution.add@1\\) failed",
    ) as raised:
        await engine.execute(
            PreparedGraphExecution(
                plan=plan,
                initial_outputs={},
                graph_id=None,
                graph_revision=None,
                secret_graph_id=None,
                secret_graph_revision=None,
                secret_node_ids=frozenset(),
                module_path=(),
                raise_node_errors=True,
            )
        )

    assert raised.value.__cause__ is not None
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert raised.value.__cause__.__cause__ is not None
    assert isinstance(raised.value.__cause__.__cause__, KeyError)
