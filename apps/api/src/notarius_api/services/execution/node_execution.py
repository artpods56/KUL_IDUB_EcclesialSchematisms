"""Execution semantics for one compiled logical node."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast
from uuid import UUID

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
)
from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.domain.node_secrets import JsonValue
from notarius_core.nodes import NodeExecutionContext
from notarius_core.plugins import NodeCachePolicy
from notarius_core.ports.node_secrets import NodeSecretResolverPort
from notarius_core.runtime.execution import NodeRuntime
from notarius_core.runtime.invocation import InvocationError, InvocationMode
from notarius_core.runtime.persistence import PersistedNodeOutput

from notarius_api.services.execution.edge_values import EdgeValueResolver
from notarius_api.services.execution.engine import (
    ExecutionTaskRunner,
    PreparedGraphExecution,
)
from notarius_api.services.execution.models import CompiledEdge, CompiledNode


class NodeExecutionService:
    """Resolve inputs and execute the ONCE or MAP semantics of one node."""

    def __init__(
        self,
        *,
        runtime: NodeRuntime,
        edge_values: EdgeValueResolver,
        node_secrets: NodeSecretResolverPort,
    ) -> None:
        self._runtime = runtime
        self._edge_values = edge_values
        self._node_secrets = node_secrets

    async def execute(
        self,
        *,
        execution: PreparedGraphExecution,
        compiled_node: CompiledNode,
        incoming_edges: Sequence[CompiledEdge],
        outputs: Mapping[str, Mapping[str, ArtifactOutputValue]],
        task_runner: ExecutionTaskRunner,
        node_run_id: UUID,
    ) -> dict[str, ArtifactOutputValue]:
        node_request = compiled_node.request
        inputs = await self._edge_values.assemble_inputs(
            compiled_node,
            incoming_edges,
            outputs,
            task_runner.workflow_run_id,
        )
        node_context = NodeExecutionContext(
            workflow_run_id=task_runner.workflow_run_id,
            node_run_id=node_run_id,
            graph_id=execution.graph_id,
            graph_revision=execution.graph_revision,
            secret_graph_id=(
                execution.secret_graph_id
                if node_request.id in execution.secret_node_ids
                else None
            ),
            secret_graph_revision=(
                execution.secret_graph_revision
                if node_request.id in execution.secret_node_ids
                else None
            ),
            node_id=node_request.id,
            module_path=execution.module_path,
        )
        registration = compiled_node.registration
        cache_policy = (
            registration.cache_policy
            if registration is not None
            else NodeCachePolicy.NEVER
        )
        opaque_secret_revisions: dict[str, str] = {}
        if (
            cache_policy is NodeCachePolicy.EXACT
            and registration is not None
            and registration.secret_inputs
        ):
            validated_config = (
                registration.node_class.config_contract.model.model_validate(
                    node_request.config
                ).model_dump(mode="json")
            )
            for secret_input in registration.secret_inputs:
                secret_dependencies = {
                    dependency: cast(JsonValue, validated_config[dependency])
                    for dependency in secret_input.config_dependencies
                }
                opaque_secret_revisions[
                    secret_input.name
                ] = await self._node_secrets.cache_revision(
                    graph_id=node_context.secret_graph_id,
                    graph_revision=node_context.secret_graph_revision,
                    node_id=node_context.node_id,
                    name=secret_input.name,
                    dependencies=secret_dependencies,
                )

        if compiled_node.invocation.mode is InvocationMode.MAP:
            result = await self._run_mapped(
                compiled_node=compiled_node,
                context=node_context,
                inputs=inputs,
                cache_policy=cache_policy,
                opaque_secret_revisions=opaque_secret_revisions,
                task_runner=task_runner,
            )
        else:
            result = await self._runtime.run_node(
                compiled_node.node,
                node_context,
                inputs,
                config=node_request.config,
                artifact_type_bindings=compiled_node.artifact_type_bindings,
                cache_policy=cache_policy,
                opaque_secret_revisions=opaque_secret_revisions,
            )

        node_outputs: dict[str, ArtifactOutputValue] = {}
        if isinstance(result, PersistedNodeOutput):
            for name in compiled_node.resolved_contracts.output_contract.ports:
                value = result.values.get(name)
                if isinstance(value, ArtifactRef | ArtifactRefSequence):
                    node_outputs[name] = value
        return node_outputs

    async def _run_mapped(
        self,
        *,
        compiled_node: CompiledNode,
        context: NodeExecutionContext,
        inputs: Mapping[str, object],
        cache_policy: NodeCachePolicy,
        opaque_secret_revisions: Mapping[str, str],
        task_runner: ExecutionTaskRunner,
    ) -> PersistedNodeOutput:
        node = compiled_node.node
        invocation = compiled_node.invocation
        map_input = invocation.map_input
        if map_input is None:
            raise InvocationError(
                f"Node {node.operator_id!r} MAP invocation requires a map_input"
            )
        raw_sequence = inputs.get(map_input)
        if not isinstance(raw_sequence, ArtifactRefSequence):
            raise InvocationError(
                f"Node {node.operator_id!r} MAP input {map_input!r} expected an "
                f"ArtifactRefSequence, got {type(raw_sequence).__name__}"
            )
        if not raw_sequence.item_refs:
            raise InvocationError(
                f"Node {node.operator_id!r} MAP input {map_input!r} must not be empty"
            )

        input_port = compiled_node.resolved_contracts.input_contract.ports[map_input]
        if not isinstance(input_port.accepts, ArtifactTypeKey):
            raise InvocationError(
                f"Node {node.operator_id!r} MAP input {map_input!r} has an "
                "unresolved artifact type contract"
            )
        sequence_key = ArtifactTypeKey(
            raw_sequence.artifact_type,
            raw_sequence.schema_version,
        )
        if sequence_key != input_port.accepts:
            raise InvocationError(
                f"Node {node.operator_id!r} MAP input {map_input!r} expected "
                f"{input_port.accepts.id}@{input_port.accepts.schema_version}, got "
                f"{sequence_key.id}@{sequence_key.schema_version}"
            )

        output_contract = compiled_node.resolved_contracts.output_contract
        refs_by_output: dict[str, list[ArtifactRef]] = {
            name: [] for name in output_contract.ports
        }
        cache_hits = 0
        cache_misses = 0
        for index, ref in enumerate(raw_sequence.item_refs):
            item_inputs = dict(inputs)
            item_inputs[map_input] = ref

            async def execute_item(item_node_run_id: UUID) -> PersistedNodeOutput:
                item_context = replace(
                    context,
                    node_run_id=item_node_run_id,
                    invocation_index=index,
                )
                return cast(
                    PersistedNodeOutput,
                    await self._runtime.run_node(
                        node,
                        item_context,
                        item_inputs,
                        config=compiled_node.request.config,
                        artifact_type_bindings=compiled_node.artifact_type_bindings,
                        cache_policy=cache_policy,
                        opaque_secret_revisions=opaque_secret_revisions,
                    ),
                )

            try:
                item_result = await task_runner.run_map_item(
                    compiled_node,
                    index,
                    execute_item,
                )
            except Exception as exc:
                raise InvocationError(
                    f"Node {node.operator_id!r} MAP input {map_input!r} failed at "
                    f"item {index} ({ref.artifact_id})"
                ) from exc
            cache_hits += item_result.cache_hits
            cache_misses += item_result.cache_misses
            for name in output_contract.ports:
                value = item_result.values.get(name)
                if not isinstance(value, ArtifactRef):
                    raise InvocationError(
                        f"Node {node.operator_id!r} MAP output {name!r} at item "
                        f"{index} expected an ArtifactRef, got "
                        f"{type(value).__name__}"
                    )
                refs_by_output[name].append(value)

        values: dict[str, object] = {}
        for name, output_port in output_contract.ports.items():
            if not isinstance(output_port.produces, ArtifactTypeKey):
                raise InvocationError(
                    f"Node {node.operator_id!r} MAP output {name!r} has an "
                    "unresolved artifact type contract"
                )
            values[name] = ArtifactRefSequence(
                artifact_type=output_port.produces.id,
                schema_version=output_port.produces.schema_version,
                item_refs=refs_by_output[name],
                ordered=raw_sequence.ordered,
                index_key=raw_sequence.index_key,
                metadata={
                    "invocation_mode": InvocationMode.MAP.value,
                    "map_input": map_input,
                    "source_sequence_id": str(raw_sequence.sequence_id),
                },
            )
        return PersistedNodeOutput(
            values=values,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )


__all__ = ["NodeExecutionService"]
