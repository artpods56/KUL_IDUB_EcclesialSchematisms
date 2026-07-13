from collections.abc import Mapping
from dataclasses import replace
from typing import cast, final

from pydantic import BaseModel

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    JsonObject,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import Node, NodeExecutionContext
from notarius_core.runtime.materialization import (
    InputMaterializer,
    MaterializationProvenance,
)
from notarius_core.runtime.invocation import (
    InvocationError,
    InvocationMode,
    NodeInvocation,
    validate_invocation,
)
from notarius_core.runtime.persistence import OutputPersister, PersistedNodeOutput


class NodeRuntime:
    def __init__(
        self,
        *,
        materializer: InputMaterializer,
        persister: OutputPersister,
    ) -> None:
        self._materializer = materializer
        self._persister = persister

    def bind[
        ConfigT: NodeConfig,
        InputT: NodeInput,
        OutputT: NodeOutput,
    ](
        self,
        node: Node[ConfigT, InputT, OutputT],
        context: NodeExecutionContext | None = None,
    ) -> "BoundNode[ConfigT, InputT, OutputT]":
        return BoundNode(
            runtime=self,
            node=node,
            context=context or NodeExecutionContext(node_id=node.operator_id),
        )

    async def run_node[
        ConfigT: NodeConfig,
        InputT: NodeInput,
        OutputT: NodeOutput,
    ](
        self,
        node: Node[ConfigT, InputT, OutputT],
        context: NodeExecutionContext,
        inputs: Mapping[str, object],
        config: JsonObject | None = None,
        invocation: NodeInvocation | None = None,
    ) -> PersistedNodeOutput | BaseModel:
        effective_invocation = invocation or NodeInvocation()
        validate_invocation(node, effective_invocation)
        raw_config: JsonObject = {} if config is None else config
        validated_config = node.config_contract.model.model_validate(raw_config)
        if effective_invocation.mode is InvocationMode.MAP:
            return await self._run_mapped(
                node=node,
                context=context,
                inputs=inputs,
                config=cast(ConfigT, validated_config),
                invocation=effective_invocation,
            )

        run_output, provenance = await self._invoke(
            node=node,
            context=context,
            inputs=inputs,
            config=cast(ConfigT, validated_config),
        )
        return await self._persister.persist(
            contract=node.output_contract,
            context=context,
            output=run_output,
            provenance=provenance,
        )

    async def _invoke[
        ConfigT: NodeConfig,
        InputT: NodeInput,
        OutputT: NodeOutput,
    ](
        self,
        *,
        node: Node[ConfigT, InputT, OutputT],
        context: NodeExecutionContext,
        inputs: Mapping[str, object],
        config: ConfigT,
    ) -> tuple[OutputT, MaterializationProvenance]:
        materialized_inputs, provenance = await self._materializer.materialize(
            contract=node.input_contract,
            inputs=inputs,
        )
        run_output = await node.run(
            context,
            config,
            cast(InputT, materialized_inputs),
        )
        return run_output, provenance

    async def _run_mapped[
        ConfigT: NodeConfig,
        InputT: NodeInput,
        OutputT: NodeOutput,
    ](
        self,
        *,
        node: Node[ConfigT, InputT, OutputT],
        context: NodeExecutionContext,
        inputs: Mapping[str, object],
        config: ConfigT,
        invocation: NodeInvocation,
    ) -> PersistedNodeOutput:
        map_input = invocation.map_input
        if map_input is None:
            raise InvocationError(
                f"Node {node.operator_id!r} MAP invocation requires a map_input"
            )
        raw_sequence = inputs.get(map_input)
        if not isinstance(raw_sequence, ArtifactRefSequence):
            value_type = type(raw_sequence).__name__
            raise InvocationError(
                f"Node {node.operator_id!r} MAP input {map_input!r} expected an "
                f"ArtifactRefSequence, got {value_type}"
            )
        if not raw_sequence.item_refs:
            raise InvocationError(
                f"Node {node.operator_id!r} MAP input {map_input!r} must not be empty"
            )

        input_port = node.input_contract.ports[map_input]
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

        completed: list[
            tuple[OutputT, MaterializationProvenance, NodeExecutionContext]
        ] = []
        for index, ref in enumerate(raw_sequence.item_refs):
            invocation_inputs = dict(inputs)
            invocation_inputs[map_input] = ref
            invocation_context = replace(context, invocation_index=index)
            try:
                run_output, provenance = await self._invoke(
                    node=node,
                    context=invocation_context,
                    inputs=invocation_inputs,
                    config=config,
                )
            except Exception as exc:
                message = (
                    f"Node {node.operator_id!r} MAP input {map_input!r} failed at "
                    f"item {index} ({ref.artifact_id})"
                )
                raise InvocationError(message) from exc
            completed.append((run_output, provenance, invocation_context))

        refs_by_output: dict[str, list[ArtifactRef]] = {
            name: [] for name in node.output_contract.ports
        }
        for index, (run_output, provenance, invocation_context) in enumerate(completed):
            try:
                persisted = await self._persister.persist(
                    contract=node.output_contract,
                    context=invocation_context,
                    output=run_output,
                    provenance=provenance,
                )
            except Exception as exc:
                message = (
                    f"Node {node.operator_id!r} MAP output persistence failed at "
                    f"item {index}"
                )
                raise InvocationError(message) from exc
            if not isinstance(persisted, PersistedNodeOutput):
                raise InvocationError(
                    f"Node {node.operator_id!r} MAP invocation did not produce "
                    "persisted artifact outputs"
                )
            for name in node.output_contract.ports:
                value = persisted.values.get(name)
                if not isinstance(value, ArtifactRef):
                    raise InvocationError(
                        f"Node {node.operator_id!r} MAP output {name!r} at item "
                        f"{index} expected an ArtifactRef, got "
                        f"{type(value).__name__}"
                    )
                refs_by_output[name].append(value)

        values: dict[str, object] = {}
        for name, output_port in node.output_contract.ports.items():
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
        return PersistedNodeOutput(values=values)


@final
class BoundNode[
    ConfigT: NodeConfig,
    InputT: NodeInput,
    OutputT: NodeOutput,
]:
    def __init__(
        self,
        *,
        runtime: NodeRuntime,
        node: Node[ConfigT, InputT, OutputT],
        context: NodeExecutionContext,
    ) -> None:
        self._runtime = runtime
        self._node = node
        self._context = context

    async def __call__(
        self,
        inputs: Mapping[str, object],
        config: JsonObject | None = None,
        invocation: NodeInvocation | None = None,
    ) -> PersistedNodeOutput | BaseModel:
        return await self._runtime.run_node(
            self._node,
            self._context,
            inputs,
            config=config,
            invocation=invocation,
        )
