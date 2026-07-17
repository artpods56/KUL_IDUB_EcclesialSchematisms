from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast, final

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
from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.nodes import (
    InputContract,
    Node,
    NodeExecutionContext,
    OutputContract,
    PortShape,
    resolve_node_contracts,
)
from notarius_core.plugins import NodeCachePolicy
from notarius_core.runtime.invocation_cache import (
    DisabledInvocationCache,
    InvocationCachePort,
    invocation_cache_key,
)
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
        invocation_cache: InvocationCachePort | None = None,
    ) -> None:
        self._materializer = materializer
        self._persister = persister
        self._invocation_cache = invocation_cache or DisabledInvocationCache()

    def bind[
        ConfigT: NodeConfig,
        InputT: NodeInput,
        OutputT: NodeOutput,
    ](
        self,
        node: Node[ConfigT, InputT, OutputT],
        context: NodeExecutionContext | None = None,
        *,
        artifact_type_bindings: Mapping[str, ArtifactTypeKey] | None = None,
        cache_policy: NodeCachePolicy = NodeCachePolicy.NEVER,
        opaque_secret_revisions: Mapping[str, str] | None = None,
    ) -> "BoundNode[ConfigT, InputT, OutputT]":
        return BoundNode(
            runtime=self,
            node=node,
            context=context or NodeExecutionContext(node_id=node.operator_id),
            artifact_type_bindings=artifact_type_bindings,
            cache_policy=cache_policy,
            opaque_secret_revisions=opaque_secret_revisions,
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
        artifact_type_bindings: Mapping[str, ArtifactTypeKey] | None = None,
        cache_policy: NodeCachePolicy = NodeCachePolicy.NEVER,
        opaque_secret_revisions: Mapping[str, str] | None = None,
    ) -> PersistedNodeOutput | BaseModel:
        effective_bindings = artifact_type_bindings or {}
        resolved_contracts = resolve_node_contracts(
            node,
            effective_bindings,
        )
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
                input_contract=cast(
                    InputContract[InputT],
                    resolved_contracts.input_contract,
                ),
                output_contract=cast(
                    OutputContract[OutputT],
                    resolved_contracts.output_contract,
                ),
                artifact_type_bindings=effective_bindings,
                cache_policy=cache_policy,
                opaque_secret_revisions=opaque_secret_revisions or {},
            )

        cache_key: str | None = None
        cache_misses = 0
        if cache_policy is NodeCachePolicy.EXACT:
            cache_misses = 1
            if _contract_supports_cache(resolved_contracts.output_contract):
                cache_key = invocation_cache_key(
                    node=node,
                    context=context,
                    inputs=inputs,
                    config=validated_config,
                    invocation=effective_invocation,
                    artifact_type_bindings=effective_bindings,
                    opaque_secret_revisions=opaque_secret_revisions or {},
                )
            if cache_key is not None:
                cached_entry = await self._invocation_cache.get(cache_key)
                if cached_entry is not None:
                    cached_values = _validated_cached_outputs(
                        resolved_contracts.output_contract,
                        cached_entry.outputs,
                    )
                    if cached_values is not None:
                        return PersistedNodeOutput(
                            values=cast(dict[str, object], cached_values),
                            cache_hits=1,
                        )
                    await self._invocation_cache.remove_if_current(
                        cache_key,
                        cached_entry.generation,
                    )

        run_output, provenance = await self._invoke(
            node=node,
            context=context,
            inputs=inputs,
            config=cast(ConfigT, validated_config),
            input_contract=cast(
                InputContract[InputT],
                resolved_contracts.input_contract,
            ),
        )
        persisted = await self._persister.persist(
            contract=resolved_contracts.output_contract,
            context=context,
            output=run_output,
            provenance=provenance,
        )
        if not isinstance(persisted, PersistedNodeOutput):
            return persisted

        cache_outputs = _validated_persisted_outputs(
            resolved_contracts.output_contract,
            persisted,
        )
        if cache_key is not None and cache_outputs is not None:
            await self._invocation_cache.put_if_absent(
                InvocationCacheEntry(
                    key_sha256=cache_key,
                    outputs=cache_outputs,
                )
            )
        return replace(persisted, cache_misses=cache_misses)

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
        input_contract: InputContract[InputT],
    ) -> tuple[OutputT, MaterializationProvenance]:
        materialized_inputs, provenance = await self._materializer.materialize(
            contract=input_contract,
            inputs=inputs,
        )
        run_output = await node.run(
            context,
            config,
            materialized_inputs,
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
        input_contract: InputContract[InputT],
        output_contract: OutputContract[OutputT],
        artifact_type_bindings: Mapping[str, ArtifactTypeKey],
        cache_policy: NodeCachePolicy,
        opaque_secret_revisions: Mapping[str, str],
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

        input_port = input_contract.ports[map_input]
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

        refs_by_output: dict[str, list[ArtifactRef]] = {
            name: [] for name in output_contract.ports
        }
        cache_hits = 0
        cache_misses = 0
        for index, ref in enumerate(raw_sequence.item_refs):
            invocation_inputs = dict(inputs)
            invocation_inputs[map_input] = ref
            invocation_context = replace(context, invocation_index=index)
            cache_key: str | None = None
            if cache_policy is NodeCachePolicy.EXACT:
                cache_misses += 1
                if _contract_supports_cache(output_contract):
                    cache_key = invocation_cache_key(
                        node=node,
                        context=invocation_context,
                        inputs=invocation_inputs,
                        config=config,
                        invocation=invocation,
                        artifact_type_bindings=artifact_type_bindings,
                        opaque_secret_revisions=opaque_secret_revisions,
                    )
                if cache_key is not None:
                    cached_entry = await self._invocation_cache.get(cache_key)
                    if cached_entry is not None:
                        cached_values = _validated_cached_outputs(
                            output_contract,
                            cached_entry.outputs,
                        )
                        if cached_values is not None:
                            for name in output_contract.ports:
                                cached_ref = cached_values[name]
                                if not isinstance(cached_ref, ArtifactRef):
                                    raise InvocationError(
                                        f"Node {node.operator_id!r} MAP cached output "
                                        f"{name!r} at item {index} expected an "
                                        "ArtifactRef"
                                    )
                                refs_by_output[name].append(cached_ref)
                            cache_hits += 1
                            cache_misses -= 1
                            continue
                        await self._invocation_cache.remove_if_current(
                            cache_key,
                            cached_entry.generation,
                        )
            try:
                run_output, provenance = await self._invoke(
                    node=node,
                    context=invocation_context,
                    inputs=invocation_inputs,
                    config=config,
                    input_contract=input_contract,
                )
            except Exception as exc:
                message = (
                    f"Node {node.operator_id!r} MAP input {map_input!r} failed at "
                    f"item {index} ({ref.artifact_id})"
                )
                raise InvocationError(message) from exc
            try:
                persisted = await self._persister.persist(
                    contract=output_contract,
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
            item_outputs: dict[str, ArtifactOutputValue] = {}
            for name in output_contract.ports:
                value = persisted.values.get(name)
                if not isinstance(value, ArtifactRef):
                    raise InvocationError(
                        f"Node {node.operator_id!r} MAP output {name!r} at item "
                        f"{index} expected an ArtifactRef, got "
                        f"{type(value).__name__}"
                    )
                refs_by_output[name].append(value)
                item_outputs[name] = value
            if cache_key is not None:
                await self._invocation_cache.put_if_absent(
                    InvocationCacheEntry(
                        key_sha256=cache_key,
                        outputs=item_outputs,
                    )
                )

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
        artifact_type_bindings: Mapping[str, ArtifactTypeKey] | None,
        cache_policy: NodeCachePolicy,
        opaque_secret_revisions: Mapping[str, str] | None,
    ) -> None:
        self._runtime = runtime
        self._node = node
        self._context = context
        self._artifact_type_bindings = dict(artifact_type_bindings or {})
        self._cache_policy = cache_policy
        self._opaque_secret_revisions = dict(opaque_secret_revisions or {})

    async def __call__(
        self,
        inputs: Mapping[str, object],
        config: JsonObject | None = None,
        invocation: NodeInvocation | None = None,
        artifact_type_bindings: Mapping[str, ArtifactTypeKey] | None = None,
    ) -> PersistedNodeOutput | BaseModel:
        effective_bindings = (
            self._artifact_type_bindings
            if artifact_type_bindings is None
            else artifact_type_bindings
        )
        return await self._runtime.run_node(
            self._node,
            self._context,
            inputs,
            config=config,
            invocation=invocation,
            artifact_type_bindings=effective_bindings,
            cache_policy=self._cache_policy,
            opaque_secret_revisions=self._opaque_secret_revisions,
        )


def _contract_supports_cache(contract: OutputContract[Any]) -> bool:
    return bool(contract.ports) and set(contract.model.model_fields) == set(
        contract.ports
    )


def _validated_persisted_outputs(
    contract: OutputContract[Any],
    persisted: PersistedNodeOutput,
) -> dict[str, ArtifactOutputValue] | None:
    outputs: dict[str, ArtifactOutputValue] = {}
    for name in contract.ports:
        value = persisted.values.get(name)
        if not isinstance(value, ArtifactRef | ArtifactRefSequence):
            return None
        outputs[name] = value
    return _validated_cached_outputs(contract, outputs)


def _validated_cached_outputs(
    contract: OutputContract[Any],
    outputs: Mapping[str, ArtifactOutputValue],
) -> dict[str, ArtifactOutputValue] | None:
    if not _contract_supports_cache(contract) or set(outputs) != set(contract.ports):
        return None
    validated: dict[str, ArtifactOutputValue] = {}
    for name, spec in contract.ports.items():
        if not isinstance(spec.produces, ArtifactTypeKey):
            return None
        value = outputs[name]
        if isinstance(value, ArtifactRef):
            if spec.shape is not PortShape.ONE or value.key() != spec.produces:
                return None
        elif (
            spec.shape is not PortShape.MANY
            or value.artifact_type != spec.produces.id
            or value.schema_version != spec.produces.schema_version
        ):
            return None
        validated[name] = value.model_copy(deep=True)
    return validated
