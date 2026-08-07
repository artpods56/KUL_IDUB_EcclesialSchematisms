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
        context: NodeExecutionContext,
        *,
        artifact_type_bindings: Mapping[str, ArtifactTypeKey] | None = None,
        cache_policy: NodeCachePolicy = NodeCachePolicy.NEVER,
        opaque_secret_revisions: Mapping[str, str] | None = None,
    ) -> "BoundNode[ConfigT, InputT, OutputT]":
        return BoundNode(
            runtime=self,
            node=node,
            context=context,
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
        artifact_type_bindings: Mapping[str, ArtifactTypeKey] | None = None,
        cache_policy: NodeCachePolicy = NodeCachePolicy.NEVER,
        opaque_secret_revisions: Mapping[str, str] | None = None,
    ) -> PersistedNodeOutput | BaseModel:
        effective_bindings = artifact_type_bindings or {}
        resolved_contracts = resolve_node_contracts(
            node,
            effective_bindings,
        )
        raw_config: JsonObject = {} if config is None else config
        validated_config = node.config_contract.model.model_validate(raw_config)

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
                    workspace_id=context.workspace_id,
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
            workspace_id=context.workspace_id,
        )
        run_output = await node.run(
            context,
            config,
            materialized_inputs,
        )
        return run_output, provenance


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
