from collections.abc import Mapping
from typing import cast, final

from pydantic import BaseModel

from notarius_core.prototype.artifacts import (
    JsonObject,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.prototype.materialization import InputMaterializer
from notarius_core.prototype.nodes import Node, NodeExecutionContext
from notarius_core.prototype.persistence import OutputPersister, PersistedNodeOutput


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
    ) -> PersistedNodeOutput | BaseModel:
        raw_config: JsonObject = {} if config is None else config
        validated_config = node.config_contract.model.model_validate(raw_config)
        materialized_inputs, provenance = await self._materializer.materialize(
            contract=node.input_contract,
            inputs=inputs,
        )
        run_output = await node.run(
            context,
            cast(ConfigT, validated_config),
            cast(InputT, materialized_inputs),
        )
        return await self._persister.persist(
            contract=node.output_contract,
            context=context,
            output=run_output,
            provenance=provenance,
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
    ) -> None:
        self._runtime = runtime
        self._node = node
        self._context = context

    async def __call__(
        self,
        inputs: Mapping[str, object],
        config: JsonObject | None = None,
    ) -> PersistedNodeOutput | BaseModel:
        return await self._runtime.run_node(
            self._node,
            self._context,
            inputs,
            config=config,
        )
