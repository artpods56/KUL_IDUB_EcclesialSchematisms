"""Direct in-process adapter for prepared graph execution."""

from collections.abc import Mapping
from typing import final
from uuid import UUID, uuid4

from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.runtime.persistence import PersistedNodeOutput

from .coordinator import GraphExecutionCoordinator
from .engine import (
    MapItemExecutionOperation,
    NodeExecutionOperation,
    PreparedGraphExecution,
)
from .models import (
    CompiledNode,
    GraphExecutionResult,
)


@final
class InlineExecutionTaskRunner:
    """Call execution operations directly with locally generated run IDs."""

    def __init__(self) -> None:
        self._workflow_run_id = uuid4()

    @property
    def workflow_run_id(self) -> UUID:
        return self._workflow_run_id

    async def run_node(
        self,
        compiled_node: CompiledNode,
        upstream_node_ids: frozenset[str],
        operation: NodeExecutionOperation,
        /,
    ) -> Mapping[str, ArtifactOutputValue]:
        del compiled_node, upstream_node_ids
        return await operation(uuid4())

    async def run_map_item(
        self,
        compiled_node: CompiledNode,
        index: int,
        operation: MapItemExecutionOperation,
        /,
    ) -> PersistedNodeOutput:
        del compiled_node, index
        return await operation(uuid4())


@final
class InlineExecutionEngine:
    def __init__(self, *, coordinator: GraphExecutionCoordinator) -> None:
        self._coordinator = coordinator

    async def execute(
        self,
        execution: PreparedGraphExecution,
        /,
    ) -> GraphExecutionResult:
        return await self._coordinator.execute(
            execution,
            InlineExecutionTaskRunner(),
        )


__all__ = ["InlineExecutionEngine", "InlineExecutionTaskRunner"]
