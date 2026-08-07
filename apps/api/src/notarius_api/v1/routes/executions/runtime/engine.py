from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol
from uuid import UUID

from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.runtime.persistence import PersistedNodeOutput

from .models import (
    CompiledGraph,
    CompiledNode,
    GraphExecutionResult,
)
from .control import RunExecutionControl


@dataclass(frozen=True, slots=True)
class PreparedGraphExecution:
    """Validated, compiled inputs required to execute one graph run."""

    plan: CompiledGraph
    initial_outputs: Mapping[str, Mapping[str, ArtifactOutputValue]]
    workspace_id: UUID
    graph_id: UUID | None
    graph_revision: int | None
    secret_graph_id: UUID | None
    secret_graph_revision: int | None
    secret_node_ids: frozenset[str]
    module_path: tuple[str, ...]
    raise_node_errors: bool
    node_path: tuple[str, ...] = ()
    invocation_path: tuple[int, ...] = ()
    control: RunExecutionControl | None = None

    def __post_init__(self) -> None:
        copied_outputs = {
            node_id: MappingProxyType(
                {
                    port: value.model_copy(deep=True)
                    for port, value in node_outputs.items()
                }
            )
            for node_id, node_outputs in self.initial_outputs.items()
        }
        object.__setattr__(
            self,
            "initial_outputs",
            MappingProxyType(copied_outputs),
        )
        object.__setattr__(self, "secret_node_ids", frozenset(self.secret_node_ids))
        object.__setattr__(self, "module_path", tuple(self.module_path))
        object.__setattr__(self, "node_path", tuple(self.node_path))
        object.__setattr__(self, "invocation_path", tuple(self.invocation_path))


type NodeExecutionOperation = Callable[
    [UUID],
    Awaitable[Mapping[str, ArtifactOutputValue]],
]
type MapItemExecutionOperation = Callable[[UUID], Awaitable[PersistedNodeOutput]]


class ExecutionTaskRunner(Protocol):
    """Per-execution adapter for logical-node and scalar MAP-item tasks.

    ``run_map_item`` calls for one logical node may overlap on the same event loop.
    """

    @property
    def workflow_run_id(self) -> UUID: ...

    async def run_node(
        self,
        compiled_node: CompiledNode,
        upstream_node_ids: frozenset[str],
        operation: NodeExecutionOperation,
        /,
    ) -> Mapping[str, ArtifactOutputValue]: ...

    async def run_map_item(
        self,
        compiled_node: CompiledNode,
        index: int,
        operation: MapItemExecutionOperation,
        /,
    ) -> PersistedNodeOutput: ...


class GraphExecutionEngine(Protocol):
    async def execute(
        self,
        execution: PreparedGraphExecution,
        /,
    ) -> GraphExecutionResult: ...


__all__ = [
    "ExecutionTaskRunner",
    "GraphExecutionEngine",
    "MapItemExecutionOperation",
    "NodeExecutionOperation",
    "PreparedGraphExecution",
]
