from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from notarius_core.artifacts import ArtifactRef
from notarius_core.domain.modules import GraphModuleDefinition
from notarius_core.nodes import NodeExecutionContext


@dataclass(frozen=True, slots=True)
class GraphModuleExecutionResult:
    outputs: Mapping[str, ArtifactRef]

    def __post_init__(self) -> None:
        raw_outputs = cast(Mapping[object, object], self.outputs)
        copied: dict[str, ArtifactRef] = {}
        for raw_name, raw_value in raw_outputs.items():
            if not isinstance(raw_name, str):
                raise TypeError(
                    "Graph module execution output names must be strings, got "
                    f"{type(raw_name).__name__}"
                )
            if not isinstance(raw_value, ArtifactRef):
                raise TypeError(
                    f"Graph module execution output {raw_name!r} must be an "
                    f"ArtifactRef, got {type(raw_value).__name__}"
                )
            copied[raw_name] = raw_value
        object.__setattr__(self, "outputs", MappingProxyType(copied))


@runtime_checkable
class GraphModuleExecutorPort(Protocol):
    async def execute_module(
        self,
        definition: GraphModuleDefinition,
        context: NodeExecutionContext,
        inputs: Mapping[str, ArtifactRef],
        /,
    ) -> GraphModuleExecutionResult: ...


__all__ = [
    "GraphModuleExecutionResult",
    "GraphModuleExecutorPort",
]
