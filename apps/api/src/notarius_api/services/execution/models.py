from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal
from uuid import UUID

from notarius_core.artifacts import ArtifactFieldProjection, ArtifactTypeKey
from notarius_core.conversions import ArtifactConversion
from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.nodes import Node, ResolvedNodeContracts
from notarius_core.plugins import NodeRegistration
from notarius_core.runtime.invocation import NodeInvocation

from notarius_api.schemas.workbench import RunEdgeRequest, RunNodeRequest

type OutputEndpoint = tuple[str, str]


@dataclass(frozen=True, slots=True)
class CompiledEdge:
    request: RunEdgeRequest
    projection: ArtifactFieldProjection | None
    conversion_path: tuple[ArtifactConversion[Any, Any], ...]


@dataclass(frozen=True, slots=True)
class CompiledNode:
    request: RunNodeRequest
    node: Node[Any, Any, Any]
    registration: NodeRegistration | None
    resolved_contracts: ResolvedNodeContracts
    invocation: NodeInvocation
    artifact_type_bindings: Mapping[str, ArtifactTypeKey]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_type_bindings",
            MappingProxyType(dict(self.artifact_type_bindings)),
        )


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    nodes: tuple[CompiledNode, ...]
    edges: tuple[CompiledEdge, ...]
    pinned_outputs: Mapping[OutputEndpoint, ArtifactOutputValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pinned_outputs",
            MappingProxyType(dict(self.pinned_outputs)),
        )


@dataclass(frozen=True, slots=True)
class NodeExecutionResult:
    node_id: str
    status: Literal["succeeded", "failed", "skipped"]
    error: str | None
    outputs: Mapping[str, ArtifactOutputValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outputs",
            MappingProxyType(dict(self.outputs)),
        )


@dataclass(frozen=True, slots=True)
class GraphExecutionResult:
    workflow_run_id: UUID
    status: Literal["succeeded", "failed"]
    node_results: tuple[NodeExecutionResult, ...]
    outputs: Mapping[str, Mapping[str, ArtifactOutputValue]]

    def __post_init__(self) -> None:
        copied_outputs = {
            node_id: MappingProxyType(dict(node_outputs))
            for node_id, node_outputs in self.outputs.items()
        }
        object.__setattr__(
            self,
            "outputs",
            MappingProxyType(copied_outputs),
        )


__all__ = [
    "CompiledEdge",
    "CompiledGraph",
    "CompiledNode",
    "GraphExecutionResult",
    "NodeExecutionResult",
    "OutputEndpoint",
]
