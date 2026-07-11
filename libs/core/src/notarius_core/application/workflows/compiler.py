from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias
from uuid import UUID

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JsonSchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import (
    NodeRun,
    NodeSpec,
    PortSpec,
    WorkflowEdge,
    WorkflowNode,
    WorkflowVersion,
)

NodeSpecRegistry: TypeAlias = Mapping[tuple[str, str], NodeSpec]


@dataclass(frozen=True, slots=True)
class NodeRunDependency:
    node_run_id: UUID
    upstream_node_run_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class WorkflowExecutionPlan:
    workflow_version_id: UUID
    workflow_run_id: UUID
    node_runs: tuple[NodeRun, ...]
    dependencies: tuple[NodeRunDependency, ...]


@dataclass(frozen=True, slots=True)
class WorkflowCompiler:
    node_specs: NodeSpecRegistry

    def compile(
        self,
        workflow_version: WorkflowVersion,
        workflow_run_id: UUID,
    ) -> WorkflowExecutionPlan:
        definition = workflow_version.definition_snapshot
        nodes_by_id = self._nodes_by_id(definition.nodes)
        specs_by_node_id = self._specs_by_node_id(definition.nodes)

        self._validate_node_configs(definition.nodes, specs_by_node_id)
        self._validate_edges(definition.edges, nodes_by_id, specs_by_node_id)
        ordered_nodes = self._topological_order(definition.nodes, definition.edges)

        node_runs_by_workflow_node_id: dict[str, NodeRun] = {}
        node_runs: list[NodeRun] = []
        for node in ordered_nodes:
            node_run = NodeRun(
                workflow_run_id=workflow_run_id,
                workflow_node_id=node.id,
                operator_id=node.operator_id,
                operator_version=node.operator_version,
                metadata={
                    "workflow_version_id": str(workflow_version.id),
                    "workflow_node_config": node.config,
                },
            )
            node_runs_by_workflow_node_id[node.id] = node_run
            node_runs.append(node_run)

        dependencies: list[NodeRunDependency] = []
        for node in ordered_nodes:
            upstream_node_run_ids: list[UUID] = []
            for edge in definition.edges:
                if edge.to_node_id != node.id:
                    continue
                upstream_node_run_id = node_runs_by_workflow_node_id[
                    edge.from_node_id
                ].id
                if upstream_node_run_id not in upstream_node_run_ids:
                    upstream_node_run_ids.append(upstream_node_run_id)

            dependencies.append(
                NodeRunDependency(
                    node_run_id=node_runs_by_workflow_node_id[node.id].id,
                    upstream_node_run_ids=tuple(upstream_node_run_ids),
                )
            )

        return WorkflowExecutionPlan(
            workflow_version_id=workflow_version.id,
            workflow_run_id=workflow_run_id,
            node_runs=tuple(node_runs),
            dependencies=tuple(dependencies),
        )

    def _nodes_by_id(self, nodes: list[WorkflowNode]) -> dict[str, WorkflowNode]:
        nodes_by_id: dict[str, WorkflowNode] = {}
        duplicate_node_ids: list[str] = []
        for node in nodes:
            if node.id in nodes_by_id:
                duplicate_node_ids.append(node.id)
                continue
            nodes_by_id[node.id] = node

        if duplicate_node_ids:
            raise ValidationError(
                "Duplicate workflow node ids: " + ", ".join(duplicate_node_ids)
            )

        return nodes_by_id

    def _specs_by_node_id(self, nodes: list[WorkflowNode]) -> dict[str, NodeSpec]:
        specs_by_node_id: dict[str, NodeSpec] = {}
        for node in nodes:
            spec = self.node_specs.get((node.operator_id, node.operator_version))
            if spec is None:
                raise ValidationError(
                    "Unknown operator for workflow node "
                    f"{node.id}: {node.operator_id}@{node.operator_version}"
                )
            specs_by_node_id[node.id] = spec

        return specs_by_node_id

    def _validate_node_configs(
        self,
        nodes: list[WorkflowNode],
        specs_by_node_id: dict[str, NodeSpec],
    ) -> None:
        for node in nodes:
            spec = specs_by_node_id[node.id]
            if not spec.config_schema:
                continue

            try:
                Draft202012Validator.check_schema(spec.config_schema)
                Draft202012Validator(spec.config_schema).validate(node.config)
            except JsonSchemaValidationError as exc:
                raise ValidationError(
                    "Workflow node config does not match operator schema for "
                    f"{node.id}: {node.operator_id}@{node.operator_version}: "
                    f"{exc.message}"
                ) from exc
            except JsonSchemaError as exc:
                raise ValidationError(
                    "Operator config schema is invalid for workflow node "
                    f"{node.id}: {node.operator_id}@{node.operator_version}: "
                    f"{exc.message}"
                ) from exc

    def _validate_edges(
        self,
        edges: list[WorkflowEdge],
        nodes_by_id: dict[str, WorkflowNode],
        specs_by_node_id: dict[str, NodeSpec],
    ) -> None:
        connected_target_ports: set[tuple[str, str]] = set()
        for edge in edges:
            if edge.from_node_id not in nodes_by_id:
                raise ValidationError(
                    "Workflow edge references missing source node: "
                    f"{edge.from_node_id}.{edge.from_port} -> "
                    f"{edge.to_node_id}.{edge.to_port}"
                )
            if edge.to_node_id not in nodes_by_id:
                raise ValidationError(
                    "Workflow edge references missing target node: "
                    f"{edge.from_node_id}.{edge.from_port} -> "
                    f"{edge.to_node_id}.{edge.to_port}"
                )

            target_port_key = (edge.to_node_id, edge.to_port)
            if target_port_key in connected_target_ports:
                raise ValidationError(
                    "Workflow graph connects multiple edges into target input port: "
                    f"{edge.to_node_id}.{edge.to_port}"
                )
            connected_target_ports.add(target_port_key)

            source_outputs = {
                port.name: port for port in specs_by_node_id[edge.from_node_id].outputs
            }
            target_inputs = {
                port.name: port for port in specs_by_node_id[edge.to_node_id].inputs
            }
            source_port = source_outputs.get(edge.from_port)
            target_port = target_inputs.get(edge.to_port)

            if source_port is None:
                raise ValidationError(
                    "Workflow edge references missing source output port: "
                    f"{edge.from_node_id}.{edge.from_port}"
                )
            if target_port is None:
                raise ValidationError(
                    "Workflow edge references missing target input port: "
                    f"{edge.to_node_id}.{edge.to_port}"
                )

            self._validate_port_compatibility(edge, source_port, target_port)

    def _validate_port_compatibility(
        self,
        edge: WorkflowEdge,
        source_port: PortSpec,
        target_port: PortSpec,
    ) -> None:
        if (
            source_port.artifact_type == target_port.artifact_type
            and source_port.schema_version == target_port.schema_version
            and source_port.sequence == target_port.sequence
        ):
            return

        raise ValidationError(
            "Workflow edge connects incompatible artifact contracts: "
            f"{edge.from_node_id}.{edge.from_port} produces "
            f"{source_port.artifact_type}@v{source_port.schema_version} "
            f"sequence={source_port.sequence}; "
            f"{edge.to_node_id}.{edge.to_port} expects "
            f"{target_port.artifact_type}@v{target_port.schema_version} "
            f"sequence={target_port.sequence}"
        )

    def _topological_order(
        self,
        nodes: list[WorkflowNode],
        edges: list[WorkflowEdge],
    ) -> list[WorkflowNode]:
        nodes_by_id = {node.id: node for node in nodes}
        outgoing_edges_by_node_id: dict[str, list[WorkflowEdge]] = {
            node.id: [] for node in nodes
        }
        incoming_counts = {node.id: 0 for node in nodes}
        for edge in edges:
            outgoing_edges_by_node_id[edge.from_node_id].append(edge)
            incoming_counts[edge.to_node_id] += 1

        ready = [node for node in nodes if incoming_counts[node.id] == 0]
        ordered_nodes: list[WorkflowNode] = []
        while ready:
            node = ready.pop(0)
            ordered_nodes.append(node)
            for edge in outgoing_edges_by_node_id[node.id]:
                incoming_counts[edge.to_node_id] -= 1
                if incoming_counts[edge.to_node_id] == 0:
                    ready.append(nodes_by_id[edge.to_node_id])

        if len(ordered_nodes) != len(nodes):
            cycle_node_ids = [node.id for node in nodes if incoming_counts[node.id] > 0]
            raise ValidationError(
                "Workflow graph contains a cycle involving nodes: "
                + ", ".join(cycle_node_ids)
            )

        return ordered_nodes
