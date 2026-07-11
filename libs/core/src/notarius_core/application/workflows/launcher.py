from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from notarius_core.application.workflows.compiler import (
    NodeSpecRegistry,
    WorkflowCompiler,
)
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import (
    ArtifactRef,
    ArtifactPortRef,
    ArtifactSequence,
    ArtifactSequenceRef,
    ExecutionMode,
    NodeSpec,
    NodeRun,
    OutboxMessage,
    PortSpec,
    WorkflowRun,
    WorkflowVersion,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort

NodeRunOutboxMessageBuilder = Callable[[WorkflowRun, NodeRun], OutboxMessage]
WorkflowRunEventOutboxMessageBuilder = Callable[[WorkflowRun], OutboxMessage]
NodeRunEventOutboxMessageBuilder = Callable[[NodeRun], OutboxMessage]
CONCRETE_MAP_EXECUTION_PLANNING = "concrete_map"


@dataclass(frozen=True, slots=True)
class WorkflowLaunchResult:
    workflow_run: WorkflowRun
    node_runs: tuple[NodeRun, ...]
    queued_node_run_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class WorkflowRunLauncher:
    node_specs: NodeSpecRegistry

    async def launch(
        self,
        uow: StudioUnitOfWorkPort,
        workflow_version: WorkflowVersion,
        input_artifact_refs: Sequence[ArtifactRef],
        metadata: dict[str, object] | None = None,
        node_run_outbox_message_builder: NodeRunOutboxMessageBuilder | None = None,
        workflow_run_queued_event_builder: (
            WorkflowRunEventOutboxMessageBuilder | None
        ) = None,
        node_run_queued_event_builder: NodeRunEventOutboxMessageBuilder | None = None,
        input_artifact_sequences: Sequence[ArtifactSequence] | None = None,
        commit: bool = True,
    ) -> WorkflowLaunchResult:
        resolved_input_sequences = list(input_artifact_sequences or [])
        workflow_run = WorkflowRun(
            workflow_version_id=workflow_version.id,
            input_artifact_refs=list(input_artifact_refs),
            input_artifact_sequence_refs=[
                sequence.ref() for sequence in resolved_input_sequences
            ],
            metadata=dict(metadata or {}),
        )
        plan = WorkflowCompiler(self.node_specs).compile(
            workflow_version,
            workflow_run.id,
        )
        dependencies_by_node_run_id = {
            dependency.node_run_id: dependency for dependency in plan.dependencies
        }
        node_runs_by_original_id: dict[UUID, list[NodeRun]] = {}
        for node_run in plan.node_runs:
            spec = self.node_specs[(node_run.operator_id, node_run.operator_version)]
            node_run.input_artifact_refs = self._bind_external_node_inputs(
                workflow_version,
                node_run,
                input_artifact_refs,
                resolved_input_sequences,
            )
            node_runs_by_original_id[node_run.id] = self._expanded_node_runs(
                workflow_version,
                node_run,
                spec,
                resolved_input_sequences,
            )

        expanded_node_runs: list[NodeRun] = []
        expanded_upstream_ids_by_node_run_id: dict[UUID, list[UUID]] = {}
        for original_node_run in plan.node_runs:
            dependency = dependencies_by_node_run_id[original_node_run.id]
            upstream_node_run_ids: list[UUID] = []
            for upstream_original_node_run_id in dependency.upstream_node_run_ids:
                for upstream_node_run in node_runs_by_original_id[
                    upstream_original_node_run_id
                ]:
                    upstream_node_run_ids.append(upstream_node_run.id)

            for node_run in node_runs_by_original_id[original_node_run.id]:
                expanded_node_runs.append(node_run)
                expanded_upstream_ids_by_node_run_id[node_run.id] = upstream_node_run_ids

        if workflow_run_queued_event_builder is not None:
            await uow.outbox_messages.add(
                workflow_run_queued_event_builder(workflow_run)
            )
        queued_node_run_ids: list[UUID] = []
        for execution_index, node_run in enumerate(expanded_node_runs):
            spec = self.node_specs[(node_run.operator_id, node_run.operator_version)]
            node_run.metadata["execution_index"] = execution_index
            node_run.metadata["execution_mode"] = spec.execution_mode.value
            node_run.metadata["expected_input_ports"] = [
                self._port_metadata(port) for port in spec.inputs
            ]
            node_run.metadata["expected_output_ports"] = [
                self._port_metadata(port) for port in spec.outputs
            ]
            upstream_node_run_ids = expanded_upstream_ids_by_node_run_id[node_run.id]
            node_run.metadata["upstream_node_run_ids"] = [
                str(node_run_id) for node_run_id in upstream_node_run_ids
            ]
            if upstream_node_run_ids:
                node_run.mark_blocked()
                continue

            queued_node_run_ids.append(node_run.id)
            if node_run_queued_event_builder is not None:
                await uow.outbox_messages.add(node_run_queued_event_builder(node_run))
            if node_run_outbox_message_builder is not None:
                await uow.outbox_messages.add(
                    node_run_outbox_message_builder(workflow_run, node_run)
                )

        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add_batch(expanded_node_runs)
        if commit:
            await uow.commit()
        return WorkflowLaunchResult(
            workflow_run=workflow_run,
            node_runs=tuple(expanded_node_runs),
            queued_node_run_ids=tuple(queued_node_run_ids),
        )

    def _expanded_node_runs(
        self,
        workflow_version: WorkflowVersion,
        node_run: NodeRun,
        spec: NodeSpec,
        input_artifact_sequences: Sequence[ArtifactSequence],
    ) -> list[NodeRun]:
        definition = workflow_version.definition_snapshot
        if definition.metadata.get("execution_planning") != (
            CONCRETE_MAP_EXECUTION_PLANNING
        ):
            return [node_run]
        if spec.execution_mode != ExecutionMode.MAP:
            return [node_run]

        sequence_inputs = {
            port.name: node_run.input_artifact_refs[port.name]
            for port in spec.inputs
            if port.sequence and port.name in node_run.input_artifact_refs
        }
        if not sequence_inputs:
            return [node_run]
        if len(sequence_inputs) > 1:
            raise ValidationError(
                "Concrete map execution currently supports one bound sequence input "
                f"for workflow node {node_run.workflow_node_id}"
            )

        source_port_name, source_ref = next(iter(sequence_inputs.items()))
        item_refs = self._map_item_refs(source_ref, input_artifact_sequences)
        if not item_refs:
            raise ValidationError(
                "Concrete map execution cannot expand empty sequence input "
                f"{node_run.workflow_node_id}.{source_port_name}"
            )

        expanded_node_runs: list[NodeRun] = []
        for item_index, item_ref in enumerate(item_refs, start=1):
            input_refs = dict(node_run.input_artifact_refs)
            input_refs[source_port_name] = [item_ref]
            metadata = dict(node_run.metadata)
            metadata["concrete_execution_kind"] = "map_item"
            metadata["map_source_port"] = source_port_name
            metadata["map_item_index"] = item_index
            metadata["map_item_count"] = len(item_refs)
            if isinstance(source_ref, ArtifactSequenceRef):
                metadata["map_source_sequence_id"] = str(source_ref.sequence_id)
            expanded_node_runs.append(
                NodeRun(
                    workflow_run_id=node_run.workflow_run_id,
                    workflow_node_id=node_run.workflow_node_id,
                    operator_id=node_run.operator_id,
                    operator_version=node_run.operator_version,
                    input_artifact_refs=input_refs,
                    metadata=metadata,
                )
            )
        return expanded_node_runs

    def _map_item_refs(
        self,
        source_ref: ArtifactPortRef,
        input_artifact_sequences: Sequence[ArtifactSequence],
    ) -> list[ArtifactRef]:
        if isinstance(source_ref, list):
            return list(source_ref)
        if isinstance(source_ref, ArtifactRef):
            return [source_ref]

        sequences_by_id: Mapping[UUID, ArtifactSequence] = {
            sequence.id: sequence for sequence in input_artifact_sequences
        }
        sequence = sequences_by_id.get(source_ref.sequence_id)
        if sequence is None:
            raise ValidationError(
                "Concrete map execution requires the full artifact sequence for "
                f"{source_ref.sequence_id}"
            )
        return list(sequence.item_refs)

    def _bind_external_node_inputs(
        self,
        workflow_version: WorkflowVersion,
        node_run: NodeRun,
        input_artifact_refs: Sequence[ArtifactRef],
        input_artifact_sequences: Sequence[ArtifactSequence],
    ) -> dict[str, ArtifactPortRef]:
        definition = workflow_version.definition_snapshot
        spec = self.node_specs[(node_run.operator_id, node_run.operator_version)]
        incoming_ports = {
            edge.to_port
            for edge in definition.edges
            if edge.to_node_id == node_run.workflow_node_id
        }
        bound_inputs: dict[str, ArtifactPortRef] = {}
        for port in spec.inputs:
            if port.name in incoming_ports:
                continue

            declared_input = self._declared_input_for_port(
                workflow_node_id=node_run.workflow_node_id,
                port=port,
                declared_inputs=definition.declared_inputs,
            )
            if declared_input is None:
                if port.required:
                    raise ValidationError(
                        "Root workflow node input has no declared workflow input: "
                        f"{node_run.workflow_node_id}.{port.name}"
                    )
                continue

            if port.sequence:
                matching_sequence_refs = self._matching_sequence_refs(
                    input_artifact_sequences,
                    declared_input,
                )
                matching_refs = self._matching_refs(input_artifact_refs, declared_input)
                if len(matching_sequence_refs) > 1:
                    raise ValidationError(
                        "Workflow run provided multiple artifact sequences for "
                        f"{declared_input.name}"
                    )
                if matching_sequence_refs and matching_refs:
                    raise ValidationError(
                        "Workflow run provided both an artifact sequence and loose "
                        f"artifacts for sequence input {declared_input.name}"
                    )
                if matching_sequence_refs:
                    bound_inputs[port.name] = matching_sequence_refs[0]
                    continue
                if not matching_refs and port.required:
                    raise ValidationError(
                        "Workflow run is missing required input artifacts for "
                        f"{declared_input.name}"
                    )
                if matching_refs:
                    bound_inputs[port.name] = matching_refs
                continue

            matching_refs = self._matching_refs(input_artifact_refs, declared_input)
            if not matching_refs:
                if port.required:
                    raise ValidationError(
                        "Workflow run is missing required input artifact for "
                        f"{declared_input.name}"
                    )
                continue
            if len(matching_refs) > 1:
                raise ValidationError(
                    "Workflow run provided multiple artifacts for scalar input "
                    f"{declared_input.name}"
                )
            bound_inputs[port.name] = matching_refs[0]

        return bound_inputs

    def _declared_input_for_port(
        self,
        workflow_node_id: str,
        port: PortSpec,
        declared_inputs: Sequence[PortSpec],
    ) -> PortSpec | None:
        matches = [
            declared_input
            for declared_input in declared_inputs
            if declared_input.name == port.name
            and declared_input.artifact_type == port.artifact_type
            and declared_input.schema_version == port.schema_version
            and declared_input.sequence == port.sequence
        ]
        if len(matches) > 1:
            raise ValidationError(
                "Workflow declares duplicate inputs for root node port "
                f"{workflow_node_id}.{port.name}"
            )
        return matches[0] if matches else None

    def _matching_refs(
        self,
        input_artifact_refs: Sequence[ArtifactRef],
        declared_input: PortSpec,
    ) -> list[ArtifactRef]:
        return [
            artifact_ref
            for artifact_ref in input_artifact_refs
            if artifact_ref.artifact_type == declared_input.artifact_type
            and artifact_ref.schema_version == declared_input.schema_version
        ]

    def _matching_sequence_refs(
        self,
        input_artifact_sequences: Sequence[ArtifactSequence],
        declared_input: PortSpec,
    ) -> list[ArtifactSequenceRef]:
        return [
            sequence.ref()
            for sequence in input_artifact_sequences
            if sequence.artifact_type == declared_input.artifact_type
            and sequence.schema_version == declared_input.schema_version
        ]

    def _port_metadata(self, port: PortSpec) -> dict[str, object]:
        return {
            "name": port.name,
            "artifact_type": port.artifact_type,
            "schema_version": port.schema_version,
            "sequence": port.sequence,
            "required": port.required,
            "description": port.description,
        }
