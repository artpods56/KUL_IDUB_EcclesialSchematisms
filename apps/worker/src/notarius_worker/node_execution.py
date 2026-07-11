from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from notarius_core.domain.models import (
    Artifact,
    ArtifactPortRef,
    ArtifactRef,
    ArtifactSequence,
    ArtifactSequenceRef,
    InputAssemblyTrace,
    NodeSpec,
    InvocationTrace,
    NodeRun,
    NodeRunStatus,
    PortSpec,
    WorkflowEdge,
    WorkflowRunStatus,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_messaging.contracts import ErrorContext, RunEventType
from notarius_messaging.outbox import (
    artifact_created_event_outbox_message,
    node_run_event_outbox_message,
    node_run_execute_requested_outbox_message,
    workflow_run_event_outbox_message,
)


@dataclass(frozen=True, slots=True)
class NodeRunExecutionError(RuntimeError):
    message: str
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class ArtifactSequenceInput:
    sequence: ArtifactSequence
    artifacts: list[Artifact]


@dataclass(frozen=True, slots=True)
class NodeExecutionRequest:
    node_run: NodeRun
    input_artifacts: dict[str, Artifact | list[Artifact] | ArtifactSequenceInput]


@dataclass(frozen=True, slots=True)
class NodeExecutionResult:
    output_artifact_refs: dict[str, ArtifactPortRef]
    artifacts: list[Artifact] = field(default_factory=list)
    artifact_sequences: list[ArtifactSequence] = field(default_factory=list)
    input_assembly_traces: list[InputAssemblyTrace] = field(default_factory=list)
    invocation_traces: list[InvocationTrace] = field(default_factory=list)


class NodeRunHandler(Protocol):
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult: ...


class NodeRunExecutor:
    def __init__(
        self,
        uow_factory: Callable[[], StudioUnitOfWorkPort],
        handlers: Mapping[tuple[str, str], NodeRunHandler],
        node_specs: Mapping[tuple[str, str], NodeSpec] | None = None,
    ):
        self.uow_factory = uow_factory
        self.handlers = handlers
        self.node_specs = dict(node_specs) if node_specs is not None else None

    async def execute_next_node_run(self) -> UUID | None:
        async with self.uow_factory() as uow:
            node_run = await uow.node_runs.next_queued()
            if node_run is None:
                return None
            node_run_id = node_run.id

        await self.execute_node_run(node_run_id)
        return node_run_id

    async def execute_node_run(self, node_run_id: UUID | str) -> None:
        resolved_node_run_id = (
            node_run_id if isinstance(node_run_id, UUID) else UUID(node_run_id)
        )

        async with self.uow_factory() as uow:
            node_run = await uow.node_runs.get(resolved_node_run_id)
            if node_run is None:
                raise NodeRunExecutionError(
                    f"NodeRun not found: {resolved_node_run_id}",
                    retryable=False,
                )
            if node_run.is_terminal or node_run.status == NodeRunStatus.CANCELLED:
                return
            workflow_run = await uow.workflow_runs.get(node_run.workflow_run_id)
            if workflow_run is None:
                error = f"WorkflowRun not found: {node_run.workflow_run_id}"
                node_run.mark_failed(error, retryable=False)
                await uow.node_runs.update(node_run)
                await uow.outbox_messages.add(
                    node_run_event_outbox_message(
                        node_run,
                        RunEventType.FAILED_PERMANENT,
                        self._node_run_error_context(
                            "execute_node_run",
                            node_run,
                            error,
                            retryable=False,
                        ),
                    )
                )
                await uow.commit()
                raise NodeRunExecutionError(
                    error,
                    retryable=False,
                )
            if workflow_run.status == WorkflowRunStatus.CANCELLED:
                node_run.mark_cancelled()
                await uow.node_runs.update(node_run)
                await uow.outbox_messages.add(
                    node_run_event_outbox_message(
                        node_run,
                        RunEventType.CANCELLED,
                    )
                )
                await uow.commit()
                return
            if (
                node_run.status == NodeRunStatus.FAILED_RETRYABLE
                and node_run.attempt_count >= node_run.max_attempts
            ):
                error = (
                    "Retry attempts exhausted for node run "
                    f"{node_run.id}: "
                    f"{node_run.attempt_count}/{node_run.max_attempts}"
                )
                node_run.mark_failed(
                    error,
                    retryable=False,
                )
                await uow.node_runs.update(node_run)
                error_context = self._node_run_error_context(
                    "execute_node_run",
                    node_run,
                    error,
                    retryable=False,
                )
                await uow.outbox_messages.add(
                    node_run_event_outbox_message(
                        node_run,
                        RunEventType.FAILED_PERMANENT,
                        error_context,
                    )
                )
                workflow_run.mark_failed(
                    self._workflow_failure_message(node_run),
                    retryable=False,
                )
                await uow.workflow_runs.update(workflow_run)
                await uow.outbox_messages.add(
                    workflow_run_event_outbox_message(
                        workflow_run,
                        RunEventType.FAILED_PERMANENT,
                        error_context,
                    )
                )
                await uow.commit()
                return
            if workflow_run.status == WorkflowRunStatus.QUEUED:
                workflow_run.mark_running()
                await uow.workflow_runs.update(workflow_run)
                await uow.outbox_messages.add(
                    workflow_run_event_outbox_message(
                        workflow_run,
                        RunEventType.RUNNING,
                    )
                )
            node_run.mark_running()
            await uow.node_runs.update(node_run)
            await uow.outbox_messages.add(
                node_run_event_outbox_message(
                    node_run,
                    RunEventType.RUNNING,
                )
            )
            await uow.commit()

        try:
            async with self.uow_factory() as uow:
                node_run = await uow.node_runs.get(resolved_node_run_id)
                if node_run is None:
                    raise NodeRunExecutionError(
                        f"NodeRun not found: {resolved_node_run_id}",
                        retryable=False,
                    )
                workflow_run = await uow.workflow_runs.get(node_run.workflow_run_id)
                if workflow_run is None:
                    raise NodeRunExecutionError(
                        f"WorkflowRun not found: {node_run.workflow_run_id}",
                        retryable=False,
                    )
                if workflow_run.status == WorkflowRunStatus.CANCELLED:
                    node_run.mark_cancelled()
                    await uow.node_runs.update(node_run)
                    await uow.outbox_messages.add(
                        node_run_event_outbox_message(
                            node_run,
                            RunEventType.CANCELLED,
                        )
                    )
                    await uow.commit()
                    return
                handler = self.handlers.get(
                    (node_run.operator_id, node_run.operator_version)
                )
                if handler is None:
                    raise NodeRunExecutionError(
                        "No node-run handler registered for "
                        f"{node_run.operator_id}:{node_run.operator_version}",
                        retryable=False,
                    )

                input_artifacts = await self._load_input_artifacts(uow, node_run)

            result = await handler.execute(
                NodeExecutionRequest(
                    node_run=node_run,
                    input_artifacts=input_artifacts,
                )
            )
            self._validate_output_refs(node_run, result)

            async with self.uow_factory() as uow:
                node_run = await uow.node_runs.get(resolved_node_run_id)
                if node_run is None:
                    raise NodeRunExecutionError(
                        f"NodeRun not found: {resolved_node_run_id}",
                        retryable=False,
                    )
                workflow_run = await uow.workflow_runs.get(node_run.workflow_run_id)
                if workflow_run is None:
                    raise NodeRunExecutionError(
                        f"WorkflowRun not found: {node_run.workflow_run_id}",
                        retryable=False,
                    )
                if (
                    node_run.is_terminal
                    or node_run.status == NodeRunStatus.CANCELLED
                    or workflow_run.status == WorkflowRunStatus.CANCELLED
                ):
                    if workflow_run.status == WorkflowRunStatus.CANCELLED:
                        node_run.mark_cancelled()
                        await uow.node_runs.update(node_run)
                        await uow.outbox_messages.add(
                            node_run_event_outbox_message(
                                node_run,
                                RunEventType.CANCELLED,
                            )
                        )
                        await uow.commit()
                    return
                for artifact in result.artifacts:
                    await uow.artifacts.add(artifact)
                    await uow.outbox_messages.add(
                        artifact_created_event_outbox_message(artifact)
                    )
                for sequence in result.artifact_sequences:
                    await uow.artifact_sequences.add(sequence)
                for trace in result.input_assembly_traces:
                    await uow.input_assembly_traces.add(trace)
                for trace in result.invocation_traces:
                    await uow.invocation_traces.add(trace)

                node_run.mark_succeeded(result.output_artifact_refs)
                await uow.node_runs.update(node_run)
                await uow.outbox_messages.add(
                    node_run_event_outbox_message(
                        node_run,
                        RunEventType.SUCCEEDED,
                    )
                )
                await self._queue_ready_downstream_node_runs(uow, node_run)
                await self._finalize_workflow_run_if_complete(
                    uow,
                    node_run.workflow_run_id,
                )
                await uow.commit()
        except NodeRunExecutionError as exc:
            await self._mark_failed(resolved_node_run_id, str(exc), exc.retryable)
            raise
        except Exception as exc:
            await self._mark_failed(resolved_node_run_id, str(exc), retryable=False)
            raise

    async def _load_input_artifacts(
        self,
        uow: StudioUnitOfWorkPort,
        node_run: NodeRun,
    ) -> dict[str, Artifact | list[Artifact] | ArtifactSequenceInput]:
        input_artifacts: dict[str, Artifact | list[Artifact] | ArtifactSequenceInput] = {}
        for port_name, artifact_ref in node_run.input_artifact_refs.items():
            if isinstance(artifact_ref, ArtifactSequenceRef):
                input_artifacts[port_name] = await self._load_input_artifact_sequence(
                    uow,
                    port_name,
                    artifact_ref,
                )
            elif isinstance(artifact_ref, list):
                input_artifacts[port_name] = [
                    await self._load_input_artifact(uow, port_name, item)
                    for item in artifact_ref
                ]
            else:
                input_artifacts[port_name] = await self._load_input_artifact(
                    uow,
                    port_name,
                    artifact_ref,
                )
        return input_artifacts

    async def _load_input_artifact_sequence(
        self,
        uow: StudioUnitOfWorkPort,
        port_name: str,
        sequence_ref: ArtifactSequenceRef,
    ) -> ArtifactSequenceInput:
        sequence = await uow.artifact_sequences.get(sequence_ref.sequence_id)
        if sequence is None:
            raise NodeRunExecutionError(
                "Input artifact sequence not found for port "
                f"{port_name}: {sequence_ref.sequence_id}",
                retryable=False,
            )
        if sequence.artifact_type != sequence_ref.artifact_type:
            raise NodeRunExecutionError(
                "Input artifact sequence type mismatch for port "
                f"{port_name} and sequence {sequence_ref.sequence_id}: expected "
                f"{sequence_ref.artifact_type}, got {sequence.artifact_type}",
                retryable=False,
            )
        if sequence.schema_version != sequence_ref.schema_version:
            raise NodeRunExecutionError(
                "Input artifact sequence schema version mismatch for port "
                f"{port_name} and sequence {sequence_ref.sequence_id}: expected "
                f"{sequence_ref.schema_version}, got {sequence.schema_version}",
                retryable=False,
            )

        return ArtifactSequenceInput(
            sequence=sequence,
            artifacts=[
                await self._load_input_artifact(uow, port_name, item_ref)
                for item_ref in sequence.item_refs
            ],
        )

    async def _load_input_artifact(
        self,
        uow: StudioUnitOfWorkPort,
        port_name: str,
        artifact_ref: ArtifactRef,
    ) -> Artifact:
        artifact = await uow.artifacts.get(artifact_ref.artifact_id)
        if artifact is None:
            raise NodeRunExecutionError(
                f"Input artifact not found for port {port_name}: "
                f"{artifact_ref.artifact_id}",
                retryable=False,
            )
        if artifact.artifact_type != artifact_ref.artifact_type:
            raise NodeRunExecutionError(
                "Input artifact type mismatch for port "
                f"{port_name} and artifact "
                f"{artifact_ref.artifact_id}: expected "
                f"{artifact_ref.artifact_type}, got {artifact.artifact_type}",
                retryable=False,
            )
        if artifact.schema_version != artifact_ref.schema_version:
            raise NodeRunExecutionError(
                "Input artifact schema version mismatch for port "
                f"{port_name} and artifact "
                f"{artifact_ref.artifact_id}: expected "
                f"{artifact_ref.schema_version}, got {artifact.schema_version}",
                retryable=False,
            )
        return artifact

    def _validate_output_refs(
        self,
        node_run: NodeRun,
        result: NodeExecutionResult,
    ) -> None:
        if self.node_specs is None:
            return

        spec = self.node_specs.get((node_run.operator_id, node_run.operator_version))
        if spec is None:
            raise NodeRunExecutionError(
                "No node spec registered for node-run handler "
                f"{node_run.operator_id}:{node_run.operator_version}",
                retryable=False,
            )

        output_ports = {port.name: port for port in spec.outputs}
        for port in spec.outputs:
            if port.required and port.name not in result.output_artifact_refs:
                raise NodeRunExecutionError(
                    "NodeRun handler did not produce required output port "
                    f"{port.name} for {node_run.operator_id}:{node_run.operator_version}",
                    retryable=False,
                )

        artifacts_by_id = {artifact.id: artifact for artifact in result.artifacts}
        sequences_by_id = {
            sequence.id: sequence for sequence in result.artifact_sequences
        }
        for port_name, output_ref in result.output_artifact_refs.items():
            port = output_ports.get(port_name)
            if port is None:
                raise NodeRunExecutionError(
                    "NodeRun handler produced undeclared output port "
                    f"{port_name} for {node_run.operator_id}:{node_run.operator_version}",
                    retryable=False,
                )
            self._validate_output_ref_contract(
                node_run,
                port,
                output_ref,
                artifacts_by_id,
                sequences_by_id,
            )

    def _validate_output_ref_contract(
        self,
        node_run: NodeRun,
        port: PortSpec,
        output_ref: ArtifactPortRef,
        artifacts_by_id: dict[UUID, Artifact],
        sequences_by_id: dict[UUID, ArtifactSequence],
    ) -> None:
        if isinstance(output_ref, ArtifactSequenceRef):
            if not port.sequence:
                raise NodeRunExecutionError(
                    "NodeRun handler produced an artifact sequence for non-sequence "
                    f"output port {port.name}",
                    retryable=False,
                )
            self._validate_sequence_ref_contract(port, output_ref, sequences_by_id)
            return

        if isinstance(output_ref, list):
            if not port.sequence:
                raise NodeRunExecutionError(
                    "NodeRun handler produced multiple artifact refs for non-sequence "
                    f"output port {port.name}",
                    retryable=False,
                )
            for item_ref in output_ref:
                self._validate_artifact_ref_contract(
                    node_run,
                    port,
                    item_ref,
                    artifacts_by_id,
                )
            return

        if port.sequence:
            raise NodeRunExecutionError(
                "NodeRun handler produced a single artifact ref for sequence output "
                f"port {port.name}",
                retryable=False,
            )
        self._validate_artifact_ref_contract(
            node_run,
            port,
            output_ref,
            artifacts_by_id,
        )

    def _validate_artifact_ref_contract(
        self,
        node_run: NodeRun,
        port: PortSpec,
        artifact_ref: ArtifactRef,
        artifacts_by_id: dict[UUID, Artifact],
    ) -> None:
        if artifact_ref.artifact_type != port.artifact_type:
            raise NodeRunExecutionError(
                "NodeRun handler output artifact type mismatch for port "
                f"{port.name}: expected {port.artifact_type}, "
                f"got {artifact_ref.artifact_type}",
                retryable=False,
            )
        if artifact_ref.schema_version != port.schema_version:
            raise NodeRunExecutionError(
                "NodeRun handler output artifact schema version mismatch for port "
                f"{port.name}: expected {port.schema_version}, "
                f"got {artifact_ref.schema_version}",
                retryable=False,
            )

        artifact = artifacts_by_id.get(artifact_ref.artifact_id)
        if artifact is None:
            return
        if artifact.artifact_type != artifact_ref.artifact_type:
            raise NodeRunExecutionError(
                "NodeRun handler output ref does not match returned artifact type "
                f"for port {port.name}: ref has {artifact_ref.artifact_type}, "
                f"artifact has {artifact.artifact_type}",
                retryable=False,
            )
        if artifact.schema_version != artifact_ref.schema_version:
            raise NodeRunExecutionError(
                "NodeRun handler output ref does not match returned artifact schema "
                f"for port {port.name}: ref has {artifact_ref.schema_version}, "
                f"artifact has {artifact.schema_version}",
                retryable=False,
            )
        if artifact.workflow_run_id != node_run.workflow_run_id:
            raise NodeRunExecutionError(
                "NodeRun handler returned output artifact for the wrong workflow run "
                f"on port {port.name}",
                retryable=False,
            )

    def _validate_sequence_ref_contract(
        self,
        port: PortSpec,
        sequence_ref: ArtifactSequenceRef,
        sequences_by_id: dict[UUID, ArtifactSequence],
    ) -> None:
        if sequence_ref.artifact_type != port.artifact_type:
            raise NodeRunExecutionError(
                "NodeRun handler output sequence type mismatch for port "
                f"{port.name}: expected {port.artifact_type}, "
                f"got {sequence_ref.artifact_type}",
                retryable=False,
            )
        if sequence_ref.schema_version != port.schema_version:
            raise NodeRunExecutionError(
                "NodeRun handler output sequence schema version mismatch for port "
                f"{port.name}: expected {port.schema_version}, "
                f"got {sequence_ref.schema_version}",
                retryable=False,
            )

        sequence = sequences_by_id.get(sequence_ref.sequence_id)
        if sequence is None:
            return
        if sequence.artifact_type != sequence_ref.artifact_type:
            raise NodeRunExecutionError(
                "NodeRun handler output ref does not match returned sequence type "
                f"for port {port.name}: ref has {sequence_ref.artifact_type}, "
                f"sequence has {sequence.artifact_type}",
                retryable=False,
            )
        if sequence.schema_version != sequence_ref.schema_version:
            raise NodeRunExecutionError(
                "NodeRun handler output ref does not match returned sequence schema "
                f"for port {port.name}: ref has {sequence_ref.schema_version}, "
                f"sequence has {sequence.schema_version}",
                retryable=False,
            )

    async def _mark_failed(
        self,
        node_run_id: UUID,
        error: str,
        retryable: bool,
    ) -> None:
        async with self.uow_factory() as uow:
            node_run = await uow.node_runs.get(node_run_id)
            if node_run is None:
                return
            workflow_run = await uow.workflow_runs.get(node_run.workflow_run_id)
            if node_run.is_terminal or node_run.status == NodeRunStatus.CANCELLED:
                return
            if (
                workflow_run is not None
                and workflow_run.status == WorkflowRunStatus.CANCELLED
            ):
                node_run.mark_cancelled()
                await uow.node_runs.update(node_run)
                await uow.outbox_messages.add(
                    node_run_event_outbox_message(
                        node_run,
                        RunEventType.CANCELLED,
                    )
                )
                await uow.commit()
                return
            node_run.mark_failed(error, retryable=retryable)
            await uow.node_runs.update(node_run)
            error_context = self._node_run_error_context(
                "execute_node_run",
                node_run,
                error,
                retryable=retryable,
            )
            await uow.outbox_messages.add(
                node_run_event_outbox_message(
                    node_run,
                    RunEventType.FAILED_RETRYABLE
                    if retryable
                    else RunEventType.FAILED_PERMANENT,
                    error_context,
                )
            )
            if workflow_run is not None:
                workflow_run.mark_failed(
                    f"NodeRun {node_run.id} failed: {error}",
                    retryable=retryable,
                )
                await uow.workflow_runs.update(workflow_run)
                await uow.outbox_messages.add(
                    workflow_run_event_outbox_message(
                        workflow_run,
                        RunEventType.FAILED_RETRYABLE
                        if retryable
                        else RunEventType.FAILED_PERMANENT,
                        error_context,
                    )
                )
            await uow.commit()

    async def _queue_ready_downstream_node_runs(
        self,
        uow: StudioUnitOfWorkPort,
        completed_node_run: NodeRun,
    ) -> None:
        workflow_version_id = completed_node_run.metadata.get("workflow_version_id")
        if not isinstance(workflow_version_id, str):
            return

        workflow_version = await uow.workflow_versions.get(UUID(workflow_version_id))
        if workflow_version is None:
            return
        workflow_run = await uow.workflow_runs.get(completed_node_run.workflow_run_id)
        if workflow_run is None:
            return

        node_runs = await uow.node_runs.list_for_workflow_run(
            completed_node_run.workflow_run_id
        )
        node_runs_by_id = {node_run.id: node_run for node_run in node_runs}
        node_runs_by_workflow_node_id: dict[str, list[NodeRun]] = {}
        for node_run in node_runs:
            node_runs_by_workflow_node_id.setdefault(
                node_run.workflow_node_id,
                [],
            ).append(node_run)
        for downstream_node_run in node_runs:
            if downstream_node_run.status != NodeRunStatus.BLOCKED:
                continue

            upstream_node_run_ids = self._upstream_node_run_ids(downstream_node_run)
            if completed_node_run.id not in upstream_node_run_ids:
                continue

            upstream_node_runs = [
                node_runs_by_id[node_run_id]
                for node_run_id in upstream_node_run_ids
                if node_run_id in node_runs_by_id
            ]
            if len(upstream_node_runs) != len(upstream_node_run_ids):
                continue
            if any(
                upstream_node_run.status != NodeRunStatus.SUCCEEDED
                for upstream_node_run in upstream_node_runs
            ):
                continue

            edges = [
                edge
                for edge in workflow_version.definition_snapshot.edges
                if edge.to_node_id == downstream_node_run.workflow_node_id
            ]
            missing_output = await self._bind_upstream_outputs(
                uow,
                downstream_node_run,
                edges,
                node_runs_by_workflow_node_id,
            )
            if missing_output is not None:
                downstream_node_run.mark_failed(missing_output, retryable=False)
                await uow.outbox_messages.add(
                    node_run_event_outbox_message(
                        downstream_node_run,
                        RunEventType.FAILED_PERMANENT,
                        self._node_run_error_context(
                            "queue_downstream_node_run",
                            downstream_node_run,
                            missing_output,
                            retryable=False,
                        ),
                    )
                )
            else:
                downstream_node_run.mark_queued()
                await uow.outbox_messages.add(
                    node_run_event_outbox_message(
                        downstream_node_run,
                        RunEventType.QUEUED,
                    )
                )
                await uow.outbox_messages.add(
                    node_run_execute_requested_outbox_message(
                        workflow_run,
                        downstream_node_run,
                    )
                )
            await uow.node_runs.update(downstream_node_run)

    def _upstream_node_run_ids(self, node_run: NodeRun) -> list[UUID]:
        raw_values = node_run.metadata.get("upstream_node_run_ids", [])
        if not isinstance(raw_values, list):
            return []
        return [UUID(str(value)) for value in raw_values]

    async def _bind_upstream_outputs(
        self,
        uow: StudioUnitOfWorkPort,
        downstream_node_run: NodeRun,
        edges: list[WorkflowEdge],
        node_runs_by_workflow_node_id: dict[str, list[NodeRun]],
    ) -> str | None:
        input_artifact_refs = dict(downstream_node_run.input_artifact_refs)
        for edge in edges:
            upstream_node_runs = node_runs_by_workflow_node_id.get(edge.from_node_id, [])
            if not upstream_node_runs:
                return (
                    "Upstream node run not found for workflow edge "
                    f"{edge.from_node_id}.{edge.from_port} -> "
                    f"{edge.to_node_id}.{edge.to_port}"
                )
            if len(upstream_node_runs) == 1:
                output_ref = upstream_node_runs[0].output_artifact_refs.get(
                    edge.from_port
                )
                if output_ref is None:
                    return (
                        "Upstream node run did not produce output port "
                        f"{edge.from_node_id}.{edge.from_port} required by "
                        f"{edge.to_node_id}.{edge.to_port}"
                    )
                input_artifact_refs[edge.to_port] = output_ref
                continue

            collected_refs: list[ArtifactRef] = []
            for upstream_node_run in sorted(
                upstream_node_runs,
                key=self._concrete_map_output_order,
            ):
                output_ref = upstream_node_run.output_artifact_refs.get(edge.from_port)
                if output_ref is None:
                    return (
                        "Upstream node run did not produce output port "
                        f"{edge.from_node_id}.{edge.from_port} required by "
                        f"{edge.to_node_id}.{edge.to_port}"
                    )
                collected_refs.extend(
                    await self._flatten_artifact_port_ref(uow, output_ref)
                )
            if not collected_refs:
                return (
                    "Upstream node runs produced no artifact refs for workflow edge "
                    f"{edge.from_node_id}.{edge.from_port} -> "
                    f"{edge.to_node_id}.{edge.to_port}"
                )
            first_ref = collected_refs[0]
            for collected_ref in collected_refs:
                if (
                    collected_ref.artifact_type != first_ref.artifact_type
                    or collected_ref.schema_version != first_ref.schema_version
                ):
                    return (
                        "Upstream node runs produced incompatible artifact refs for "
                        f"workflow edge {edge.from_node_id}.{edge.from_port} -> "
                        f"{edge.to_node_id}.{edge.to_port}"
                    )
            sequence = ArtifactSequence(
                artifact_type=first_ref.artifact_type,
                schema_version=first_ref.schema_version,
                item_refs=collected_refs,
                metadata={
                    "source_workflow_node_id": edge.from_node_id,
                    "source_output_port": edge.from_port,
                    "target_workflow_node_id": edge.to_node_id,
                    "target_input_port": edge.to_port,
                },
            )
            await uow.artifact_sequences.add(sequence)
            input_artifact_refs[edge.to_port] = sequence.ref()

        downstream_node_run.input_artifact_refs = input_artifact_refs
        return None

    def _concrete_map_output_order(self, node_run: NodeRun) -> tuple[int, int]:
        raw_map_item_index = node_run.metadata.get("map_item_index")
        raw_execution_index = node_run.metadata.get("execution_index")
        map_item_index = (
            raw_map_item_index if type(raw_map_item_index) is int else 0
        )
        execution_index = (
            raw_execution_index if type(raw_execution_index) is int else 0
        )
        return (map_item_index, execution_index)

    async def _finalize_workflow_run_if_complete(
        self,
        uow: StudioUnitOfWorkPort,
        workflow_run_id: UUID,
    ) -> None:
        workflow_run = await uow.workflow_runs.get(workflow_run_id)
        if workflow_run is None or workflow_run.is_terminal:
            return

        node_runs = await uow.node_runs.list_for_workflow_run(workflow_run_id)
        if not node_runs:
            return
        open_statuses = {
            NodeRunStatus.QUEUED,
            NodeRunStatus.BLOCKED,
            NodeRunStatus.RUNNING,
        }
        if any(node_run.status in open_statuses for node_run in node_runs):
            return

        failed_retryable = next(
            (
                node_run
                for node_run in node_runs
                if node_run.status == NodeRunStatus.FAILED_RETRYABLE
            ),
            None,
        )
        if failed_retryable is not None:
            error = self._workflow_failure_message(failed_retryable)
            workflow_run.mark_failed(
                error,
                retryable=True,
            )
            await uow.workflow_runs.update(workflow_run)
            await uow.outbox_messages.add(
                workflow_run_event_outbox_message(
                    workflow_run,
                    RunEventType.FAILED_RETRYABLE,
                    self._node_run_error_context(
                        "finalize_workflow_run",
                        failed_retryable,
                        error,
                        retryable=True,
                    ),
                )
            )
            return

        failed_permanent = next(
            (
                node_run
                for node_run in node_runs
                if node_run.status == NodeRunStatus.FAILED_PERMANENT
            ),
            None,
        )
        if failed_permanent is not None:
            error = self._workflow_failure_message(failed_permanent)
            workflow_run.mark_failed(
                error,
                retryable=False,
            )
            await uow.workflow_runs.update(workflow_run)
            await uow.outbox_messages.add(
                workflow_run_event_outbox_message(
                    workflow_run,
                    RunEventType.FAILED_PERMANENT,
                    self._node_run_error_context(
                        "finalize_workflow_run",
                        failed_permanent,
                        error,
                        retryable=False,
                    ),
                )
            )
            return

        if any(node_run.status == NodeRunStatus.CANCELLED for node_run in node_runs):
            workflow_run.mark_cancelled()
            await uow.workflow_runs.update(workflow_run)
            await uow.outbox_messages.add(
                workflow_run_event_outbox_message(
                    workflow_run,
                    RunEventType.CANCELLED,
                )
            )
            return

        workflow_run.mark_succeeded(
            await self._workflow_output_artifact_refs(uow, node_runs)
        )
        await uow.workflow_runs.update(workflow_run)
        await uow.outbox_messages.add(
            workflow_run_event_outbox_message(
                workflow_run,
                RunEventType.SUCCEEDED,
            )
        )

    def _workflow_failure_message(self, node_run: NodeRun) -> str:
        detail = node_run.error or node_run.status.value
        return f"NodeRun {node_run.id} failed: {detail}"

    def _node_run_error_context(
        self,
        operation: str,
        node_run: NodeRun,
        error: str,
        retryable: bool,
    ) -> ErrorContext:
        return ErrorContext(
            operation=operation,
            error_code="node_run_execution_failed",
            error_message=error,
            retryable=retryable,
            details={
                "workflow_run_id": str(node_run.workflow_run_id),
                "node_run_id": str(node_run.id),
                "workflow_node_id": node_run.workflow_node_id,
                "operator_id": node_run.operator_id,
                "operator_version": node_run.operator_version,
                "node_run_status": node_run.status.value,
            },
        )

    async def _workflow_output_artifact_refs(
        self,
        uow: StudioUnitOfWorkPort,
        node_runs: list[NodeRun],
    ) -> list[ArtifactRef]:
        workflow_version_id = node_runs[0].metadata.get("workflow_version_id")
        if not isinstance(workflow_version_id, str):
            return await self._flatten_node_run_output_refs(uow, node_runs)

        workflow_version = await uow.workflow_versions.get(UUID(workflow_version_id))
        if workflow_version is None:
            return await self._flatten_node_run_output_refs(uow, node_runs)

        upstream_node_ids = {
            edge.from_node_id for edge in workflow_version.definition_snapshot.edges
        }
        leaf_node_ids = {
            node.id
            for node in workflow_version.definition_snapshot.nodes
            if node.id not in upstream_node_ids
        }
        leaf_node_runs = [
            node_run
            for node_run in node_runs
            if node_run.workflow_node_id in leaf_node_ids
        ]
        if not leaf_node_runs:
            return await self._flatten_node_run_output_refs(uow, node_runs)
        return await self._flatten_node_run_output_refs(uow, leaf_node_runs)

    async def _flatten_node_run_output_refs(
        self,
        uow: StudioUnitOfWorkPort,
        node_runs: list[NodeRun],
    ) -> list[ArtifactRef]:
        output_refs: list[ArtifactRef] = []
        for node_run in node_runs:
            for artifact_ref in node_run.output_artifact_refs.values():
                output_refs.extend(
                    await self._flatten_artifact_port_ref(uow, artifact_ref)
                )
        return output_refs

    async def _flatten_artifact_port_ref(
        self,
        uow: StudioUnitOfWorkPort,
        artifact_ref: ArtifactPortRef,
    ) -> list[ArtifactRef]:
        if isinstance(artifact_ref, ArtifactRef):
            return [artifact_ref]
        if isinstance(artifact_ref, list):
            return list(artifact_ref)

        sequence = await uow.artifact_sequences.get(artifact_ref.sequence_id)
        return list(sequence.item_refs) if sequence is not None else []
