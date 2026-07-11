import json
from collections.abc import Callable
from uuid import UUID

from notarius_api.schemas.platform import (
    ArtifactResponse,
    ArtifactSequenceResponse,
    InputAssemblyTraceResponse,
    InvocationTraceResponse,
    NodeRunResponse,
    OutputArtifactPayloadResponse,
    OutputArtifactResponse,
    WorkflowRunOutputBundleResponse,
    WorkflowRunResponse,
    WorkflowRunTraceBundleResponse,
)
from notarius_core.domain.errors import NotFoundError
from notarius_core.domain.models import Artifact, ArtifactSequence
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_storage import ArtifactPayloadStoragePort, parse_artifact_payload_ref


class WorkflowRunOutputBundleService:
    def __init__(
        self,
        uow_factory: Callable[[], StudioUnitOfWorkPort],
        storage: ArtifactPayloadStoragePort,
    ) -> None:
        self.uow_factory = uow_factory
        self.storage = storage

    async def build_workflow_run_output_bundle(
        self,
        workflow_run_id: UUID,
        *,
        artifact_type: str | None,
        include_payloads: bool,
        include_text_payloads: bool,
        include_traces: bool,
    ) -> WorkflowRunOutputBundleResponse:
        trace_bundles: list[WorkflowRunTraceBundleResponse] = []
        async with self.uow_factory() as uow:
            workflow_run = await uow.workflow_runs.get(workflow_run_id)
            if workflow_run is None:
                raise NotFoundError("WorkflowRun", str(workflow_run_id))
            artifacts = await uow.artifacts.list_for_workflow_run(workflow_run_id)
            if include_traces:
                node_runs = await uow.node_runs.list_for_workflow_run(workflow_run_id)
                for node_run in node_runs:
                    input_assembly_traces = (
                        await uow.input_assembly_traces.list_for_node_run(node_run.id)
                    )
                    invocation_traces = await uow.invocation_traces.list_for_node_run(
                        node_run.id
                    )
                    trace_bundles.append(
                        WorkflowRunTraceBundleResponse(
                            node_run=NodeRunResponse.from_domain(node_run),
                            input_assembly_traces=[
                                InputAssemblyTraceResponse.from_domain(trace)
                                for trace in input_assembly_traces
                            ],
                            invocation_traces=[
                                InvocationTraceResponse.from_domain(trace)
                                for trace in invocation_traces
                            ],
                        )
                    )

        filtered_artifacts = [
            artifact
            for artifact in artifacts
            if artifact_type is None or artifact.artifact_type == artifact_type
        ]
        filtered_artifact_ids = {artifact.id for artifact in filtered_artifacts}
        async with self.uow_factory() as uow:
            artifact_sequences = await self._output_artifact_sequences(
                uow,
                filtered_artifacts,
                filtered_artifact_ids,
            )
        return WorkflowRunOutputBundleResponse(
            workflow_run=WorkflowRunResponse.from_domain(workflow_run),
            artifacts=[
                OutputArtifactResponse(
                    artifact=ArtifactResponse.from_domain(artifact),
                    payload=self._output_artifact_payload(
                        artifact,
                        include_payloads=include_payloads,
                        include_text_payloads=include_text_payloads,
                    ),
                )
                for artifact in filtered_artifacts
            ],
            artifact_sequences=[
                ArtifactSequenceResponse.from_domain(sequence)
                for sequence in artifact_sequences
            ],
            traces=trace_bundles,
        )

    async def _output_artifact_sequences(
        self,
        uow: StudioUnitOfWorkPort,
        artifacts: list[Artifact],
        artifact_ids: set[UUID],
    ) -> list[ArtifactSequence]:
        artifact_types = sorted({artifact.artifact_type for artifact in artifacts})
        sequences: list[ArtifactSequence] = []
        seen_sequence_ids: set[UUID] = set()
        for artifact_type in artifact_types:
            for sequence in await uow.artifact_sequences.list_by_artifact_type(
                artifact_type
            ):
                if sequence.id in seen_sequence_ids:
                    continue
                if not sequence.item_refs:
                    continue
                if all(ref.artifact_id in artifact_ids for ref in sequence.item_refs):
                    sequences.append(sequence)
                    seen_sequence_ids.add(sequence.id)
        return sequences

    def _output_artifact_payload(
        self,
        artifact: Artifact,
        *,
        include_payloads: bool,
        include_text_payloads: bool,
    ) -> OutputArtifactPayloadResponse | None:
        return artifact_payload_response(
            self.storage,
            artifact,
            include_payloads=include_payloads,
            include_text_payloads=include_text_payloads,
        )


def artifact_payload_response(
    storage: ArtifactPayloadStoragePort,
    artifact: Artifact,
    *,
    include_payloads: bool,
    include_text_payloads: bool,
) -> OutputArtifactPayloadResponse | None:
    content_type = artifact.metadata.get("content_type")
    if not isinstance(content_type, str):
        content_type = "application/octet-stream"
    if not include_payloads and not (
        include_text_payloads and content_type.startswith("text/")
    ):
        return None

    byte_size = artifact.metadata.get("byte_size")
    if type(byte_size) is not int:
        byte_size = 0
    try:
        location = parse_artifact_payload_ref(artifact.payload_ref)
        stored = storage.load(location.bucket, location.key)
    except (OSError, ValueError) as exc:
        return OutputArtifactPayloadResponse(
            content_type=content_type,
            byte_size=byte_size,
            error=str(exc),
        )

    if content_type == "application/json":
        try:
            payload = json.loads(stored.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return OutputArtifactPayloadResponse(
                content_type=content_type,
                byte_size=stored.byte_size,
                error=f"JSON payload could not be decoded: {exc}",
            )
        return OutputArtifactPayloadResponse(
            content_type=content_type,
            byte_size=stored.byte_size,
            json_payload=payload,
        )
    if content_type.startswith("text/") or content_type == "application/x-ndjson":
        try:
            text = stored.payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            return OutputArtifactPayloadResponse(
                content_type=content_type,
                byte_size=stored.byte_size,
                error=f"Text payload could not be decoded: {exc}",
            )
        return OutputArtifactPayloadResponse(
            content_type=content_type,
            byte_size=stored.byte_size,
            text=text,
        )

    return OutputArtifactPayloadResponse(
        content_type=content_type,
        byte_size=stored.byte_size,
    )
