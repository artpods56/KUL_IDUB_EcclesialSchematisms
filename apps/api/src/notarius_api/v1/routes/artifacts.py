import json
import re
from typing import Annotated
from uuid import UUID
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.services.output_bundles import artifact_payload_response
from notarius_api.schemas.platform import (
    ArtifactCreate,
    ArtifactGraphEdgeResponse,
    ArtifactGraphResponse,
    ArtifactInspectionResponse,
    ArtifactJsonPayloadCreate,
    ArtifactResponse,
    ArtifactSequenceCreate,
    ArtifactSequenceResponse,
    NodeRunResponse,
    WorkflowRunResponse,
)
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import (
    Artifact,
    ArtifactPortRef,
    ArtifactRef,
    ArtifactSequence,
    ArtifactSequenceRef,
    NodeRun,
    WorkflowRun,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_storage import (
    ArtifactPayloadStoragePort,
    SaveArtifactPayloadCommand,
    artifact_payload_ref,
    parse_artifact_payload_ref,
)

router = APIRouter(tags=["artifacts"])


@router.post(
    "/artifacts",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_artifact(
    body: ArtifactCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ArtifactResponse:
    async with uow:
        if body.workflow_run_id is not None:
            await deps.get_workflow_run_or_404(uow, body.workflow_run_id)
        if body.producer_node_run_id is not None:
            await deps.get_node_run_or_404(uow, body.producer_node_run_id)
        artifact = Artifact(
            artifact_type=body.artifact_type,
            schema_version=body.schema_version,
            workflow_run_id=body.workflow_run_id,
            producer_node_run_id=body.producer_node_run_id,
            payload_ref=body.payload_ref,
            producer_operator_id=body.producer_operator_id,
            producer_operator_version=body.producer_operator_version,
            input_artifact_ids=body.input_artifact_ids,
            content_hash=body.content_hash,
            preview_ref=body.preview_ref,
            metadata=body.metadata,
        )
        await uow.artifacts.add(artifact)
        await uow.commit()
        return ArtifactResponse.from_domain(artifact)


@router.post(
    "/artifacts/json",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_json_artifact(
    body: ArtifactJsonPayloadCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    storage: Annotated[
        ArtifactPayloadStoragePort,
        Depends(deps.get_artifact_payload_storage),
    ],
) -> ArtifactResponse:
    async with uow:
        artifact_id = uuid4()
        await _validate_artifact_parent_refs(
            uow,
            workflow_run_id=body.workflow_run_id,
            producer_node_run_id=body.producer_node_run_id,
        )
        try:
            payload = json.dumps(body.payload, indent=2, sort_keys=True).encode(
                "utf-8"
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Artifact JSON payload is not serializable: {exc}") from exc

        key = body.key or f"{body.artifact_type}/{artifact_id}/payload.json"
        stored = storage.save(
            SaveArtifactPayloadCommand(
                bucket=body.bucket,
                key=key,
                payload=payload,
            )
        )
        artifact = Artifact(
            id=artifact_id,
            artifact_type=body.artifact_type,
            schema_version=body.schema_version,
            workflow_run_id=body.workflow_run_id,
            producer_node_run_id=body.producer_node_run_id,
            payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
            producer_operator_id=body.producer_operator_id,
            producer_operator_version=body.producer_operator_version,
            input_artifact_ids=body.input_artifact_ids,
            content_hash=stored.sha256,
            preview_ref=body.preview_ref,
            metadata={
                **body.metadata,
                "content_type": body.content_type,
                "byte_size": stored.byte_size,
            },
        )
        await uow.artifacts.add(artifact)
        await uow.commit()
        return ArtifactResponse.from_domain(artifact)


@router.post(
    "/artifacts/upload",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_artifact_payload(
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    storage: Annotated[
        ArtifactPayloadStoragePort,
        Depends(deps.get_artifact_payload_storage),
    ],
    file: UploadFile = File(...),
    artifact_type: str = Form(...),
    schema_version: int = Form(...),
    workflow_run_id: UUID | None = Form(None),
    producer_node_run_id: UUID | None = Form(None),
    producer_operator_id: str | None = Form(None),
    producer_operator_version: str | None = Form(None),
    metadata_json: str | None = Form(None),
    bucket: str = Form("script-artifacts"),
    key: str | None = Form(None),
) -> ArtifactResponse:
    async with uow:
        artifact_id = uuid4()
        await _validate_artifact_parent_refs(
            uow,
            workflow_run_id=workflow_run_id,
            producer_node_run_id=producer_node_run_id,
        )
        content = await file.read()
        if not content:
            raise ValidationError("Uploaded artifact payload is empty")
        metadata = _metadata_from_json(metadata_json)
        filename = _safe_filename(file.filename or "payload.bin")
        stored = storage.save(
            SaveArtifactPayloadCommand(
                bucket=bucket,
                key=key or f"{artifact_type}/{artifact_id}/{filename}",
                payload=content,
            )
        )
        artifact = Artifact(
            id=artifact_id,
            artifact_type=artifact_type,
            schema_version=schema_version,
            workflow_run_id=workflow_run_id,
            producer_node_run_id=producer_node_run_id,
            payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
            producer_operator_id=producer_operator_id,
            producer_operator_version=producer_operator_version,
            content_hash=stored.sha256,
            metadata={
                **metadata,
                "filename": filename,
                "content_type": file.content_type or "application/octet-stream",
                "byte_size": stored.byte_size,
            },
        )
        await uow.artifacts.add(artifact)
        await uow.commit()
        return ArtifactResponse.from_domain(artifact)


@router.get("/artifacts/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ArtifactResponse:
    async with uow:
        artifact = await deps.get_artifact_or_404(uow, artifact_id)
        return ArtifactResponse.from_domain(artifact)


@router.get("/artifacts/{artifact_id}/inspect", response_model=ArtifactInspectionResponse)
async def inspect_artifact(
    artifact_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    storage: Annotated[
        ArtifactPayloadStoragePort,
        Depends(deps.get_artifact_payload_storage),
    ],
    include_payload: bool = False,
    include_text_payload: bool = False,
    include_lineage: bool = False,
) -> ArtifactInspectionResponse:
    async with uow:
        artifact = await deps.get_artifact_or_404(uow, artifact_id)
        lineage = None
        if include_lineage:
            workflow_run = (
                await uow.workflow_runs.get(artifact.workflow_run_id)
                if artifact.workflow_run_id is not None
                else None
            )
            graph = _ArtifactGraphBuilder()
            await _add_artifact_lineage(uow, graph, artifact)
            lineage = graph.to_response(
                workflow_run=workflow_run,
                root_artifact=artifact,
            )

    payload = artifact_payload_response(
        storage,
        artifact,
        include_payloads=include_payload,
        include_text_payloads=include_text_payload,
    )
    return ArtifactInspectionResponse(
        artifact=ArtifactResponse.from_domain(artifact),
        payload=payload,
        lineage=lineage,
    )


@router.get("/artifacts/{artifact_id}/lineage", response_model=ArtifactGraphResponse)
async def get_artifact_lineage(
    artifact_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ArtifactGraphResponse:
    async with uow:
        root_artifact = await deps.get_artifact_or_404(uow, artifact_id)
        workflow_run = (
            await uow.workflow_runs.get(root_artifact.workflow_run_id)
            if root_artifact.workflow_run_id is not None
            else None
        )
        graph = _ArtifactGraphBuilder()
        await _add_artifact_lineage(uow, graph, root_artifact)
        return graph.to_response(
            workflow_run=workflow_run,
            root_artifact=root_artifact,
        )


@router.get("/artifacts/{artifact_id}/payload")
async def get_artifact_payload(
    artifact_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    storage: Annotated[
        ArtifactPayloadStoragePort,
        Depends(deps.get_artifact_payload_storage),
    ],
) -> Response:
    async with uow:
        artifact = await deps.get_artifact_or_404(uow, artifact_id)

    try:
        location = parse_artifact_payload_ref(artifact.payload_ref)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    stored = storage.load(location.bucket, location.key)
    content_type = artifact.metadata.get("content_type")
    media_type = content_type if isinstance(content_type, str) else "application/octet-stream"
    return Response(content=stored.payload, media_type=media_type)


@router.get(
    "/workflow-runs/{workflow_run_id}/artifacts",
    response_model=list[ArtifactResponse],
)
async def list_workflow_run_artifacts(
    workflow_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[ArtifactResponse]:
    async with uow:
        await deps.get_workflow_run_or_404(uow, workflow_run_id)
        return [
            ArtifactResponse.from_domain(artifact)
            for artifact in await uow.artifacts.list_for_workflow_run(workflow_run_id)
        ]


@router.get(
    "/workflow-runs/{workflow_run_id}/artifact-graph",
    response_model=ArtifactGraphResponse,
)
async def get_workflow_run_artifact_graph(
    workflow_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ArtifactGraphResponse:
    async with uow:
        workflow_run = await deps.get_workflow_run_or_404(uow, workflow_run_id)
        node_runs = await uow.node_runs.list_for_workflow_run(workflow_run_id)
        artifacts = await uow.artifacts.list_for_workflow_run(workflow_run_id)
        graph = _ArtifactGraphBuilder()
        for artifact in artifacts:
            graph.add_artifact(artifact)
            for input_artifact_id in artifact.input_artifact_ids:
                input_artifact = await uow.artifacts.get(input_artifact_id)
                if input_artifact is None:
                    continue
                graph.add_artifact(input_artifact)
                graph.add_edge(
                    edge_type="artifact_input",
                    from_kind="artifact",
                    from_id=input_artifact.id,
                    to_kind="artifact",
                    to_id=artifact.id,
                )
            if artifact.producer_node_run_id is not None:
                node_run = await uow.node_runs.get(artifact.producer_node_run_id)
                if node_run is not None:
                    graph.add_node_run(node_run)
                    if not _node_run_outputs_artifact(node_run, artifact.id):
                        graph.add_edge(
                            edge_type="node_output",
                            from_kind="node_run",
                            from_id=node_run.id,
                            to_kind="artifact",
                            to_id=artifact.id,
                        )

        for sequence_ref in workflow_run.input_artifact_sequence_refs:
            await _add_sequence_ref(uow, graph, sequence_ref, None, None)

        for node_run in node_runs:
            graph.add_node_run(node_run)
            await _add_node_run_port_refs(
                uow,
                graph,
                node_run,
                node_run.input_artifact_refs,
                incoming=True,
            )
            await _add_node_run_port_refs(
                uow,
                graph,
                node_run,
                node_run.output_artifact_refs,
                incoming=False,
            )

        return graph.to_response(workflow_run=workflow_run)


@router.get("/node-runs/{node_run_id}/artifacts", response_model=list[ArtifactResponse])
async def list_node_run_artifacts(
    node_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[ArtifactResponse]:
    async with uow:
        await deps.get_node_run_or_404(uow, node_run_id)
        return [
            ArtifactResponse.from_domain(artifact)
            for artifact in await uow.artifacts.list_for_node_run(node_run_id)
        ]


@router.post(
    "/artifact-sequences",
    response_model=ArtifactSequenceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_artifact_sequence(
    body: ArtifactSequenceCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ArtifactSequenceResponse:
    async with uow:
        item_refs = [item.to_domain() for item in body.item_refs]
        for item_ref in item_refs:
            await _validate_artifact_ref(uow, item_ref)
        sequence = ArtifactSequence(
            artifact_type=body.artifact_type,
            schema_version=body.schema_version,
            item_refs=item_refs,
            ordered=body.ordered,
            index_key=body.index_key,
            metadata=body.metadata,
        )
        await uow.artifact_sequences.add(sequence)
        await uow.commit()
        return ArtifactSequenceResponse.from_domain(sequence)


@router.get(
    "/artifact-sequences/{sequence_id}",
    response_model=ArtifactSequenceResponse,
)
async def get_artifact_sequence(
    sequence_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> ArtifactSequenceResponse:
    async with uow:
        sequence = await deps.get_artifact_sequence_or_404(uow, sequence_id)
        return ArtifactSequenceResponse.from_domain(sequence)


@router.get("/artifact-sequences", response_model=list[ArtifactSequenceResponse])
async def list_artifact_sequences_by_type(
    artifact_type: str,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[ArtifactSequenceResponse]:
    async with uow:
        return [
            ArtifactSequenceResponse.from_domain(sequence)
            for sequence in await uow.artifact_sequences.list_by_artifact_type(
                artifact_type
            )
        ]


async def _validate_artifact_ref(
    uow: StudioUnitOfWorkPort,
    item_ref: ArtifactRef,
) -> None:
    artifact = await deps.get_artifact_or_404(uow, item_ref.artifact_id)
    if artifact.artifact_type != item_ref.artifact_type:
        raise ValidationError(
            "Artifact sequence item type mismatch for "
            f"{item_ref.artifact_id}: expected {item_ref.artifact_type}, "
            f"got {artifact.artifact_type}"
        )
    if artifact.schema_version != item_ref.schema_version:
        raise ValidationError(
            "Artifact sequence item schema version mismatch for "
            f"{item_ref.artifact_id}: expected {item_ref.schema_version}, "
            f"got {artifact.schema_version}"
        )
    if item_ref.content_hash is not None and artifact.content_hash != item_ref.content_hash:
        raise ValidationError(
            "Artifact sequence item content hash mismatch for "
            f"{item_ref.artifact_id}: expected {item_ref.content_hash}, "
            f"got {artifact.content_hash}"
        )


async def _validate_artifact_parent_refs(
    uow: StudioUnitOfWorkPort,
    *,
    workflow_run_id: UUID | None,
    producer_node_run_id: UUID | None,
) -> None:
    if workflow_run_id is not None:
        await deps.get_workflow_run_or_404(uow, workflow_run_id)
    if producer_node_run_id is not None:
        await deps.get_node_run_or_404(uow, producer_node_run_id)


def _metadata_from_json(metadata_json: str | None) -> dict[str, object]:
    if metadata_json is None or metadata_json.strip() == "":
        return {}
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"metadata_json is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValidationError("metadata_json must decode to an object")
    return metadata


def _safe_filename(filename: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip(".-")
    return sanitized or "payload.bin"


class _ArtifactGraphBuilder:
    def __init__(self) -> None:
        self.artifacts: dict[UUID, Artifact] = {}
        self.node_runs: dict[UUID, NodeRun] = {}
        self.artifact_sequences: dict[UUID, ArtifactSequence] = {}
        self.edges: list[ArtifactGraphEdgeResponse] = []
        self.edge_keys: set[tuple[str, str, UUID, str, UUID, str | None]] = set()
        self.expanded_artifact_ids: set[UUID] = set()

    def add_artifact(self, artifact: Artifact) -> None:
        self.artifacts[artifact.id] = artifact

    def add_node_run(self, node_run: NodeRun) -> None:
        self.node_runs[node_run.id] = node_run

    def add_sequence(self, sequence: ArtifactSequence) -> None:
        self.artifact_sequences[sequence.id] = sequence

    def add_edge(
        self,
        *,
        edge_type: str,
        from_kind: str,
        from_id: UUID,
        to_kind: str,
        to_id: UUID,
        port_name: str | None = None,
    ) -> None:
        edge_key = (edge_type, from_kind, from_id, to_kind, to_id, port_name)
        if edge_key in self.edge_keys:
            return
        self.edge_keys.add(edge_key)
        self.edges.append(
            ArtifactGraphEdgeResponse(
                edge_type=edge_type,
                from_kind=from_kind,
                from_id=from_id,
                to_kind=to_kind,
                to_id=to_id,
                port_name=port_name,
            )
        )

    def to_response(
        self,
        *,
        workflow_run: WorkflowRun | None = None,
        root_artifact: Artifact | None = None,
    ) -> ArtifactGraphResponse:
        return ArtifactGraphResponse(
            workflow_run=(
                WorkflowRunResponse.from_domain(workflow_run)
                if workflow_run is not None
                else None
            ),
            root_artifact=(
                ArtifactResponse.from_domain(root_artifact)
                if root_artifact is not None
                else None
            ),
            node_runs=[
                NodeRunResponse.from_domain(node_run)
                for node_run in sorted(
                    self.node_runs.values(),
                    key=lambda value: value.queued_at,
                )
            ],
            artifacts=[
                ArtifactResponse.from_domain(artifact)
                for artifact in sorted(
                    self.artifacts.values(),
                    key=lambda value: value.created_at,
                )
            ],
            artifact_sequences=[
                ArtifactSequenceResponse.from_domain(sequence)
                for sequence in sorted(
                    self.artifact_sequences.values(),
                    key=lambda value: value.created_at,
                )
            ],
            edges=self.edges,
        )


async def _add_artifact_lineage(
    uow: StudioUnitOfWorkPort,
    graph: _ArtifactGraphBuilder,
    artifact: Artifact,
) -> None:
    if artifact.id in graph.expanded_artifact_ids:
        return
    graph.add_artifact(artifact)
    graph.expanded_artifact_ids.add(artifact.id)
    for input_artifact_id in artifact.input_artifact_ids:
        input_artifact = await uow.artifacts.get(input_artifact_id)
        if input_artifact is None:
            continue
        graph.add_edge(
            edge_type="artifact_input",
            from_kind="artifact",
            from_id=input_artifact.id,
            to_kind="artifact",
            to_id=artifact.id,
        )
        await _add_artifact_lineage(uow, graph, input_artifact)

    if artifact.producer_node_run_id is None:
        return
    node_run = await uow.node_runs.get(artifact.producer_node_run_id)
    if node_run is None:
        return
    graph.add_node_run(node_run)
    graph.add_edge(
        edge_type="node_output",
        from_kind="node_run",
        from_id=node_run.id,
        to_kind="artifact",
        to_id=artifact.id,
    )
    await _add_node_run_port_refs(
        uow,
        graph,
        node_run,
        node_run.input_artifact_refs,
        incoming=True,
        include_artifact_lineage=True,
    )


async def _add_node_run_port_refs(
    uow: StudioUnitOfWorkPort,
    graph: _ArtifactGraphBuilder,
    node_run: NodeRun,
    refs_by_port: dict[str, ArtifactPortRef],
    *,
    incoming: bool,
    include_artifact_lineage: bool = False,
) -> None:
    for port_name, port_ref in refs_by_port.items():
        if isinstance(port_ref, ArtifactSequenceRef):
            await _add_sequence_ref(
                uow,
                graph,
                port_ref,
                node_run,
                port_name,
                incoming=incoming,
                include_artifact_lineage=include_artifact_lineage,
            )
            continue

        artifact_refs = port_ref if isinstance(port_ref, list) else [port_ref]
        for artifact_ref in artifact_refs:
            artifact = await uow.artifacts.get(artifact_ref.artifact_id)
            if artifact is None:
                continue
            graph.add_artifact(artifact)
            _add_node_artifact_edge(
                graph,
                node_run,
                artifact.id,
                port_name,
                incoming=incoming,
            )
            if include_artifact_lineage:
                await _add_artifact_lineage(uow, graph, artifact)


async def _add_sequence_ref(
    uow: StudioUnitOfWorkPort,
    graph: _ArtifactGraphBuilder,
    sequence_ref: ArtifactSequenceRef,
    node_run: NodeRun | None,
    port_name: str | None,
    *,
    incoming: bool = True,
    include_artifact_lineage: bool = False,
) -> None:
    sequence = await uow.artifact_sequences.get(sequence_ref.sequence_id)
    if sequence is None:
        return
    graph.add_sequence(sequence)
    if node_run is not None:
        edge_type = "node_input" if incoming else "node_output"
        if incoming:
            graph.add_edge(
                edge_type=edge_type,
                from_kind="artifact_sequence",
                from_id=sequence.id,
                to_kind="node_run",
                to_id=node_run.id,
                port_name=port_name,
            )
        else:
            graph.add_edge(
                edge_type=edge_type,
                from_kind="node_run",
                from_id=node_run.id,
                to_kind="artifact_sequence",
                to_id=sequence.id,
                port_name=port_name,
            )

    for item_ref in sequence.item_refs:
        artifact = await uow.artifacts.get(item_ref.artifact_id)
        if artifact is None:
            continue
        graph.add_artifact(artifact)
        graph.add_edge(
            edge_type="artifact_sequence_item",
            from_kind="artifact_sequence",
            from_id=sequence.id,
            to_kind="artifact",
            to_id=artifact.id,
        )
        if include_artifact_lineage:
            await _add_artifact_lineage(uow, graph, artifact)


def _add_node_artifact_edge(
    graph: _ArtifactGraphBuilder,
    node_run: NodeRun,
    artifact_id: UUID,
    port_name: str,
    *,
    incoming: bool,
) -> None:
    if incoming:
        graph.add_edge(
            edge_type="node_input",
            from_kind="artifact",
            from_id=artifact_id,
            to_kind="node_run",
            to_id=node_run.id,
            port_name=port_name,
        )
        return

    graph.add_edge(
        edge_type="node_output",
        from_kind="node_run",
        from_id=node_run.id,
        to_kind="artifact",
        to_id=artifact_id,
        port_name=port_name,
    )


def _node_run_outputs_artifact(node_run: NodeRun, artifact_id: UUID) -> bool:
    for port_ref in node_run.output_artifact_refs.values():
        if isinstance(port_ref, ArtifactRef) and port_ref.artifact_id == artifact_id:
            return True
        if isinstance(port_ref, list):
            if any(item_ref.artifact_id == artifact_id for item_ref in port_ref):
                return True
    return False
