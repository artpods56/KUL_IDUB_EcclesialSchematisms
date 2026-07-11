from functools import partial
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends
from pydantic import ValidationError as PydanticValidationError
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.schemas.platform import (
    ArtifactResponse,
    NodeRunResponse,
    WorkflowDefinitionCreate,
    WorkflowDefinitionResponse,
    WorkflowExecutionPlanResponse,
    WorkflowRunExecutionCreate,
    WorkflowRunExecutionNodeError,
    WorkflowRunExecutionResponse,
    WorkflowRunCreate,
    WorkflowRunOutputBundleResponse,
    WorkflowRunResponse,
    WorkflowRunSummaryError,
    WorkflowRunSummaryResponse,
    WorkflowRunTimelineEventResponse,
    WorkflowRunTimelineResponse,
    WorkflowValidationResponse,
    WorkflowVersionCreate,
    WorkflowVersionResponse,
)
from notarius_api.services.output_bundles import WorkflowRunOutputBundleService
from notarius_api.services.workflow_execution import WorkflowRunExecutionService
from notarius_core.application.workflows import (
    NodeSpecRegistry,
    WorkflowCompiler,
    WorkflowRunLauncher,
)
from notarius_core.domain.errors import ValidationError
from notarius_core.domain.models import (
    Artifact,
    ArtifactSequence,
    ArtifactSequenceRef,
    NodeRun,
    OutboxMessage,
    WorkflowRun,
    WorkflowVersion,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_messaging.contracts import (
    ArtifactEvent,
    DlqMessage,
    NodeRunEvent,
    RunEventType,
    WorkflowRunEvent,
)
from notarius_messaging.outbox import (
    node_run_event_outbox_message,
    node_run_execute_requested_outbox_message,
    workflow_run_event_outbox_message,
)

router = APIRouter(tags=["workflows"])

WorkflowRunExecutionServiceDependency = Annotated[
    WorkflowRunExecutionService,
    Depends(deps.create_workflow_run_execution_service),
]
WorkflowRunOutputBundleServiceDependency = Annotated[
    WorkflowRunOutputBundleService,
    Depends(deps.create_workflow_run_output_bundle_service),
]


@router.post(
    "/workflows",
    response_model=WorkflowDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_workflow_definition(
    body: WorkflowDefinitionCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> WorkflowDefinitionResponse:
    async with uow:
        definition = body.to_domain()
        await uow.workflow_definitions.add(definition)
        await uow.commit()
        return WorkflowDefinitionResponse.from_domain(definition)


@router.post(
    "/workflows/validate",
    response_model=WorkflowValidationResponse,
)
async def validate_workflow_definition(
    body: WorkflowDefinitionCreate,
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> WorkflowValidationResponse:
    definition = body.to_domain()
    workflow_version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )
    try:
        plan = WorkflowCompiler(node_specs).compile(workflow_version, uuid4())
    except ValidationError as exc:
        return WorkflowValidationResponse(
            valid=False,
            errors=[str(exc)],
            node_count=len(definition.nodes),
            edge_count=len(definition.edges),
        )

    return WorkflowValidationResponse(
        valid=True,
        node_count=len(definition.nodes),
        edge_count=len(definition.edges),
        execution_order=[node_run.workflow_node_id for node_run in plan.node_runs],
        execution_plan=WorkflowExecutionPlanResponse.from_compiled_plan(
            plan,
            node_specs,
        ),
    )


@router.get("/workflows", response_model=list[WorkflowDefinitionResponse])
async def list_workflow_definitions(
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[WorkflowDefinitionResponse]:
    async with uow:
        return [
            WorkflowDefinitionResponse.from_domain(definition)
            for definition in await uow.workflow_definitions.list()
        ]


@router.get(
    "/workflows/{workflow_definition_id}",
    response_model=WorkflowDefinitionResponse,
)
async def get_workflow_definition(
    workflow_definition_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> WorkflowDefinitionResponse:
    async with uow:
        definition = await deps.get_workflow_definition_or_404(
            uow,
            workflow_definition_id,
        )
        return WorkflowDefinitionResponse.from_domain(definition)


@router.post(
    "/workflows/{workflow_definition_id}/versions",
    response_model=WorkflowVersionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_workflow_version(
    workflow_definition_id: UUID,
    body: WorkflowVersionCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> WorkflowVersionResponse:
    async with uow:
        definition = await deps.get_workflow_definition_or_404(
            uow,
            workflow_definition_id,
        )
        latest = await uow.workflow_versions.latest_for_definition(definition.id)
        version_number = 1 if latest is None else latest.version_number + 1
        version = WorkflowVersion(
            workflow_definition_id=definition.id,
            version_number=version_number,
            definition_snapshot=definition,
            created_by=body.created_by,
            change_note=body.change_note,
        )
        await uow.workflow_versions.add(version)
        await uow.commit()
        return WorkflowVersionResponse.from_domain(version)


@router.get(
    "/workflows/{workflow_definition_id}/versions",
    response_model=list[WorkflowVersionResponse],
)
async def list_workflow_versions(
    workflow_definition_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[WorkflowVersionResponse]:
    async with uow:
        await deps.get_workflow_definition_or_404(uow, workflow_definition_id)
        return [
            WorkflowVersionResponse.from_domain(version)
            for version in await uow.workflow_versions.list_for_definition(
                workflow_definition_id
            )
        ]


@router.get(
    "/workflow-versions/{workflow_version_id}",
    response_model=WorkflowVersionResponse,
)
async def get_workflow_version(
    workflow_version_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> WorkflowVersionResponse:
    async with uow:
        version = await deps.get_workflow_version_or_404(uow, workflow_version_id)
        return WorkflowVersionResponse.from_domain(version)


@router.post(
    "/workflow-runs",
    response_model=WorkflowRunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_workflow_run(
    body: WorkflowRunCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> WorkflowRunResponse:
    async with uow:
        version = await deps.get_workflow_version_or_404(uow, body.workflow_version_id)
        input_sequences = [
            await _get_matching_artifact_sequence(uow, sequence_ref.to_domain())
            for sequence_ref in body.input_artifact_sequence_refs
        ]
        result = await WorkflowRunLauncher(node_specs).launch(
            uow,
            version,
            [ref.to_domain() for ref in body.input_artifact_refs],
            metadata=body.metadata,
            node_run_outbox_message_builder=node_run_execute_requested_outbox_message,
            workflow_run_queued_event_builder=partial(
                workflow_run_event_outbox_message,
                event_type=RunEventType.QUEUED,
            ),
            node_run_queued_event_builder=partial(
                node_run_event_outbox_message,
                event_type=RunEventType.QUEUED,
            ),
            input_artifact_sequences=input_sequences,
        )
        return WorkflowRunResponse.from_domain(result.workflow_run)


@router.get("/workflow-runs/{workflow_run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(
    workflow_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> WorkflowRunResponse:
    async with uow:
        run = await deps.get_workflow_run_or_404(uow, workflow_run_id)
        return WorkflowRunResponse.from_domain(run)


@router.get(
    "/workflow-runs/{workflow_run_id}/execution-plan",
    response_model=WorkflowExecutionPlanResponse,
)
async def get_workflow_run_execution_plan(
    workflow_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    node_specs: Annotated[NodeSpecRegistry, Depends(deps.get_node_spec_registry)],
) -> WorkflowExecutionPlanResponse:
    async with uow:
        workflow_run = await deps.get_workflow_run_or_404(uow, workflow_run_id)
        node_runs = await uow.node_runs.list_for_workflow_run(workflow_run.id)
        return WorkflowExecutionPlanResponse.from_node_runs(
            workflow_version_id=workflow_run.workflow_version_id,
            workflow_run_id=workflow_run.id,
            node_runs=node_runs,
            node_specs=node_specs,
        )


@router.get(
    "/workflow-runs/{workflow_run_id}/summary",
    response_model=WorkflowRunSummaryResponse,
)
async def summarize_workflow_run(
    workflow_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> WorkflowRunSummaryResponse:
    async with uow:
        workflow_run = await deps.get_workflow_run_or_404(uow, workflow_run_id)
        node_runs = await uow.node_runs.list_for_workflow_run(workflow_run_id)
        artifacts = await uow.artifacts.list_for_workflow_run(workflow_run_id)

    return WorkflowRunSummaryResponse(
        workflow_run=WorkflowRunResponse.from_domain(workflow_run),
        node_runs=[NodeRunResponse.from_domain(node_run) for node_run in node_runs],
        artifacts=[ArtifactResponse.from_domain(artifact) for artifact in artifacts],
        node_run_status_counts=_node_run_status_counts(node_runs),
        artifact_counts=_artifact_counts(artifacts),
        errors=_workflow_run_errors(workflow_run, node_runs),
    )


@router.get(
    "/workflow-runs/{workflow_run_id}/events",
    response_model=WorkflowRunTimelineResponse,
)
async def get_workflow_run_events(
    workflow_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> WorkflowRunTimelineResponse:
    async with uow:
        workflow_run = await deps.get_workflow_run_or_404(uow, workflow_run_id)
        messages = await uow.outbox_messages.list_for_workflow_run(workflow_run_id)

    events: list[WorkflowRunTimelineEventResponse] = []
    for message in messages:
        try:
            event = _workflow_run_timeline_event(message, workflow_run_id)
        except PydanticValidationError as exc:
            event = _malformed_outbox_timeline_event(message, workflow_run_id, exc)
        if event is not None:
            events.append(event)
    return WorkflowRunTimelineResponse(
        workflow_run=WorkflowRunResponse.from_domain(workflow_run),
        events=sorted(
            events,
            key=lambda event: (
                event.occurred_at,
                event.outbox_created_at,
                event.outbox_message_id,
            ),
        ),
    )


@router.get(
    "/workflow-runs/{workflow_run_id}/outputs",
    response_model=WorkflowRunOutputBundleResponse,
)
async def get_workflow_run_outputs(
    workflow_run_id: UUID,
    output_bundle_service: WorkflowRunOutputBundleServiceDependency,
    artifact_type: str | None = None,
    include_payloads: bool = False,
    include_text_payloads: bool = False,
    include_traces: bool = False,
) -> WorkflowRunOutputBundleResponse:
    return await output_bundle_service.build_workflow_run_output_bundle(
        workflow_run_id,
        artifact_type=artifact_type,
        include_payloads=include_payloads,
        include_text_payloads=include_text_payloads,
        include_traces=include_traces,
    )


@router.post(
    "/workflow-runs/{workflow_run_id}/execute",
    response_model=WorkflowRunExecutionResponse,
)
async def execute_workflow_run(
    workflow_run_id: UUID,
    body: WorkflowRunExecutionCreate,
    execution_service: WorkflowRunExecutionServiceDependency,
) -> WorkflowRunExecutionResponse:
    result = await execution_service.execute_workflow_run(
        workflow_run_id,
        max_node_runs=body.max_node_runs,
    )
    return WorkflowRunExecutionResponse(
        workflow_run_id=workflow_run_id,
        workflow_run=WorkflowRunResponse.from_domain(result.workflow_run),
        processed_node_run_ids=result.processed_node_run_ids,
        errors=[
            WorkflowRunExecutionNodeError(
                node_run_id=error.node_run_id,
                error=error.error,
            )
            for error in result.errors
        ],
    )


@router.get(
    "/workflow-versions/{workflow_version_id}/runs",
    response_model=list[WorkflowRunResponse],
)
async def list_workflow_version_runs(
    workflow_version_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[WorkflowRunResponse]:
    async with uow:
        await deps.get_workflow_version_or_404(uow, workflow_version_id)
        return [
            WorkflowRunResponse.from_domain(run)
            for run in await uow.workflow_runs.list_for_version(workflow_version_id)
        ]


async def _get_matching_artifact_sequence(
    uow: StudioUnitOfWorkPort,
    sequence_ref: ArtifactSequenceRef,
) -> ArtifactSequence:
    sequence = await deps.get_artifact_sequence_or_404(uow, sequence_ref.sequence_id)
    if sequence.artifact_type != sequence_ref.artifact_type:
        raise ValidationError(
            "Artifact sequence type mismatch for "
            f"{sequence.id}: expected {sequence_ref.artifact_type}, "
            f"got {sequence.artifact_type}"
        )
    if sequence.schema_version != sequence_ref.schema_version:
        raise ValidationError(
            "Artifact sequence schema version mismatch for "
            f"{sequence.id}: expected {sequence_ref.schema_version}, "
            f"got {sequence.schema_version}"
        )
    return sequence


def _node_run_status_counts(node_runs: list[NodeRun]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node_run in node_runs:
        status_value = node_run.status.value
        counts[status_value] = counts.get(status_value, 0) + 1
    return counts


def _artifact_counts(artifacts: list[Artifact]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        counts[artifact.artifact_type] = counts.get(artifact.artifact_type, 0) + 1
    return counts


def _workflow_run_timeline_event(
    message: OutboxMessage,
    workflow_run_id: UUID,
) -> WorkflowRunTimelineEventResponse | None:
    if message.message_type == WorkflowRunEvent.__name__:
        event = WorkflowRunEvent.model_validate(message.payload)
        if event.workflow_run_id != workflow_run_id:
            return None
        return WorkflowRunTimelineEventResponse(
            outbox_message_id=message.id,
            subject=message.subject,
            message_type=message.message_type,
            outbox_status=message.status,
            outbox_created_at=message.created_at,
            outbox_published_at=message.published_at,
            event_kind="workflow_run",
            event_type=event.event_type.value,
            occurred_at=event.occurred_at,
            workflow_run_id=event.workflow_run_id,
            error=_event_error(event.error),
        )

    if message.message_type == NodeRunEvent.__name__:
        event = NodeRunEvent.model_validate(message.payload)
        if event.workflow_run_id != workflow_run_id:
            return None
        return WorkflowRunTimelineEventResponse(
            outbox_message_id=message.id,
            subject=message.subject,
            message_type=message.message_type,
            outbox_status=message.status,
            outbox_created_at=message.created_at,
            outbox_published_at=message.published_at,
            event_kind="node_run",
            event_type=event.event_type.value,
            occurred_at=event.occurred_at,
            workflow_run_id=event.workflow_run_id,
            node_run_id=event.node_run_id,
            error=_event_error(event.error),
        )

    if message.message_type == ArtifactEvent.__name__:
        event = ArtifactEvent.model_validate(message.payload)
        if event.workflow_run_id != workflow_run_id:
            return None
        return WorkflowRunTimelineEventResponse(
            outbox_message_id=message.id,
            subject=message.subject,
            message_type=message.message_type,
            outbox_status=message.status,
            outbox_created_at=message.created_at,
            outbox_published_at=message.published_at,
            event_kind="artifact",
            event_type=event.event_type.value,
            occurred_at=event.occurred_at,
            workflow_run_id=event.workflow_run_id,
            node_run_id=event.node_run_id,
            artifact_id=event.artifact_id,
            artifact_type=event.artifact_type,
        )

    if message.message_type == DlqMessage.__name__:
        event = DlqMessage.model_validate(message.payload)
        if event.workflow_run_id != workflow_run_id:
            return None
        return WorkflowRunTimelineEventResponse(
            outbox_message_id=message.id,
            subject=message.subject,
            message_type=message.message_type,
            outbox_status=message.status,
            outbox_created_at=message.created_at,
            outbox_published_at=message.published_at,
            event_kind="dead_letter",
            event_type="dead_letter",
            occurred_at=event.failed_at,
            workflow_run_id=event.workflow_run_id,
            node_run_id=event.node_run_id,
            artifact_id=event.artifact_id,
            error=_event_error(event.failure),
            details={
                "original_subject": event.original_subject,
                "original_message_id": event.original_message_id,
                "consumer_name": event.consumer_name,
                "attempt_count": event.attempt_count,
            },
        )

    return None


def _malformed_outbox_timeline_event(
    message: OutboxMessage,
    workflow_run_id: UUID,
    exc: PydanticValidationError,
) -> WorkflowRunTimelineEventResponse:
    return WorkflowRunTimelineEventResponse(
        outbox_message_id=message.id,
        subject=message.subject,
        message_type=message.message_type,
        outbox_status=message.status,
        outbox_created_at=message.created_at,
        outbox_published_at=message.published_at,
        event_kind="malformed_outbox",
        event_type="malformed",
        occurred_at=message.created_at,
        workflow_run_id=workflow_run_id,
        error={
            "operation": "normalize_workflow_run_timeline_event",
            "error_code": "malformed_outbox_payload",
            "error_message": str(exc),
            "retryable": False,
            "details": {},
        },
        details={
            "payload_keys": sorted(message.payload),
            "validation_errors": exc.errors(),
        },
    )


def _event_error(error: object | None) -> dict[str, object] | None:
    if error is None:
        return None
    if hasattr(error, "model_dump"):
        dumped = error.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {"error_message": str(error)}


def _workflow_run_errors(
    workflow_run: WorkflowRun,
    node_runs: list[NodeRun],
) -> list[WorkflowRunSummaryError]:
    errors: list[WorkflowRunSummaryError] = []
    if workflow_run.error is not None:
        errors.append(
            WorkflowRunSummaryError(
                node_run_id=None,
                status=workflow_run.status.value,
                error=workflow_run.error,
            )
        )
    for node_run in node_runs:
        if node_run.error is not None:
            errors.append(
                WorkflowRunSummaryError(
                    node_run_id=node_run.id,
                    status=node_run.status.value,
                    error=node_run.error,
                )
            )
    return errors
