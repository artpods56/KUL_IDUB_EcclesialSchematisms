from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from starlette import status

from notarius_api import dependencies as deps
from notarius_api.schemas.platform import (
    InputAssemblyTraceResponse,
    InvocationTraceResponse,
    NodeRunCreate,
    NodeRunExecutionResponse,
    NodeRunRetryResponse,
    NodeRunResponse,
    WorkflowRunResponse,
    artifact_ref_map_to_domain,
)
from notarius_core.domain.errors import ConflictError
from notarius_core.domain.models import NodeRun, NodeRunStatus
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_messaging.contracts import RunEventType
from notarius_messaging.outbox import (
    node_run_event_outbox_message,
    node_run_execute_requested_outbox_message,
    workflow_run_event_outbox_message,
)
from notarius_worker.node_execution import NodeRunExecutionError, NodeRunExecutor

router = APIRouter(tags=["node runs"])

NodeRunExecutorDependency = Annotated[
    NodeRunExecutor,
    Depends(deps.create_node_run_executor),
]
UowFactoryDependency = Annotated[
    Callable[[], StudioUnitOfWorkPort],
    Depends(deps.create_uow_factory),
]


@router.post(
    "/workflow-runs/{workflow_run_id}/node-runs",
    response_model=NodeRunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_node_run(
    workflow_run_id: UUID,
    body: NodeRunCreate,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> NodeRunResponse:
    async with uow:
        await deps.get_workflow_run_or_404(uow, workflow_run_id)
        node_run = NodeRun(
            workflow_run_id=workflow_run_id,
            workflow_node_id=body.workflow_node_id,
            operator_id=body.operator_id,
            operator_version=body.operator_version,
            input_artifact_refs=artifact_ref_map_to_domain(body.input_artifact_refs),
            metadata=body.metadata,
        )
        await uow.node_runs.add(node_run)
        await uow.commit()
        return NodeRunResponse.from_domain(node_run)


@router.get(
    "/workflow-runs/{workflow_run_id}/node-runs",
    response_model=list[NodeRunResponse],
)
async def list_workflow_run_node_runs(
    workflow_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[NodeRunResponse]:
    async with uow:
        await deps.get_workflow_run_or_404(uow, workflow_run_id)
        return [
            NodeRunResponse.from_domain(node_run)
            for node_run in await uow.node_runs.list_for_workflow_run(workflow_run_id)
        ]


@router.post(
    "/node-runs/next/execute",
    response_model=NodeRunExecutionResponse,
)
async def execute_next_node_run(
    executor: NodeRunExecutorDependency,
    uow_factory: UowFactoryDependency,
) -> NodeRunExecutionResponse:
    async with uow_factory() as uow:
        next_node_run = await uow.node_runs.next_queued()
        if next_node_run is None:
            return NodeRunExecutionResponse(
                requested_node_run_id=None,
                processed_node_run_id=None,
                node_run=None,
            )
        node_run_id = next_node_run.id

    error = await _execute_node_run(executor, node_run_id)
    return await _execution_response(
        uow_factory,
        requested_node_run_id=None,
        processed_node_run_id=node_run_id,
        error=error,
    )


@router.post(
    "/node-runs/{node_run_id}/execute",
    response_model=NodeRunExecutionResponse,
)
async def execute_node_run(
    node_run_id: UUID,
    executor: NodeRunExecutorDependency,
    uow_factory: UowFactoryDependency,
) -> NodeRunExecutionResponse:
    async with uow_factory() as uow:
        await deps.get_node_run_or_404(uow, node_run_id)

    error = await _execute_node_run(executor, node_run_id)
    return await _execution_response(
        uow_factory,
        requested_node_run_id=node_run_id,
        processed_node_run_id=node_run_id,
        error=error,
    )


@router.get("/node-runs/{node_run_id}", response_model=NodeRunResponse)
async def get_node_run(
    node_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> NodeRunResponse:
    async with uow:
        node_run = await deps.get_node_run_or_404(uow, node_run_id)
        return NodeRunResponse.from_domain(node_run)


@router.post(
    "/node-runs/{node_run_id}/retry",
    response_model=NodeRunRetryResponse,
)
async def retry_node_run(
    node_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> NodeRunRetryResponse:
    async with uow:
        node_run = await deps.get_node_run_or_404(uow, node_run_id)
        workflow_run = await deps.get_workflow_run_or_404(uow, node_run.workflow_run_id)
        if node_run.status != NodeRunStatus.FAILED_RETRYABLE:
            raise ConflictError(
                f"Cannot retry node run {node_run.id}: status is "
                f"{node_run.status.value}"
            )
        if node_run.attempt_count >= node_run.max_attempts:
            raise ConflictError(
                f"Cannot retry node run {node_run.id}: attempt limit "
                f"{node_run.attempt_count}/{node_run.max_attempts} reached"
            )

        node_run.mark_queued()
        workflow_run.mark_queued()
        outbox_message = node_run_execute_requested_outbox_message(
            workflow_run,
            node_run,
        )
        await uow.node_runs.update(node_run)
        await uow.workflow_runs.update(workflow_run)
        await uow.outbox_messages.add(
            workflow_run_event_outbox_message(
                workflow_run,
                RunEventType.QUEUED,
            )
        )
        await uow.outbox_messages.add(
            node_run_event_outbox_message(
                node_run,
                RunEventType.QUEUED,
            )
        )
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()
        return NodeRunRetryResponse(
            workflow_run=WorkflowRunResponse.from_domain(workflow_run),
            node_run=NodeRunResponse.from_domain(node_run),
            outbox_message_id=outbox_message.id,
        )


@router.get(
    "/node-runs/{node_run_id}/input-assembly-traces",
    response_model=list[InputAssemblyTraceResponse],
)
async def list_node_run_input_assembly_traces(
    node_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[InputAssemblyTraceResponse]:
    async with uow:
        await deps.get_node_run_or_404(uow, node_run_id)
        return [
            InputAssemblyTraceResponse.from_domain(trace)
            for trace in await uow.input_assembly_traces.list_for_node_run(node_run_id)
        ]


@router.get(
    "/node-runs/{node_run_id}/invocation-traces",
    response_model=list[InvocationTraceResponse],
)
async def list_node_run_invocation_traces(
    node_run_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> list[InvocationTraceResponse]:
    async with uow:
        await deps.get_node_run_or_404(uow, node_run_id)
        return [
            InvocationTraceResponse.from_domain(trace)
            for trace in await uow.invocation_traces.list_for_node_run(node_run_id)
        ]


async def _execute_node_run(
    executor: NodeRunExecutor,
    node_run_id: UUID,
) -> str | None:
    try:
        await executor.execute_node_run(node_run_id)
    except NodeRunExecutionError as exc:
        return str(exc)
    return None


async def _execution_response(
    uow_factory: Callable[[], StudioUnitOfWorkPort],
    requested_node_run_id: UUID | None,
    processed_node_run_id: UUID | None,
    error: str | None,
) -> NodeRunExecutionResponse:
    node_run_response = None
    if processed_node_run_id is not None:
        async with uow_factory() as uow:
            node_run = await uow.node_runs.get(processed_node_run_id)
        if node_run is not None:
            node_run_response = NodeRunResponse.from_domain(node_run)

    return NodeRunExecutionResponse(
        requested_node_run_id=requested_node_run_id,
        processed_node_run_id=processed_node_run_id,
        node_run=node_run_response,
        error=error,
    )
