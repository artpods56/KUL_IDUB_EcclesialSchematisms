from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from notarius_core.domain.errors import NotFoundError
from notarius_core.domain.models import (
    NodeRun,
    NodeRunStatus,
    WorkflowRun,
    WorkflowRunStatus,
)
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_worker.node_execution import NodeRunExecutionError, NodeRunExecutor


@dataclass(frozen=True, slots=True)
class WorkflowRunExecutionNodeFailure:
    node_run_id: UUID
    error: str


@dataclass(frozen=True, slots=True)
class WorkflowRunExecutionResult:
    workflow_run: WorkflowRun
    processed_node_run_ids: list[UUID] = field(default_factory=list)
    errors: list[WorkflowRunExecutionNodeFailure] = field(default_factory=list)


class WorkflowRunExecutionService:
    def __init__(
        self,
        executor: NodeRunExecutor,
        uow_factory: Callable[[], StudioUnitOfWorkPort],
    ) -> None:
        self.executor = executor
        self.uow_factory = uow_factory

    async def execute_workflow_run(
        self,
        workflow_run_id: UUID,
        max_node_runs: int,
    ) -> WorkflowRunExecutionResult:
        async with self.uow_factory() as uow:
            workflow_run = await uow.workflow_runs.get(workflow_run_id)
            if workflow_run is None:
                raise NotFoundError("WorkflowRun", str(workflow_run_id))

        processed_node_run_ids: list[UUID] = []
        errors: list[WorkflowRunExecutionNodeFailure] = []
        for _ in range(max_node_runs):
            next_node_run = await self._next_queued_node_run_for_workflow(
                workflow_run_id
            )
            if next_node_run is None:
                break

            processed_node_run_ids.append(next_node_run.id)
            try:
                await self.executor.execute_node_run(next_node_run.id)
            except NodeRunExecutionError as exc:
                errors.append(
                    WorkflowRunExecutionNodeFailure(
                        node_run_id=next_node_run.id,
                        error=str(exc),
                    )
                )
                break

        workflow_run = await self._workflow_run(workflow_run_id)
        return WorkflowRunExecutionResult(
            workflow_run=workflow_run,
            processed_node_run_ids=processed_node_run_ids,
            errors=errors,
        )

    async def _next_queued_node_run_for_workflow(
        self,
        workflow_run_id: UUID,
    ) -> NodeRun | None:
        async with self.uow_factory() as uow:
            workflow_run = await uow.workflow_runs.get(workflow_run_id)
            if workflow_run is None:
                raise NotFoundError("WorkflowRun", str(workflow_run_id))
            if workflow_run.status in {
                WorkflowRunStatus.SUCCEEDED,
                WorkflowRunStatus.FAILED_PERMANENT,
                WorkflowRunStatus.CANCELLED,
            }:
                return None
            node_runs = await uow.node_runs.list_for_workflow_run(workflow_run_id)

        queued_node_runs = [
            node_run
            for node_run in node_runs
            if node_run.status == NodeRunStatus.QUEUED
        ]
        return queued_node_runs[0] if queued_node_runs else None

    async def _workflow_run(self, workflow_run_id: UUID) -> WorkflowRun:
        async with self.uow_factory() as uow:
            workflow_run = await uow.workflow_runs.get(workflow_run_id)
            if workflow_run is None:
                raise NotFoundError("WorkflowRun", str(workflow_run_id))
            return workflow_run
