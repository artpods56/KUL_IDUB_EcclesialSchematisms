"""In-process ownership and observation of asynchronous graph executions."""

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from notarius_core.domain.errors import NotFoundError

from notarius_api.schemas.workbench import RunRequest
from notarius_api.services.execution.control import RunExecutionControl
from notarius_api.services.execution.models import GraphExecutionResult
from notarius_api.services.execution.run_graph import RunGraph


type RunExecutionStatus = Literal[
    "queued",
    "running",
    "cancelling",
    "cancelled",
    "succeeded",
    "failed",
]

_TERMINAL_STATUSES = frozenset({"cancelled", "succeeded", "failed"})


@dataclass(frozen=True, slots=True)
class RunExecutionSnapshot:
    execution_id: UUID
    status: RunExecutionStatus
    active_node_id: str | None
    result: GraphExecutionResult | None
    error: str | None


@dataclass(slots=True)
class _RunExecutionRecord:
    execution_id: UUID
    control: RunExecutionControl
    status: RunExecutionStatus = "queued"
    task: asyncio.Task[None] | None = None
    result: GraphExecutionResult | None = None
    error: str | None = None
    retained_terminal: bool = False

    def snapshot(self) -> RunExecutionSnapshot:
        return RunExecutionSnapshot(
            execution_id=self.execution_id,
            status=self.status,
            active_node_id=self.control.active_node_id,
            result=self.result,
            error=self.error,
        )


class RunExecutionManager:
    """Own background graph tasks and retain a bounded set of terminal results."""

    def __init__(
        self,
        run_graph: RunGraph,
        *,
        terminal_retention: int = 100,
    ) -> None:
        if terminal_retention < 1:
            raise ValueError("Execution terminal retention must be at least one")
        self._run_graph = run_graph
        self._terminal_retention = terminal_retention
        self._executions: dict[UUID, _RunExecutionRecord] = {}
        self._terminal_order: deque[UUID] = deque()
        self._lock = asyncio.Lock()
        self._shutting_down = False

    async def start(self, request: RunRequest) -> RunExecutionSnapshot:
        async with self._lock:
            if self._shutting_down:
                raise RuntimeError("Run execution manager is shutting down")
            execution_id = uuid4()
            record = _RunExecutionRecord(
                execution_id=execution_id,
                control=RunExecutionControl(),
            )
            self._executions[execution_id] = record
            task = asyncio.create_task(
                self._run(execution_id, request.model_copy(deep=True)),
                name=f"notarius-run-{execution_id}",
            )
            record.task = task
            task.add_done_callback(
                lambda completed, owned_id=execution_id: self._task_done(
                    owned_id,
                    completed,
                )
            )
            return record.snapshot()

    async def get(self, execution_id: UUID) -> RunExecutionSnapshot:
        async with self._lock:
            record = self._executions.get(execution_id)
            if record is None:
                raise NotFoundError("Run execution", str(execution_id))
            return record.snapshot()

    async def cancel(self, execution_id: UUID) -> RunExecutionSnapshot:
        async with self._lock:
            record = self._executions.get(execution_id)
            if record is None:
                raise NotFoundError("Run execution", str(execution_id))
            if record.status in _TERMINAL_STATUSES:
                return record.snapshot()
            record.control.request_cancel()
            record.status = "cancelling"
            if record.task is not None:
                record.task.cancel()
            return record.snapshot()

    async def shutdown(self) -> None:
        async with self._lock:
            self._shutting_down = True
            active_records = [
                record
                for record in self._executions.values()
                if record.status not in _TERMINAL_STATUSES
            ]
            tasks = [
                record.task for record in active_records if record.task is not None
            ]
            for record in active_records:
                record.control.request_cancel()
                record.status = "cancelling"
            for task in tasks:
                task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for record in active_records:
            if record.status not in _TERMINAL_STATUSES:
                self._complete(record, status="cancelled")

    async def _run(self, execution_id: UUID, request: RunRequest) -> None:
        record = self._executions[execution_id]
        try:
            record.control.check_cancelled()
            record.status = "running"
            result = await self._run_graph.run(request, control=record.control)
        except asyncio.CancelledError:
            self._complete(record, status="cancelled")
        except Exception as exc:
            if record.control.cancel_requested or _contains_cancellation(exc):
                self._complete(record, status="cancelled")
            else:
                self._complete(
                    record,
                    status="failed",
                    error=_render_exception_chain(exc),
                )
        else:
            if record.control.cancel_requested:
                self._complete(record, status="cancelled")
            elif result.status == "failed":
                self._complete(record, status="failed", result=result)
            else:
                self._complete(record, status="succeeded", result=result)

    def _task_done(self, execution_id: UUID, task: asyncio.Task[None]) -> None:
        record = self._executions.get(execution_id)
        if record is None or record.status in _TERMINAL_STATUSES:
            return
        if record.control.cancel_requested or task.cancelled():
            self._complete(record, status="cancelled")
            return
        exception = task.exception()
        if exception is None:
            self._complete(
                record,
                status="failed",
                error="Run execution ended without a terminal result",
            )
            return
        self._complete(
            record,
            status="failed",
            error=_render_exception_chain(exception),
        )

    def _complete(
        self,
        record: _RunExecutionRecord,
        *,
        status: Literal["cancelled", "succeeded", "failed"],
        result: GraphExecutionResult | None = None,
        error: str | None = None,
    ) -> None:
        if record.status in _TERMINAL_STATUSES:
            return
        active_node_id = record.control.active_node_id
        if active_node_id is not None:
            record.control.finish_outer_node(active_node_id)
        record.status = status
        record.task = None
        record.result = result
        record.error = error
        if record.retained_terminal:
            return
        record.retained_terminal = True
        self._terminal_order.append(record.execution_id)
        while len(self._terminal_order) > self._terminal_retention:
            expired_id = self._terminal_order.popleft()
            self._executions.pop(expired_id, None)


def _contains_cancellation(exception: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen:
        if isinstance(current, asyncio.CancelledError):
            return True
        seen.add(id(current))
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        current = None if current.__suppress_context__ else current.__context__
    return False


def _render_exception_chain(exception: BaseException) -> str:
    rendered: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen and len(rendered) < 12:
        seen.add(id(current))
        rendered.append(f"{type(current).__name__}: {current}")
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        current = None if current.__suppress_context__ else current.__context__
    return " <- caused by ".join(rendered)


__all__ = [
    "RunExecutionManager",
    "RunExecutionSnapshot",
    "RunExecutionStatus",
]
