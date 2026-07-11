import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import UUID

from faststream.nats import NatsBroker
from pydantic import ValidationError

from notarius_core.domain.models import NodeRunStatus, OutboxMessage
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_messaging.contracts import ErrorContext, NodeRunExecuteRequested
from notarius_messaging.outbox import (
    OutboxDispatcher,
    OutboxPublishPort,
    dlq_node_run_execute_outbox_message,
)
from notarius_messaging.subjects import (
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    TASK_SUBJECTS,
)
from notarius_worker.node_execution import NodeRunExecutor
from notarius_worker.streams import WorkerStreamNames

logger = logging.getLogger(__name__)


class NatsBrokerOutboxPublisher:
    def __init__(
        self,
        broker: NatsBroker,
        stream_names: WorkerStreamNames | None = None,
    ):
        self.broker = broker
        self.stream_names = stream_names or WorkerStreamNames()

    async def publish(self, subject: str, payload: dict[str, object]) -> None:
        stream = self._stream_for_subject(subject)
        if stream is None:
            await self.broker.publish(payload, subject)
            return
        await self.broker.publish(payload, subject, stream=stream)

    def _stream_for_subject(self, subject: str) -> str | None:
        if subject in TASK_SUBJECTS:
            return self.stream_names.tasks
        if subject.startswith("events."):
            return self.stream_names.events
        if subject.startswith("live."):
            return self.stream_names.live_deltas
        if subject.startswith("dlq."):
            return self.stream_names.dlq
        return None


class OutboxDrainLoop:
    def __init__(
        self,
        uow_factory: Callable[[], StudioUnitOfWorkPort],
        publisher: OutboxPublishPort,
        interval_seconds: float = 1.0,
        max_publish_attempts: int = 5,
    ):
        self.uow_factory = uow_factory
        self.dispatcher = OutboxDispatcher(
            publisher,
            max_attempts=max_publish_attempts,
        )
        self.interval_seconds = interval_seconds
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def dispatch_once(self) -> int:
        async with self.uow_factory() as uow:
            return await self.dispatcher.dispatch_pending(uow)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        await self._task
        self._task = None

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.dispatch_once()
            except Exception:
                logger.exception("outbox_dispatch_failed")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.interval_seconds,
                )
            except TimeoutError:
                continue


@dataclass(slots=True)
class LocalNodeRunOutboxDrainError:
    outbox_message_id: UUID
    error: str


@dataclass(slots=True)
class LocalNodeRunOutboxDrainResult:
    processed_message_ids: list[UUID] = field(default_factory=list)
    processed_node_run_ids: list[UUID] = field(default_factory=list)
    errors: list[LocalNodeRunOutboxDrainError] = field(default_factory=list)


class LocalNodeRunOutboxDrainer:
    def __init__(
        self,
        uow_factory: Callable[[], StudioUnitOfWorkPort],
        executor: NodeRunExecutor,
    ):
        self.uow_factory = uow_factory
        self.executor = executor

    async def drain_once(self, max_messages: int = 100) -> LocalNodeRunOutboxDrainResult:
        if max_messages < 1:
            raise ValueError("max_messages must be greater than 0")

        async with self.uow_factory() as uow:
            pending_messages = await uow.outbox_messages.list_pending()

        messages = [
            message
            for message in pending_messages
            if message.subject == NODE_RUN_EXECUTE_REQUESTED_SUBJECT
        ][:max_messages]
        processed_message_ids: list[UUID] = []
        processed_node_run_ids: list[UUID] = []
        errors: list[LocalNodeRunOutboxDrainError] = []
        for message in messages:
            request = self._node_run_request(message)
            if isinstance(request, LocalNodeRunOutboxDrainError):
                await self._dead_letter_invalid_message(message.id, request.error)
                errors.append(request)
                continue

            try:
                await self.executor.execute_node_run(request.node_run_id)
            except Exception as exc:
                error = LocalNodeRunOutboxDrainError(
                    outbox_message_id=message.id,
                    error=str(exc),
                )
                await self._handle_execution_failure(message.id, request, error.error)
                errors.append(error)
                continue

            completion_error = await self._handle_execution_completion(
                message.id,
                request,
            )
            if completion_error is not None:
                errors.append(completion_error)
                continue

            await self._mark_message_published(message.id)
            processed_message_ids.append(message.id)
            processed_node_run_ids.append(request.node_run_id)

        return LocalNodeRunOutboxDrainResult(
            processed_message_ids=processed_message_ids,
            processed_node_run_ids=processed_node_run_ids,
            errors=errors,
        )

    async def drain_until_idle(
        self,
        max_messages: int = 100,
    ) -> LocalNodeRunOutboxDrainResult:
        if max_messages < 1:
            raise ValueError("max_messages must be greater than 0")

        processed_message_ids: list[UUID] = []
        processed_node_run_ids: list[UUID] = []
        errors: list[LocalNodeRunOutboxDrainError] = []
        while len(processed_message_ids) < max_messages:
            remaining = max_messages - len(processed_message_ids)
            batch = await self.drain_once(max_messages=remaining)
            processed_message_ids.extend(batch.processed_message_ids)
            processed_node_run_ids.extend(batch.processed_node_run_ids)
            errors.extend(batch.errors)
            if batch.errors or not batch.processed_message_ids:
                break

        return LocalNodeRunOutboxDrainResult(
            processed_message_ids=processed_message_ids,
            processed_node_run_ids=processed_node_run_ids,
            errors=errors,
        )

    def _node_run_request(
        self,
        message: OutboxMessage,
    ) -> NodeRunExecuteRequested | LocalNodeRunOutboxDrainError:
        try:
            return NodeRunExecuteRequested.model_validate(message.payload)
        except ValidationError as exc:
            return LocalNodeRunOutboxDrainError(
                outbox_message_id=message.id,
                error=str(exc),
            )

    async def _mark_message_published(self, message_id: UUID) -> None:
        async with self.uow_factory() as uow:
            message = await uow.outbox_messages.get(message_id)
            if message is None:
                return
            message.mark_published()
            await uow.outbox_messages.update(message)
            await uow.commit()

    async def _mark_message_failed(self, message_id: UUID, error: str) -> None:
        async with self.uow_factory() as uow:
            message = await uow.outbox_messages.get(message_id)
            if message is None:
                return
            message.mark_failed(error)
            await uow.outbox_messages.update(message)
            await uow.commit()

    async def _handle_execution_failure(
        self,
        message_id: UUID,
        request: NodeRunExecuteRequested,
        error: str,
    ) -> None:
        async with self.uow_factory() as uow:
            node_run = await uow.node_runs.get(request.node_run_id)

        if node_run is not None and node_run.status == NodeRunStatus.FAILED_RETRYABLE:
            await self._mark_message_failed(message_id, error)
            return
        if node_run is not None and node_run.status in {
            NodeRunStatus.CANCELLED,
            NodeRunStatus.SUCCEEDED,
        }:
            await self._mark_message_published(message_id)
            return

        error_code = "node_run_execution_failed"
        if node_run is None:
            error_code = "node_run_not_found"
        elif node_run.status == NodeRunStatus.FAILED_PERMANENT:
            error_code = "node_run_failed_permanent"

        await self._dead_letter_message(
            message_id,
            error,
            error_code=error_code,
            request=request,
        )

    async def _handle_execution_completion(
        self,
        message_id: UUID,
        request: NodeRunExecuteRequested,
    ) -> LocalNodeRunOutboxDrainError | None:
        async with self.uow_factory() as uow:
            node_run = await uow.node_runs.get(request.node_run_id)

        if node_run is None:
            error = f"NodeRun not found after execution: {request.node_run_id}"
            await self._dead_letter_message(
                message_id,
                error,
                error_code="node_run_not_found",
                request=request,
            )
            return LocalNodeRunOutboxDrainError(
                outbox_message_id=message_id,
                error=error,
            )
        if node_run.status == NodeRunStatus.FAILED_PERMANENT:
            error = node_run.error or "node run failed permanently"
            await self._dead_letter_message(
                message_id,
                error,
                error_code="node_run_failed_permanent",
                request=request,
            )
            return LocalNodeRunOutboxDrainError(
                outbox_message_id=message_id,
                error=error,
            )
        if node_run.status == NodeRunStatus.FAILED_RETRYABLE:
            error = node_run.error or "node run failed retryably"
            await self._mark_message_failed(message_id, error)
            return LocalNodeRunOutboxDrainError(
                outbox_message_id=message_id,
                error=error,
            )
        return None

    async def _dead_letter_invalid_message(self, message_id: UUID, error: str) -> None:
        await self._dead_letter_message(
            message_id,
            error,
            error_code="invalid_node_run_execute_requested",
            request=None,
        )

    async def _dead_letter_message(
        self,
        message_id: UUID,
        error: str,
        *,
        error_code: str,
        request: NodeRunExecuteRequested | None,
    ) -> None:
        async with self.uow_factory() as uow:
            message = await uow.outbox_messages.get(message_id)
            if message is None:
                return
            details = {
                "outbox_message_id": str(message.id),
                "message_type": message.message_type,
                "subject": message.subject,
            }
            if request is not None:
                details["workflow_run_id"] = str(request.workflow_run_id)
                details["node_run_id"] = str(request.node_run_id)
            workflow_run_id = None
            node_run_id = None
            if request is not None:
                workflow_run_id = request.workflow_run_id
                node_run_id = request.node_run_id
            await uow.outbox_messages.add(
                dlq_node_run_execute_outbox_message(
                    original_subject=message.subject,
                    original_message_id=str(message.id),
                    consumer_name="local-node-run-outbox-drainer",
                    failure=ErrorContext(
                        operation="drain_node_run_outbox_message",
                        error_code=error_code,
                        error_message=error,
                        retryable=False,
                        details=details,
                    ),
                    attempt_count=message.attempts + 1,
                    workflow_run_id=workflow_run_id,
                    node_run_id=node_run_id,
                )
            )
            message.mark_published()
            await uow.outbox_messages.update(message)
            await uow.commit()
