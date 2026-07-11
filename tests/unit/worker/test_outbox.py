from uuid import uuid4

import pytest

from notarius_core.application.operators import (
    DEBUG_EMIT_TEXT_OPERATOR_ID,
    DEBUG_EMIT_TEXT_OPERATOR_VERSION,
)
from notarius_core.domain.models import (
    NodeRun,
    NodeRunStatus,
    OutboxMessage,
    OutboxMessageStatus,
    WorkflowRun,
    WorkflowRunStatus,
)
from notarius_messaging.outbox import node_run_execute_requested_outbox_message
from notarius_messaging.subjects import (
    DLQ_NODE_RUN_EXECUTE_SUBJECT,
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    WORKFLOW_RUN_RUNNING_EVENT_SUBJECT,
)
from notarius_persistence.adapters.in_memory import (
    InMemoryDataStore,
    InMemoryUnitOfWork,
)
from notarius_worker.node_execution import (
    NodeExecutionRequest,
    NodeExecutionResult,
    NodeRunExecutionError,
    NodeRunExecutor,
)
from notarius_worker.operators import builtin_node_handlers
from notarius_worker.outbox import (
    LocalNodeRunOutboxDrainer,
    NatsBrokerOutboxPublisher,
    OutboxDrainLoop,
)


class RecordingPublisher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, object]]] = []

    async def publish(self, subject: str, payload: dict[str, object]) -> None:
        self.messages.append((subject, payload))


class FailingPublisher:
    async def publish(self, subject: str, payload: dict[str, object]) -> None:
        raise RuntimeError("nats unavailable")


class FakeBroker:
    def __init__(self) -> None:
        self.published: list[tuple[dict[str, object], str, str | None]] = []

    async def publish(
        self,
        payload: dict[str, object],
        subject: str,
        stream: str | None = None,
    ) -> None:
        self.published.append((payload, subject, stream))


class PermanentFailureHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raise NodeRunExecutionError("bad node config", retryable=False)


class RetryableFailureHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raise NodeRunExecutionError("provider busy", retryable=True)


class UnexpectedHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raise AssertionError("handler should not run")


@pytest.mark.asyncio
async def test_outbox_drain_loop_publishes_pending_messages() -> None:
    store = InMemoryDataStore()
    outbox_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()
    publisher = RecordingPublisher()
    drain_loop = OutboxDrainLoop(
        lambda: InMemoryUnitOfWork(store),
        publisher,
        interval_seconds=0.01,
    )

    published_count = await drain_loop.dispatch_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)
        pending_messages = await uow.outbox_messages.list_pending()

    assert published_count == 1
    assert publisher.messages == [(outbox_message.subject, outbox_message.payload)]
    assert pending_messages == []
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.PUBLISHED
    assert stored_message.published_at is not None


@pytest.mark.asyncio
async def test_outbox_drain_loop_marks_publish_failures() -> None:
    store = InMemoryDataStore()
    outbox_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()
    drain_loop = OutboxDrainLoop(
        lambda: InMemoryUnitOfWork(store),
        FailingPublisher(),
        interval_seconds=0.01,
    )

    published_count = await drain_loop.dispatch_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)

    assert published_count == 0
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.PENDING
    assert stored_message.attempts == 1
    assert stored_message.error == "nats unavailable"


@pytest.mark.asyncio
async def test_outbox_drain_loop_marks_publish_failures_terminal_after_attempt_cap() -> None:
    store = InMemoryDataStore()
    outbox_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()
    drain_loop = OutboxDrainLoop(
        lambda: InMemoryUnitOfWork(store),
        FailingPublisher(),
        interval_seconds=0.01,
        max_publish_attempts=2,
    )

    await drain_loop.dispatch_once()
    published_count = await drain_loop.dispatch_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)
        pending_messages = await uow.outbox_messages.list_pending()

    assert published_count == 0
    assert pending_messages == []
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.FAILED
    assert stored_message.attempts == 2
    assert stored_message.error == "nats unavailable"


@pytest.mark.asyncio
async def test_nats_broker_outbox_publisher_delegates_to_broker() -> None:
    broker = FakeBroker()
    publisher = NatsBrokerOutboxPublisher(broker)
    payload = {"node_run_id": str(uuid4())}

    await publisher.publish(NODE_RUN_EXECUTE_REQUESTED_SUBJECT, payload)

    assert broker.published == [(payload, NODE_RUN_EXECUTE_REQUESTED_SUBJECT, "TASKS")]


@pytest.mark.asyncio
async def test_nats_broker_outbox_publisher_maps_event_and_dlq_streams() -> None:
    broker = FakeBroker()
    publisher = NatsBrokerOutboxPublisher(broker)
    event_payload = {"workflow_run_id": str(uuid4())}
    dlq_payload = {"original_message_id": str(uuid4())}

    await publisher.publish(WORKFLOW_RUN_RUNNING_EVENT_SUBJECT, event_payload)
    await publisher.publish(DLQ_NODE_RUN_EXECUTE_SUBJECT, dlq_payload)

    assert broker.published == [
        (event_payload, WORKFLOW_RUN_RUNNING_EVENT_SUBJECT, "EVENTS"),
        (dlq_payload, DLQ_NODE_RUN_EXECUTE_SUBJECT, "DLQ"),
    ]


@pytest.mark.asyncio
async def test_local_node_run_outbox_drainer_executes_pending_node_run() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="emit",
        operator_id=DEBUG_EMIT_TEXT_OPERATOR_ID,
        operator_version=DEBUG_EMIT_TEXT_OPERATOR_VERSION,
        metadata={"workflow_node_config": {"text": "hello from local drain"}},
    )
    outbox_message = node_run_execute_requested_outbox_message(workflow_run, node_run)
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()

    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        builtin_node_handlers(),
    )
    result = await LocalNodeRunOutboxDrainer(
        lambda: InMemoryUnitOfWork(store),
        executor,
    ).drain_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)
        stored_node_run = await uow.node_runs.get(node_run.id)
        stored_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        artifacts = await uow.artifacts.list_for_node_run(node_run.id)

    assert result.processed_message_ids == [outbox_message.id]
    assert result.processed_node_run_ids == [node_run.id]
    assert result.errors == []
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.PUBLISHED
    assert stored_node_run is not None
    assert stored_node_run.status == NodeRunStatus.SUCCEEDED
    assert stored_workflow_run is not None
    assert stored_workflow_run.status == WorkflowRunStatus.SUCCEEDED
    assert artifacts[0].metadata == {"text": "hello from local drain"}


@pytest.mark.asyncio
async def test_local_node_run_outbox_drainer_dead_letters_invalid_payload() -> None:
    store = InMemoryDataStore()
    outbox_message = OutboxMessage(
        subject="jobs.node_run.execute.requested",
        message_type="NodeRunExecuteRequested",
        payload={"node_run_id": str(uuid4())},
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()

    executor = NodeRunExecutor(lambda: InMemoryUnitOfWork(store), {})
    result = await LocalNodeRunOutboxDrainer(
        lambda: InMemoryUnitOfWork(store),
        executor,
    ).drain_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)
        pending_messages = await uow.outbox_messages.list_pending()

    assert result.processed_message_ids == []
    assert result.processed_node_run_ids == []
    assert len(result.errors) == 1
    assert result.errors[0].outbox_message_id == outbox_message.id
    assert "workflow_run_id" in result.errors[0].error
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.PUBLISHED
    assert [message.subject for message in pending_messages] == [
        DLQ_NODE_RUN_EXECUTE_SUBJECT
    ]
    assert pending_messages[0].message_type == "DlqMessage"
    assert pending_messages[0].payload["original_message_id"] == str(
        outbox_message.id
    )
    assert pending_messages[0].payload["failure"]["error_code"] == (
        "invalid_node_run_execute_requested"
    )


@pytest.mark.asyncio
async def test_local_node_run_outbox_drainer_dead_letters_permanent_execution_failure() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="permanent",
        operator_id="test.permanent_failure",
        operator_version="1.0.0",
    )
    outbox_message = node_run_execute_requested_outbox_message(workflow_run, node_run)
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()

    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("test.permanent_failure", "1.0.0"): PermanentFailureHandler()},
    )
    result = await LocalNodeRunOutboxDrainer(
        lambda: InMemoryUnitOfWork(store),
        executor,
    ).drain_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)
        stored_node_run = await uow.node_runs.get(node_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    dlq_messages = [
        message
        for message in pending_messages
        if message.subject == DLQ_NODE_RUN_EXECUTE_SUBJECT
    ]
    assert result.processed_message_ids == []
    assert result.processed_node_run_ids == []
    assert len(result.errors) == 1
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.PUBLISHED
    assert stored_node_run is not None
    assert stored_node_run.status == NodeRunStatus.FAILED_PERMANENT
    assert len(dlq_messages) == 1
    assert dlq_messages[0].payload["workflow_run_id"] == str(workflow_run.id)
    assert dlq_messages[0].payload["node_run_id"] == str(node_run.id)
    assert dlq_messages[0].payload["failure"]["error_code"] == (
        "node_run_failed_permanent"
    )
    assert dlq_messages[0].payload["failure"]["retryable"] is False


@pytest.mark.asyncio
async def test_local_node_run_outbox_drainer_keeps_retryable_execution_failure_pending() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="retryable",
        operator_id="test.retryable_failure",
        operator_version="1.0.0",
    )
    outbox_message = node_run_execute_requested_outbox_message(workflow_run, node_run)
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()

    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("test.retryable_failure", "1.0.0"): RetryableFailureHandler()},
    )
    result = await LocalNodeRunOutboxDrainer(
        lambda: InMemoryUnitOfWork(store),
        executor,
    ).drain_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)
        stored_node_run = await uow.node_runs.get(node_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    dlq_messages = [
        message
        for message in pending_messages
        if message.subject == DLQ_NODE_RUN_EXECUTE_SUBJECT
    ]
    assert result.processed_message_ids == []
    assert result.processed_node_run_ids == []
    assert len(result.errors) == 1
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.PENDING
    assert stored_message.attempts == 1
    assert stored_message.error == "provider busy"
    assert stored_node_run is not None
    assert stored_node_run.status == NodeRunStatus.FAILED_RETRYABLE
    assert dlq_messages == []


@pytest.mark.asyncio
async def test_local_node_run_outbox_drainer_dead_letters_exhausted_retryable_node_run() -> None:
    store = InMemoryDataStore()
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="retryable",
        operator_id="test.retryable_failure",
        operator_version="1.0.0",
        max_attempts=2,
    )
    for _ in range(node_run.max_attempts):
        node_run.mark_running()
        node_run.mark_failed("provider busy", retryable=True)
    workflow_run.mark_running()
    workflow_run.mark_failed("provider busy", retryable=True)
    outbox_message = node_run_execute_requested_outbox_message(workflow_run, node_run)
    async with InMemoryUnitOfWork(store) as uow:
        await uow.workflow_runs.add(workflow_run)
        await uow.node_runs.add(node_run)
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()

    executor = NodeRunExecutor(
        lambda: InMemoryUnitOfWork(store),
        {("test.retryable_failure", "1.0.0"): UnexpectedHandler()},
    )
    result = await LocalNodeRunOutboxDrainer(
        lambda: InMemoryUnitOfWork(store),
        executor,
    ).drain_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)
        stored_node_run = await uow.node_runs.get(node_run.id)
        stored_workflow_run = await uow.workflow_runs.get(workflow_run.id)
        pending_messages = await uow.outbox_messages.list_pending()

    dlq_messages = [
        message
        for message in pending_messages
        if message.subject == DLQ_NODE_RUN_EXECUTE_SUBJECT
    ]
    assert result.processed_message_ids == []
    assert result.processed_node_run_ids == []
    assert len(result.errors) == 1
    assert result.errors[0].outbox_message_id == outbox_message.id
    assert "Retry attempts exhausted" in result.errors[0].error
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.PUBLISHED
    assert stored_node_run is not None
    assert stored_node_run.status == NodeRunStatus.FAILED_PERMANENT
    assert stored_workflow_run is not None
    assert stored_workflow_run.status == WorkflowRunStatus.FAILED_PERMANENT
    assert len(dlq_messages) == 1
    assert dlq_messages[0].payload["failure"]["error_code"] == (
        "node_run_failed_permanent"
    )


@pytest.mark.asyncio
async def test_local_node_run_outbox_drainer_ignores_non_node_run_messages() -> None:
    store = InMemoryDataStore()
    outbox_message = OutboxMessage(
        subject="jobs.workflow.run.requested",
        message_type="WorkflowRunRequested",
        payload={"workflow_run_id": str(uuid4())},
    )
    async with InMemoryUnitOfWork(store) as uow:
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()

    executor = NodeRunExecutor(lambda: InMemoryUnitOfWork(store), {})
    result = await LocalNodeRunOutboxDrainer(
        lambda: InMemoryUnitOfWork(store),
        executor,
    ).drain_once()

    async with InMemoryUnitOfWork(store) as uow:
        stored_message = await uow.outbox_messages.get(outbox_message.id)

    assert result.processed_message_ids == []
    assert result.processed_node_run_ids == []
    assert result.errors == []
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.PENDING
