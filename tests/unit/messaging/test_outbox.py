from uuid import uuid4

import pytest

from notarius_core.application.workflows import WorkflowRunLauncher
from notarius_core.domain.models import (
    Artifact,
    ArtifactRef,
    ExecutionMode,
    NodeRun,
    OutboxMessageStatus,
    NodeSpec,
    PortSpec,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRun,
    WorkflowVersion,
)
from notarius_messaging.contracts import ErrorContext, RunEventType
from notarius_messaging.outbox import (
    OutboxDispatcher,
    artifact_created_event_outbox_message,
    dlq_node_run_execute_outbox_message,
    node_run_event_outbox_message,
    node_run_execute_requested_outbox_message,
    workflow_run_event_outbox_message,
)
from notarius_messaging.subjects import (
    ARTIFACT_CREATED_EVENT_SUBJECT,
    DLQ_NODE_RUN_EXECUTE_SUBJECT,
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    NODE_RUN_QUEUED_EVENT_SUBJECT,
    NODE_RUN_RUNNING_EVENT_SUBJECT,
    WORKFLOW_RUN_QUEUED_EVENT_SUBJECT,
)
from notarius_persistence.adapters.in_memory import InMemoryUnitOfWork


class RecordingPublisher:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict[str, object]]] = []

    async def publish(self, subject: str, payload: dict[str, object]) -> None:
        self.messages.append((subject, payload))


class FailingPublisher:
    async def publish(self, subject: str, payload: dict[str, object]) -> None:
        raise RuntimeError("nats unavailable")


@pytest.mark.asyncio
async def test_node_run_outbox_builder_and_dispatcher_publish_pending_messages() -> (
    None
):
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
    )
    outbox_message = node_run_execute_requested_outbox_message(workflow_run, node_run)
    publisher = RecordingPublisher()

    async with InMemoryUnitOfWork() as uow:
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()

        published_count = await OutboxDispatcher(publisher).dispatch_pending(uow)
        pending_messages = await uow.outbox_messages.list_pending()
        stored_message = await uow.outbox_messages.get(outbox_message.id)

    assert published_count == 1
    assert pending_messages == []
    assert stored_message is not None
    assert stored_message.published_at is not None
    assert publisher.messages == [
        (
            NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
            outbox_message.payload,
        )
    ]
    assert outbox_message.payload["workflow_run_id"] == str(workflow_run.id)
    assert outbox_message.payload["node_run_id"] == str(node_run.id)


@pytest.mark.asyncio
async def test_outbox_dispatcher_marks_publish_failures_terminal_after_attempt_cap() -> None:
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
    )
    outbox_message = node_run_execute_requested_outbox_message(workflow_run, node_run)

    async with InMemoryUnitOfWork() as uow:
        await uow.outbox_messages.add(outbox_message)
        await uow.commit()

        dispatcher = OutboxDispatcher(FailingPublisher(), max_attempts=2)
        first_published_count = await dispatcher.dispatch_pending(uow)
        second_published_count = await dispatcher.dispatch_pending(uow)
        pending_messages = await uow.outbox_messages.list_pending()
        stored_message = await uow.outbox_messages.get(outbox_message.id)

    assert first_published_count == 0
    assert second_published_count == 0
    assert pending_messages == []
    assert stored_message is not None
    assert stored_message.status == OutboxMessageStatus.FAILED
    assert stored_message.attempts == 2
    assert stored_message.error == "nats unavailable"


def test_lifecycle_event_outbox_builders_use_typed_contract_subjects() -> None:
    workflow_run = WorkflowRun(workflow_version_id=uuid4())
    node_run = NodeRun(
        workflow_run_id=workflow_run.id,
        workflow_node_id="ocr",
        operator_id="ocr.provider",
        operator_version="1.0.0",
    )
    artifact = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=workflow_run.id,
        producer_node_run_id=node_run.id,
        payload_ref="memory://ocr/page-1.json",
    )

    workflow_event = workflow_run_event_outbox_message(
        workflow_run,
        RunEventType.QUEUED,
    )
    node_event = node_run_event_outbox_message(
        node_run,
        RunEventType.RUNNING,
    )
    artifact_event = artifact_created_event_outbox_message(artifact)

    assert workflow_event.subject == WORKFLOW_RUN_QUEUED_EVENT_SUBJECT
    assert workflow_event.message_type == "WorkflowRunEvent"
    assert workflow_event.payload["workflow_run_id"] == str(workflow_run.id)
    assert workflow_event.payload["event_type"] == "queued"
    assert node_event.subject == NODE_RUN_RUNNING_EVENT_SUBJECT
    assert node_event.message_type == "NodeRunEvent"
    assert node_event.payload["workflow_run_id"] == str(workflow_run.id)
    assert node_event.payload["node_run_id"] == str(node_run.id)
    assert node_event.payload["event_type"] == "running"
    assert artifact_event.subject == ARTIFACT_CREATED_EVENT_SUBJECT
    assert artifact_event.message_type == "ArtifactEvent"
    assert artifact_event.payload["artifact_id"] == str(artifact.id)
    assert artifact_event.payload["workflow_run_id"] == str(workflow_run.id)
    assert artifact_event.payload["node_run_id"] == str(node_run.id)
    assert artifact_event.payload["event_type"] == "created"


def test_dlq_outbox_builder_preserves_failure_context() -> None:
    workflow_run_id = uuid4()
    node_run_id = uuid4()
    failure = ErrorContext(
        operation="execute_node_run",
        error_code="node_run_execution_failed",
        error_message="provider unavailable",
        retryable=False,
        details={"operator_id": "ocr.provider"},
    )

    outbox_message = dlq_node_run_execute_outbox_message(
        original_subject=NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
        original_message_id=str(uuid4()),
        consumer_name="notarius-worker",
        failure=failure,
        attempt_count=3,
        workflow_run_id=workflow_run_id,
        node_run_id=node_run_id,
    )

    assert outbox_message.subject == DLQ_NODE_RUN_EXECUTE_SUBJECT
    assert outbox_message.message_type == "DlqMessage"
    assert outbox_message.payload["workflow_run_id"] == str(workflow_run_id)
    assert outbox_message.payload["node_run_id"] == str(node_run_id)
    assert outbox_message.payload["attempt_count"] == 3
    assert outbox_message.payload["failure"] == {
        "operation": "execute_node_run",
        "error_code": "node_run_execution_failed",
        "error_message": "provider unavailable",
        "retryable": False,
        "details": {"operator_id": "ocr.provider"},
    }


@pytest.mark.asyncio
async def test_workflow_launch_persists_outbox_messages_for_root_fanout() -> None:
    page_refs = [
        ArtifactRef(
            artifact_id=uuid4(),
            artifact_type="source.page_image",
            schema_version=1,
        ),
    ]
    page_input = PortSpec(
        name="pages",
        artifact_type="source.page_image",
        schema_version=1,
        sequence=True,
    )
    ocr_spec = NodeSpec(
        id="ocr.run",
        version="1.0.0",
        execution_mode=ExecutionMode.MAP,
        inputs=(page_input,),
        outputs=(
            PortSpec(
                name="ocr_pages",
                artifact_type="ocr.page_result",
                schema_version=1,
                sequence=True,
            ),
        ),
    )
    thumbnail_spec = NodeSpec(
        id="thumbnail.render",
        version="1.0.0",
        execution_mode=ExecutionMode.MAP,
        inputs=(page_input,),
        outputs=(
            PortSpec(
                name="thumbnails",
                artifact_type="source.page_thumbnail",
                schema_version=1,
                sequence=True,
            ),
        ),
    )
    definition = WorkflowDefinition(
        name="Root fan-out",
        declared_inputs=[page_input],
        nodes=[
            WorkflowNode(
                id="ocr",
                operator_id="ocr.run",
                operator_version="1.0.0",
            ),
            WorkflowNode(
                id="thumbnail",
                operator_id="thumbnail.render",
                operator_version="1.0.0",
            ),
        ],
    )
    version = WorkflowVersion(
        workflow_definition_id=definition.id,
        version_number=1,
        definition_snapshot=definition,
    )

    async with InMemoryUnitOfWork() as uow:
        result = await WorkflowRunLauncher(
            {
                (ocr_spec.id, ocr_spec.version): ocr_spec,
                (thumbnail_spec.id, thumbnail_spec.version): thumbnail_spec,
            }
        ).launch(
            uow,
            version,
            page_refs,
            node_run_outbox_message_builder=node_run_execute_requested_outbox_message,
            workflow_run_queued_event_builder=lambda workflow_run: (
                workflow_run_event_outbox_message(
                    workflow_run,
                    RunEventType.QUEUED,
                )
            ),
            node_run_queued_event_builder=lambda node_run: (
                node_run_event_outbox_message(
                    node_run,
                    RunEventType.QUEUED,
                )
            ),
        )

        outbox_messages = await uow.outbox_messages.list_pending()

    execute_messages = [
        message
        for message in outbox_messages
        if message.subject == NODE_RUN_EXECUTE_REQUESTED_SUBJECT
    ]
    node_queued_events = [
        message
        for message in outbox_messages
        if message.subject == NODE_RUN_QUEUED_EVENT_SUBJECT
    ]
    workflow_queued_events = [
        message
        for message in outbox_messages
        if message.subject == WORKFLOW_RUN_QUEUED_EVENT_SUBJECT
    ]

    assert len(outbox_messages) == 5
    assert len(workflow_queued_events) == 1
    assert len(node_queued_events) == 2
    assert len(execute_messages) == 2
    assert workflow_queued_events[0].payload["workflow_run_id"] == str(
        result.workflow_run.id
    )
    assert {message.payload["workflow_run_id"] for message in execute_messages} == {
        str(result.workflow_run.id)
    }
    assert {message.payload["node_run_id"] for message in execute_messages} == {
        str(node_run_id) for node_run_id in result.queued_node_run_ids
    }
    assert {message.payload["node_run_id"] for message in node_queued_events} == {
        str(node_run_id) for node_run_id in result.queued_node_run_ids
    }
