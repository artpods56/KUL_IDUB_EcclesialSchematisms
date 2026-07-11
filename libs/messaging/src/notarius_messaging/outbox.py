from typing import Protocol
from uuid import UUID

from notarius_core.domain.models import Artifact, NodeRun, OutboxMessage, WorkflowRun
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_messaging.contracts import (
    ArtifactEvent,
    ArtifactEventType,
    DlqMessage,
    ErrorContext,
    NodeRunEvent,
    NodeRunExecuteRequested,
    RunEventType,
    WorkflowRunEvent,
)
from notarius_messaging.subjects import (
    DLQ_NODE_RUN_EXECUTE_SUBJECT,
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    artifact_event_subject,
    node_run_event_subject,
    workflow_run_event_subject,
)


class OutboxPublishPort(Protocol):
    async def publish(self, subject: str, payload: dict[str, object]) -> None: ...


def node_run_execute_requested_outbox_message(
    workflow_run: WorkflowRun,
    node_run: NodeRun,
) -> OutboxMessage:
    message = NodeRunExecuteRequested(
        correlation_id=workflow_run.id,
        workflow_run_id=workflow_run.id,
        node_run_id=node_run.id,
    )
    return OutboxMessage(
        subject=NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
        message_type=NodeRunExecuteRequested.__name__,
        payload=message.model_dump(mode="json"),
    )


def workflow_run_event_outbox_message(
    workflow_run: WorkflowRun,
    event_type: RunEventType,
    error: ErrorContext | None = None,
) -> OutboxMessage:
    message = WorkflowRunEvent(
        correlation_id=workflow_run.id,
        workflow_run_id=workflow_run.id,
        event_type=event_type,
        error=error,
    )
    return OutboxMessage(
        subject=workflow_run_event_subject(event_type.value),
        message_type=WorkflowRunEvent.__name__,
        payload=message.model_dump(mode="json"),
    )


def node_run_event_outbox_message(
    node_run: NodeRun,
    event_type: RunEventType,
    error: ErrorContext | None = None,
) -> OutboxMessage:
    message = NodeRunEvent(
        correlation_id=node_run.workflow_run_id,
        workflow_run_id=node_run.workflow_run_id,
        node_run_id=node_run.id,
        event_type=event_type,
        error=error,
    )
    return OutboxMessage(
        subject=node_run_event_subject(event_type.value),
        message_type=NodeRunEvent.__name__,
        payload=message.model_dump(mode="json"),
    )


def artifact_created_event_outbox_message(artifact: Artifact) -> OutboxMessage:
    message = ArtifactEvent(
        correlation_id=artifact.workflow_run_id,
        artifact_id=artifact.id,
        event_type=ArtifactEventType.CREATED,
        artifact_type=artifact.artifact_type,
        workflow_run_id=artifact.workflow_run_id,
        node_run_id=artifact.producer_node_run_id,
    )
    return OutboxMessage(
        subject=artifact_event_subject(ArtifactEventType.CREATED.value),
        message_type=ArtifactEvent.__name__,
        payload=message.model_dump(mode="json"),
    )


def dlq_node_run_execute_outbox_message(
    *,
    original_subject: str,
    original_message_id: str,
    consumer_name: str,
    failure: ErrorContext,
    attempt_count: int,
    workflow_run_id: UUID | None = None,
    node_run_id: UUID | None = None,
    artifact_id: UUID | None = None,
) -> OutboxMessage:
    message = DlqMessage(
        original_subject=original_subject,
        original_message_id=original_message_id,
        consumer_name=consumer_name,
        failure=failure,
        attempt_count=attempt_count,
        workflow_run_id=workflow_run_id,
        node_run_id=node_run_id,
        artifact_id=artifact_id,
    )
    return OutboxMessage(
        subject=DLQ_NODE_RUN_EXECUTE_SUBJECT,
        message_type=DlqMessage.__name__,
        payload=message.model_dump(mode="json"),
    )


class OutboxDispatcher:
    def __init__(self, publisher: OutboxPublishPort, max_attempts: int = 5):
        if max_attempts < 1:
            raise ValueError("max_attempts must be greater than 0")
        self.publisher = publisher
        self.max_attempts = max_attempts

    async def dispatch_pending(self, uow: StudioUnitOfWorkPort) -> int:
        pending_messages = await uow.outbox_messages.list_pending()
        published_count = 0
        for message in pending_messages:
            try:
                await self.publisher.publish(message.subject, message.payload)
            except Exception as exc:
                message.mark_failed(str(exc))
                if message.attempts >= self.max_attempts:
                    message.mark_permanently_failed(str(exc))
                await uow.outbox_messages.update(message)
                continue

            message.mark_published()
            await uow.outbox_messages.update(message)
            published_count += 1

        await uow.commit()
        return published_count
