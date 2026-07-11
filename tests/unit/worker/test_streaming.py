from faststream import FastStream

from notarius_messaging.subjects import NODE_RUN_EXECUTE_REQUESTED_SUBJECT
from notarius_shared.message_contracts import JOB_RUN_SUBJECT
from notarius_worker.streams import (
    CONSUMER_NODE_RUN_EXECUTE,
    NODE_RUN_EXECUTE_CONSUMER_CONFIG,
)
from notarius_worker.streaming import create_app


def test_worker_streaming_app_registers_job_and_node_run_subscribers() -> None:
    app = create_app(nats_url="nats://example:4222")

    subjects = _registered_subjects(app)

    assert JOB_RUN_SUBJECT in subjects
    assert NODE_RUN_EXECUTE_REQUESTED_SUBJECT in subjects


def test_worker_streaming_app_registers_outbox_drain_lifecycle_hooks() -> None:
    app = create_app(nats_url="nats://example:4222", outbox_interval_seconds=0.01)

    assert len(app.__dict__["_after_startup_calling"]) == 2
    assert len(app.__dict__["_on_shutdown_calling"]) == 1


def test_worker_streaming_app_registers_node_run_jetstream_consumer() -> None:
    app = create_app(nats_url="nats://example:4222")

    subscriber = _registered_subscriber(app, NODE_RUN_EXECUTE_REQUESTED_SUBJECT)

    assert subscriber.stream.name == "TASKS"
    assert subscriber.config.durable_name == CONSUMER_NODE_RUN_EXECUTE
    assert subscriber.config.name == CONSUMER_NODE_RUN_EXECUTE
    assert subscriber.config.filter_subject == NODE_RUN_EXECUTE_REQUESTED_SUBJECT
    assert subscriber.config.max_deliver == NODE_RUN_EXECUTE_CONSUMER_CONFIG.max_deliver
    assert subscriber.config.ack_wait == NODE_RUN_EXECUTE_CONSUMER_CONFIG.ack_wait
    assert subscriber.config.backoff == NODE_RUN_EXECUTE_CONSUMER_CONFIG.backoff


def _registered_subjects(app: FastStream) -> set[str]:
    subscribers = app.broker.__dict__["_Registrator__persistent_subscribers"]
    return {subscriber.__dict__["_subject"] for subscriber in subscribers}


def _registered_subscriber(app: FastStream, subject: str):
    subscribers = app.broker.__dict__["_Registrator__persistent_subscribers"]
    for subscriber in subscribers:
        if subscriber.__dict__["_subject"] == subject:
            return subscriber
    raise AssertionError(f"subscriber not registered for {subject}")
