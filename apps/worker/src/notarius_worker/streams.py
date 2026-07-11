import logging
from dataclasses import dataclass, replace

import nats
from faststream.nats import JStream
from nats.js.api import ConsumerConfig, RetentionPolicy, StorageType, StreamConfig
from nats.js.errors import NotFoundError
from notarius_messaging.subjects import (
    DLQ_SUBJECTS,
    EVENT_SUBJECTS,
    LIVE_DELTA_SUBJECTS,
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
    TASK_SUBJECTS,
    WORKFLOW_COMPILE_REQUESTED_SUBJECT,
    WORKFLOW_RUN_REQUESTED_SUBJECT,
)

logger = logging.getLogger(__name__)

TASK_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
EVENT_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
LIVE_DELTA_MAX_AGE_SECONDS = 5 * 60
DUPLICATE_WINDOW_SECONDS = 120

CONSUMER_WORKFLOW_COMPILE = "consumer-workflow-compile"
CONSUMER_WORKFLOW_RUN = "consumer-workflow-run"
CONSUMER_NODE_RUN_EXECUTE = "consumer-node-run-execute"


@dataclass(frozen=True, slots=True)
class WorkerStreamNames:
    tasks: str = "TASKS"
    events: str = "EVENTS"
    live_deltas: str = "LIVE_DELTAS"
    dlq: str = "DLQ"


@dataclass(frozen=True, slots=True)
class WorkerStreams:
    tasks: JStream
    events: JStream
    live_deltas: JStream
    dlq: JStream


def create_worker_streams(names: WorkerStreamNames | None = None) -> WorkerStreams:
    stream_names = names or WorkerStreamNames()
    return WorkerStreams(
        tasks=JStream(
            name=stream_names.tasks,
            subjects=TASK_SUBJECTS,
            retention=RetentionPolicy.WORK_QUEUE,
            storage=StorageType.FILE,
            max_age=TASK_MAX_AGE_SECONDS,
            duplicate_window=DUPLICATE_WINDOW_SECONDS,
        ),
        events=JStream(
            name=stream_names.events,
            subjects=EVENT_SUBJECTS,
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.FILE,
            max_age=EVENT_MAX_AGE_SECONDS,
            duplicate_window=DUPLICATE_WINDOW_SECONDS,
        ),
        live_deltas=JStream(
            name=stream_names.live_deltas,
            subjects=LIVE_DELTA_SUBJECTS,
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.MEMORY,
            max_age=LIVE_DELTA_MAX_AGE_SECONDS,
            duplicate_window=DUPLICATE_WINDOW_SECONDS,
        ),
        dlq=JStream(
            name=stream_names.dlq,
            subjects=DLQ_SUBJECTS,
            retention=RetentionPolicy.LIMITS,
            storage=StorageType.FILE,
            max_age=EVENT_MAX_AGE_SECONDS,
            duplicate_window=DUPLICATE_WINDOW_SECONDS,
        ),
    )


async def ensure_worker_stream_definitions(
    nats_url: str,
    names: WorkerStreamNames | None = None,
) -> None:
    nats_client = await nats.connect(nats_url)
    try:
        jetstream = nats_client.jetstream()
        streams = create_worker_streams(names)
        for stream in (streams.tasks, streams.events, streams.live_deltas, streams.dlq):
            desired = stream_config_for(stream)
            try:
                existing = await jetstream.stream_info(stream.name)
            except NotFoundError:
                await jetstream.add_stream(desired)
                logger.info(
                    "nats_stream_created",
                    extra={"stream": stream.name, "subjects": desired.subjects},
                )
                continue

            updated = worker_stream_update_config(existing.config, desired)
            if not worker_stream_config_matches(existing.config, updated):
                await jetstream.update_stream(updated)
                logger.info(
                    "nats_stream_updated",
                    extra={
                        "stream": stream.name,
                        "previous_subjects": existing.config.subjects,
                        "subjects": updated.subjects,
                    },
                )
    finally:
        await nats_client.close()


def stream_config_for(stream: JStream) -> StreamConfig:
    return replace(stream.config, subjects=list(stream.subjects))


def worker_stream_update_config(
    existing: StreamConfig, desired: StreamConfig
) -> StreamConfig:
    return replace(
        existing,
        subjects=list(desired.subjects or []),
        retention=desired.retention,
        storage=desired.storage,
        max_age=desired.max_age,
        duplicate_window=desired.duplicate_window,
    )


def worker_stream_config_matches(current: StreamConfig, desired: StreamConfig) -> bool:
    return (
        current.subjects == desired.subjects
        and current.retention == desired.retention
        and current.storage == desired.storage
        and current.max_age == desired.max_age
        and current.duplicate_window == desired.duplicate_window
    )


def task_consumer_config(durable_name: str, filter_subject: str) -> ConsumerConfig:
    return ConsumerConfig(
        durable_name=durable_name,
        name=durable_name,
        filter_subject=filter_subject,
        max_deliver=5,
        ack_wait=30,
        backoff=[30, 120, 600, 1800, 7200],
    )


def node_run_consumer_config(durable_name: str, filter_subject: str) -> ConsumerConfig:
    return ConsumerConfig(
        durable_name=durable_name,
        name=durable_name,
        filter_subject=filter_subject,
        max_deliver=5,
        ack_wait=600,
        backoff=[600, 1800, 3600, 7200, 7200],
    )


WORKFLOW_COMPILE_CONSUMER_CONFIG = task_consumer_config(
    CONSUMER_WORKFLOW_COMPILE,
    WORKFLOW_COMPILE_REQUESTED_SUBJECT,
)
WORKFLOW_RUN_CONSUMER_CONFIG = task_consumer_config(
    CONSUMER_WORKFLOW_RUN,
    WORKFLOW_RUN_REQUESTED_SUBJECT,
)
NODE_RUN_EXECUTE_CONSUMER_CONFIG = node_run_consumer_config(
    CONSUMER_NODE_RUN_EXECUTE,
    NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
)

__all__ = [
    "CONSUMER_NODE_RUN_EXECUTE",
    "CONSUMER_WORKFLOW_COMPILE",
    "CONSUMER_WORKFLOW_RUN",
    "DUPLICATE_WINDOW_SECONDS",
    "EVENT_MAX_AGE_SECONDS",
    "LIVE_DELTA_MAX_AGE_SECONDS",
    "NODE_RUN_EXECUTE_CONSUMER_CONFIG",
    "TASK_MAX_AGE_SECONDS",
    "WORKFLOW_COMPILE_CONSUMER_CONFIG",
    "WORKFLOW_RUN_CONSUMER_CONFIG",
    "WorkerStreamNames",
    "WorkerStreams",
    "create_worker_streams",
    "ensure_worker_stream_definitions",
    "node_run_consumer_config",
    "stream_config_for",
    "task_consumer_config",
    "worker_stream_config_matches",
    "worker_stream_update_config",
]
