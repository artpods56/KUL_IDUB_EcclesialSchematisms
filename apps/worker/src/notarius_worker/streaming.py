import os
from collections.abc import Mapping

from faststream import FastStream
from faststream.nats import NatsBroker

from notarius_messaging.contracts import NodeRunExecuteRequested
from notarius_messaging.subjects import NODE_RUN_EXECUTE_REQUESTED_SUBJECT
from notarius_shared.message_contracts import JOB_RUN_SUBJECT, JobRunRequested
from notarius_storage import ArtifactPayloadStoragePort
from notarius_core.application.operators import builtin_node_specs
from notarius_worker.dependencies import create_uow_factory, get_artifact_payload_storage
from notarius_worker.node_execution import NodeRunExecutor, NodeRunHandler
from notarius_worker.operators import builtin_node_handlers
from notarius_worker.outbox import NatsBrokerOutboxPublisher, OutboxDrainLoop
from notarius_worker.runner import WorkerRunner
from notarius_worker.streams import (
    NODE_RUN_EXECUTE_CONSUMER_CONFIG,
    create_worker_streams,
    ensure_worker_stream_definitions,
)


def create_app(
    nats_url: str | None = None,
    node_handlers: Mapping[tuple[str, str], NodeRunHandler] | None = None,
    outbox_interval_seconds: float = 1.0,
    payload_storage: ArtifactPayloadStoragePort | None = None,
) -> FastStream:
    resolved_nats_url = nats_url or os.getenv("NATS_URL", "nats://localhost:4222")
    broker = NatsBroker(resolved_nats_url)
    uow_factory = create_uow_factory()
    runner = WorkerRunner(uow_factory)
    streams = create_worker_streams()
    handlers = builtin_node_handlers(payload_storage or get_artifact_payload_storage())
    if node_handlers is not None:
        handlers.update(node_handlers)
    node_run_executor = NodeRunExecutor(uow_factory, handlers, builtin_node_specs())
    outbox_drain_loop = OutboxDrainLoop(
        uow_factory,
        NatsBrokerOutboxPublisher(broker),
        interval_seconds=outbox_interval_seconds,
    )

    async def ensure_streams() -> None:
        await ensure_worker_stream_definitions(resolved_nats_url)

    @broker.subscriber(JOB_RUN_SUBJECT)
    async def run_job(message: JobRunRequested) -> None:
        await runner.run_job(message.job_id)

    @broker.subscriber(
        NODE_RUN_EXECUTE_REQUESTED_SUBJECT,
        stream=streams.tasks,
        config=NODE_RUN_EXECUTE_CONSUMER_CONFIG,
    )
    async def execute_node_run(message: NodeRunExecuteRequested) -> None:
        await node_run_executor.execute_node_run(message.node_run_id)

    return FastStream(
        broker,
        after_startup=[ensure_streams, outbox_drain_loop.start],
        on_shutdown=[outbox_drain_loop.stop],
    )


app = create_app()
