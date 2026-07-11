import os
from typing import Protocol
from uuid import UUID

from faststream.nats import NatsBroker

from notarius_shared.message_contracts import JOB_RUN_SUBJECT, JobRunRequested


class JobPublisher(Protocol):
    async def publish_job_run_requested(self, job_id: UUID) -> None: ...


class NoOpJobPublisher:
    async def publish_job_run_requested(self, job_id: UUID) -> None:
        return None


class NatsJobPublisher:
    def __init__(self, nats_url: str):
        self.nats_url = nats_url

    async def publish_job_run_requested(self, job_id: UUID) -> None:
        message = JobRunRequested(job_id=job_id).model_dump(mode="json")
        broker = NatsBroker(self.nats_url)
        async with broker:
            await broker.publish(message, JOB_RUN_SUBJECT)


def create_job_publisher() -> JobPublisher:
    nats_url = os.getenv("NATS_URL")
    if not nats_url:
        return NoOpJobPublisher()
    return NatsJobPublisher(nats_url=nats_url)
