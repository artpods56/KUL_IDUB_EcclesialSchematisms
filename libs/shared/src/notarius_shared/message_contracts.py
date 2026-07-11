from uuid import UUID

from pydantic import BaseModel

JOB_RUN_SUBJECT = "notarius.jobs.run"


class JobRunRequested(BaseModel):
    job_id: UUID
