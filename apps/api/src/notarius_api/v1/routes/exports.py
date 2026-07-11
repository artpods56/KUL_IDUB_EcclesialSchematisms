import csv
import io
import json
from dataclasses import asdict
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from notarius_api import dependencies as deps
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort

router = APIRouter(prefix="/jobs", tags=["exports"])


@router.get("/{job_id}/exports/jsonl", response_class=PlainTextResponse)
async def export_job_jsonl(
    job_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> PlainTextResponse:
    async with uow:
        await deps.get_job_or_404(uow, job_id)
        lines = []
        for item in await uow.job_items.list_for_job(job_id):
            lines.append(
                json.dumps(
                    {
                        "job_item_id": str(item.id),
                        "source_item_id": str(item.source_item_id),
                        "order": item.order,
                        "status": item.status,
                        "structured_output": item.structured_output,
                        "context_trace": asdict(item.context_trace)
                        if item.context_trace
                        else None,
                    },
                    default=str,
                )
            )
        return PlainTextResponse("\n".join(lines), media_type="application/x-jsonlines")


@router.get("/{job_id}/exports/csv", response_class=PlainTextResponse)
async def export_job_csv(
    job_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> PlainTextResponse:
    async with uow:
        await deps.get_job_or_404(uow, job_id)
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=["job_item_id", "source_item_id", "order", "status", "output"],
        )
        writer.writeheader()
        for item in await uow.job_items.list_for_job(job_id):
            writer.writerow(
                {
                    "job_item_id": item.id,
                    "source_item_id": item.source_item_id,
                    "order": item.order,
                    "status": item.status,
                    "output": json.dumps(item.structured_output or {}, default=str),
                }
            )
        return PlainTextResponse(buffer.getvalue(), media_type="text/csv")
