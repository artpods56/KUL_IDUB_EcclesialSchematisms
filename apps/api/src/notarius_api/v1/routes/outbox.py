from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from notarius_api import dependencies as deps
from notarius_api.schemas.platform import (
    DlqSummaryResponse,
    OutboxCleanupRequest,
    OutboxCleanupResponse,
    OutboxMessageResponse,
)
from notarius_core.domain.errors import ConflictError, ValidationError
from notarius_core.domain.models import OutboxMessageStatus
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_messaging.contracts import DlqMessage

router = APIRouter(prefix="/outbox-messages", tags=["outbox"])


@router.get("", response_model=list[OutboxMessageResponse])
async def list_outbox_messages(
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    status: OutboxMessageStatus = OutboxMessageStatus.PENDING,
    subject_prefix: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[OutboxMessageResponse]:
    async with uow:
        messages = await uow.outbox_messages.list_by_status(status)
        if subject_prefix is not None:
            messages = [
                message
                for message in messages
                if message.subject.startswith(subject_prefix)
            ]
        return [
            OutboxMessageResponse.from_domain(message)
            for message in messages[offset : offset + limit]
        ]


@router.get("/dlq-summary", response_model=list[DlqSummaryResponse])
async def summarize_dlq_messages(
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
    status: OutboxMessageStatus | None = None,
    consumer_name: str | None = None,
    error_code: str | None = None,
    original_subject: str | None = None,
) -> list[DlqSummaryResponse]:
    async with uow:
        statuses = (
            [status] if status is not None else list(OutboxMessageStatus)
        )
        summaries: dict[tuple[str, str, str], DlqSummaryResponse] = {}
        for current_status in statuses:
            messages = await uow.outbox_messages.list_by_status(current_status)
            for message in messages:
                if message.message_type != DlqMessage.__name__:
                    continue
                dlq_message = DlqMessage.model_validate(message.payload)
                if consumer_name is not None and dlq_message.consumer_name != consumer_name:
                    continue
                if error_code is not None and dlq_message.failure.error_code != error_code:
                    continue
                if (
                    original_subject is not None
                    and dlq_message.original_subject != original_subject
                ):
                    continue

                key = (
                    dlq_message.consumer_name,
                    dlq_message.failure.error_code,
                    dlq_message.original_subject,
                )
                existing = summaries.get(key)
                if existing is None:
                    summaries[key] = DlqSummaryResponse(
                        consumer_name=dlq_message.consumer_name,
                        error_code=dlq_message.failure.error_code,
                        original_subject=dlq_message.original_subject,
                        count=1,
                        latest_failed_at=dlq_message.failed_at,
                        latest_outbox_message_id=message.id,
                    )
                    continue

                existing.count += 1
                if dlq_message.failed_at > existing.latest_failed_at:
                    existing.latest_failed_at = dlq_message.failed_at
                    existing.latest_outbox_message_id = message.id

        return sorted(
            summaries.values(),
            key=lambda summary: (
                summary.consumer_name,
                summary.error_code,
                summary.original_subject,
            ),
        )


@router.post("/cleanup", response_model=OutboxCleanupResponse)
async def cleanup_outbox_messages(
    request: OutboxCleanupRequest,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> OutboxCleanupResponse:
    if request.older_than.tzinfo is None or request.older_than.utcoffset() is None:
        raise ValidationError("older_than must include a timezone")

    statuses = list(dict.fromkeys(request.statuses))
    if OutboxMessageStatus.PENDING in statuses:
        raise ValidationError("Outbox cleanup only supports published and failed statuses")

    async with uow:
        matched_messages = []
        for status in statuses:
            messages = await uow.outbox_messages.list_by_status(status)
            for message in messages:
                if message.created_at >= request.older_than:
                    continue
                if (
                    request.subject_prefix is not None
                    and not message.subject.startswith(request.subject_prefix)
                ):
                    continue
                if (
                    request.message_type is not None
                    and message.message_type != request.message_type
                ):
                    continue
                matched_messages.append(message)

        deleted_count = 0
        if not request.dry_run:
            deleted_count = await uow.outbox_messages.delete_many(
                message.id for message in matched_messages
            )
            await uow.commit()

        return OutboxCleanupResponse(
            dry_run=request.dry_run,
            matched_count=len(matched_messages),
            deleted_count=deleted_count,
            messages=[
                OutboxMessageResponse.from_domain(message)
                for message in matched_messages
            ],
        )


@router.get("/{outbox_message_id}", response_model=OutboxMessageResponse)
async def get_outbox_message(
    outbox_message_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> OutboxMessageResponse:
    async with uow:
        message = await deps.get_outbox_message_or_404(uow, outbox_message_id)
        return OutboxMessageResponse.from_domain(message)


@router.post("/{outbox_message_id}/requeue", response_model=OutboxMessageResponse)
async def requeue_outbox_message(
    outbox_message_id: UUID,
    uow: Annotated[StudioUnitOfWorkPort, Depends(deps.create_uow)],
) -> OutboxMessageResponse:
    async with uow:
        message = await deps.get_outbox_message_or_404(uow, outbox_message_id)
        if message.status != OutboxMessageStatus.FAILED:
            raise ConflictError(
                f"Cannot requeue outbox message {message.id}: status is "
                f"{message.status.value}"
            )

        message.requeue()
        await uow.outbox_messages.update(message)
        await uow.commit()
        return OutboxMessageResponse.from_domain(message)
