from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

from typing import final


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class StagedUpload:
    workspace_id: UUID
    upload_key: str
    original_filename: str
    byte_size: int
    created_by_user_id: UUID | None = None
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.upload_key.strip() == "":
            raise ValueError("Staged upload key must not be blank")
        if len(self.upload_key) > 1024:
            raise ValueError("Staged upload key must be at most 1024 characters")
        if self.original_filename.strip() == "":
            raise ValueError("Staged upload original filename must not be blank")
        if len(self.original_filename) > 255:
            raise ValueError(
                "Staged upload original filename must be at most 255 characters"
            )
        if self.byte_size < 0:
            raise ValueError("Staged upload byte size must not be negative")
        if self.created_at.tzinfo is None:
            raise ValueError("Staged upload timestamp must be timezone-aware")


@final
class InMemoryStagedUploadRepository:
    def __init__(self) -> None:
        self._uploads: dict[tuple[UUID, str], StagedUpload] = {}

    async def add(self, upload: StagedUpload) -> None:
        self._uploads[(upload.workspace_id, upload.upload_key)] = upload

    async def get(self, workspace_id: UUID, upload_key: str) -> StagedUpload | None:
        return self._uploads.get((workspace_id, upload_key))

    async def list_for_workspace(self, workspace_id: UUID) -> list[StagedUpload]:
        return [
            upload
            for (stored_workspace_id, _), upload in self._uploads.items()
            if stored_workspace_id == workspace_id
        ]

    async def remove(self, workspace_id: UUID, upload_key: str) -> None:
        self._uploads.pop((workspace_id, upload_key), None)
