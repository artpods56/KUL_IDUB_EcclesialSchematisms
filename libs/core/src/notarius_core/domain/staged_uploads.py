from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

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
        if (
            self.upload_key.strip() == ""
            or self.upload_key in {".", ".."}
            or "/" in self.upload_key
            or "\\" in self.upload_key
            or "\x00" in self.upload_key
        ):
            raise ValueError(
                "Staged upload key must be a non-path opaque key"
            )
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
