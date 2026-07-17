from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

from notarius_core.domain.artifact_outputs import (
    ArtifactOutputValue,
    normalize_artifact_outputs,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class InvocationCacheEntry:
    key_sha256: str
    outputs: dict[str, ArtifactOutputValue]
    generation: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if len(self.key_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.key_sha256
        ):
            raise ValueError(
                "Invocation cache key SHA-256 must be 64 lowercase hexadecimal "
                "characters"
            )
        if self.created_at.tzinfo is None:
            raise ValueError("Invocation cache timestamp must be timezone-aware")
        self.outputs = normalize_artifact_outputs(self.outputs)
