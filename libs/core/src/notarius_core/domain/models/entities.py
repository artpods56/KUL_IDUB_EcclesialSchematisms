import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4


class ArtifactKind(StrEnum):
    IMAGE = "image"

class ArtifactStatus(StrEnum):
    FAILED = "failed"
    AVAILABLE = "available"
    PENDING = "pending"

type ContentType = Literal["image/png", "image/jpeg", "application/pdf", "text/plain", "application/xml", "application/json"]

type StorageBackend = Literal["s3", "local"]

def uuid() -> UUID:
    return uuid4()

@dataclass(kw_only=True)
class Base(abc.ABC):
    id: UUID  = field(default_factory=uuid4)

@dataclass
class ArtifactObject(Base):
    organization_id: UUID
    kind: ArtifactKind
    storage_backend: StorageBackend
    bucket: str
    object_key: str
    content_type: ContentType
    created_at: datetime
    byte_size: int
    original_filename: str
    sha256: str