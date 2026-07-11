from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

from PIL import Image

from notarius_core.ports import FileStreamProtocol


class ImageRepositoryProtocol(Protocol):
    def get(self, path: Path) -> Image.Image: ...


class FileMetadata(TypedDict, total=False):
    organization_id: str
    accounting_client_id: str
    original_filename: str | None
    source: str
    encrypted: str
    encryption_algorithm: str | None
    artifact_id: str
    job_id: str
    artifact_kind: str
    xml_artifact_id: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SaveFileCommand[MetadataT: FileMetadata]:
    bucket: str
    path: str
    stream: FileStreamProtocol
    content_type: str
    metadata: MetadataT
    allow_overwrite: bool = False
    encryption_aad: str | None = None


@dataclass(frozen=True, slots=True)
class StoredFile:
    bucket: str
    path: str
    etag: str | None
    version_id: str | None
    byte_size: int
    sha256: str


@runtime_checkable
class FileStoragePort(Protocol):
    async def save(self, command: SaveFileCommand[FileMetadata]) -> StoredFile: ...

    async def move(self, bucket: str, source_path: str, destination_path: str) -> None: ...

    async def load(self, bucket: str, path: str) -> FileStreamProtocol: ...

    async def delete(self, bucket: str, path: str) -> None: ...

    def exists(self, bucket: str, path: str) -> bool: ...
