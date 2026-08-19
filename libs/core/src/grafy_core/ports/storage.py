from dataclasses import dataclass
from typing import BinaryIO, Protocol, TypedDict, runtime_checkable


FileStreamProtocol = BinaryIO


class FileMetadata(TypedDict, total=False):
    original_filename: str | None
    source: str
    artifact_id: str
    artifact_kind: str
    job_id: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SaveFileCommand:
    bucket: str
    path: str
    stream: FileStreamProtocol
    content_type: str
    metadata: FileMetadata
    allow_overwrite: bool = False


@dataclass(frozen=True, slots=True)
class StoredFile:
    bucket: str
    path: str
    etag: str | None
    version_id: str | None
    byte_size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class StoredObjectInfo:
    bucket: str
    path: str
    byte_size: int
    etag: str | None
    version_id: str | None


@runtime_checkable
class FileStoragePort(Protocol):
    async def save(self, command: SaveFileCommand) -> StoredFile: ...

    async def move(self, bucket: str, source_path: str, destination_path: str) -> None: ...

    async def load(self, bucket: str, path: str) -> FileStreamProtocol: ...

    async def stat(self, bucket: str, path: str) -> StoredObjectInfo | None: ...

    async def load_range(
        self,
        bucket: str,
        path: str,
        start: int,
        end_exclusive: int,
    ) -> bytes: ...

    async def delete(self, bucket: str, path: str) -> None: ...
