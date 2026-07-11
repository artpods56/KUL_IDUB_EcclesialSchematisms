import os
import shutil
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import final, override

from notarius_core.domain.errors import ObjectAlreadyExistsError
from notarius_core.ports.storage import (
    FileStoragePort,
    FileStreamProtocol,
    SaveFileCommand,
    StoredFile,
)


@final
class LocalFileObjectStore(FileStoragePort):
    def __init__(self, root: Path):
        self._root = root

    @override
    async def save(self, command: SaveFileCommand) -> StoredFile:
        path = self._path_for(command.bucket, command.path)
        if path.exists() and not command.allow_overwrite:
            raise ObjectAlreadyExistsError(f"File already exists: {command.bucket}/{command.path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        digest = sha256()
        byte_size = 0

        with temp_path.open("wb") as target:
            while chunk := command.stream.read(1024 * 1024):
                digest.update(chunk)
                byte_size += len(chunk)
                _ = target.write(chunk)
            target.flush()
            os.fsync(target.fileno())

        os.replace(temp_path, path)
        return StoredFile(
            bucket=command.bucket,
            path=command.path,
            etag=None,
            version_id=None,
            byte_size=byte_size,
            sha256=digest.hexdigest(),
        )

    @override
    async def move(self, bucket: str, source_path: str, destination_path: str) -> None:
        source = self._path_for(bucket, source_path)
        destination = self._path_for(bucket, destination_path)
        if not source.exists() and destination.exists():
            return
        if not source.exists():
            raise FileNotFoundError(f"Source file does not exist: {bucket}/{source_path}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        _ = shutil.move(str(source), str(destination))

    @override
    async def load(self, bucket: str, path: str) -> FileStreamProtocol:
        return self._path_for(bucket, path).open("rb")

    @override
    async def delete(self, bucket: str, path: str) -> None:
        file_path = self._path_for(bucket, path)
        if file_path.exists():
            file_path.unlink()

    @override
    def exists(self, bucket: str, path: str) -> bool:
        return self._path_for(bucket, path).is_file()

    def _path_for(self, bucket: str, key: str) -> Path:
        _validate_segment(bucket)
        _validate_key(key)
        return self._root / bucket / Path(*PurePosixPath(key).parts)


def _validate_segment(segment: str) -> None:
    if not segment or "/" in segment or "\\" in segment or segment in {".", ".."}:
        raise ValueError(f"Unsafe storage segment: {segment}")


def _validate_key(key: str) -> None:
    path = PurePosixPath(key)
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        raise ValueError(f"Unsafe object key: {key}")
#
