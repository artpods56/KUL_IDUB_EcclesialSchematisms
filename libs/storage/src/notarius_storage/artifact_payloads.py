import hashlib
import os
import pathlib
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, Self, final, override
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class SaveArtifactPayloadCommand:
    bucket: str
    key: str
    payload: bytes
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class StoredArtifactPayload:
    bucket: str
    key: str
    payload: bytes
    sha256: str
    byte_size: int

    @classmethod
    def from_payload(cls, *, bucket: str, key: str, payload: bytes) -> Self:
        return cls(
            bucket=bucket,
            key=key,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_size=len(payload),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPayloadLocation:
    bucket: str
    key: str

    @property
    def ref(self) -> str:
        return artifact_payload_ref(bucket=self.bucket, key=self.key)


def artifact_payload_ref(*, bucket: str, key: str) -> str:
    return f"artifact://{bucket}/{key}"


def parse_artifact_payload_ref(payload_ref: str) -> ArtifactPayloadLocation:
    parsed = urlsplit(payload_ref)
    if parsed.scheme != "artifact" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Unsupported artifact payload ref: {payload_ref}")
    return ArtifactPayloadLocation(bucket=parsed.netloc, key=parsed.path.lstrip("/"))


class ArtifactPayloadStoragePort(Protocol):
    def save(self, command: SaveArtifactPayloadCommand) -> StoredArtifactPayload: ...
    def load(self, bucket: str, key: str) -> StoredArtifactPayload: ...
    def delete(self, bucket: str, key: str) -> None: ...
    def exists(self, bucket: str, key: str) -> bool: ...


@final
class LocalArtifactPayloadStorage(ArtifactPayloadStoragePort):
    def __init__(self, storage_root: pathlib.Path | str) -> None:
        self.storage_root = pathlib.Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.storage_root = self.storage_root.resolve()

    @classmethod
    def from_path(cls, storage_root: pathlib.Path | str) -> Self:
        return cls(storage_root=storage_root)

    @override
    def save(self, command: SaveArtifactPayloadCommand) -> StoredArtifactPayload:
        payload_path = self._payload_path(command.bucket, command.key)
        temp_path: pathlib.Path | None = None
        try:
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_inside_storage_root(
                path=payload_path.parent.resolve(strict=True),
                bucket=command.bucket,
                key=command.key,
            )
            temp_file_descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{payload_path.name}.",
                suffix=".tmp",
                dir=payload_path.parent,
            )
            temp_path = pathlib.Path(temp_name)
            with os.fdopen(temp_file_descriptor, "wb") as temp_file:
                temp_file.write(command.payload)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            if command.overwrite:
                os.replace(temp_path, payload_path)
            else:
                os.link(temp_path, payload_path)

            temp_path.unlink(missing_ok=True)
            temp_path = None
        except FileExistsError as exc:
            if not os.path.lexists(payload_path):
                raise OSError(
                    "Failed to save artifact payload for "
                    f"bucket {command.bucket!r} and key {command.key!r}: {exc}"
                ) from exc
            raise FileExistsError(
                "Artifact payload already exists for "
                f"bucket {command.bucket!r} and key {command.key!r}"
            ) from exc
        except OSError as exc:
            raise OSError(
                "Failed to save artifact payload for "
                f"bucket {command.bucket!r} and key {command.key!r}: {exc}"
            ) from exc
        finally:
            if temp_path is not None:
                with suppress(OSError):
                    temp_path.unlink()

        return StoredArtifactPayload.from_payload(
            bucket=command.bucket,
            key=command.key,
            payload=command.payload,
        )

    @override
    def load(self, bucket: str, key: str) -> StoredArtifactPayload:
        payload_path = self._payload_path(bucket, key)
        try:
            payload = payload_path.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Artifact payload not found for bucket {bucket!r} and key {key!r}"
            ) from exc
        except OSError as exc:
            raise OSError(
                f"Failed to load artifact payload for bucket {bucket!r} and key {key!r}: "
                f"{exc}"
            ) from exc

        return StoredArtifactPayload.from_payload(
            bucket=bucket, key=key, payload=payload
        )

    @override
    def delete(self, bucket: str, key: str) -> None:
        payload_path = self._payload_path(bucket, key)
        try:
            payload_path.unlink()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Artifact payload not found for bucket {bucket!r} and key {key!r}"
            ) from exc
        except OSError as exc:
            raise OSError(
                "Failed to delete artifact payload for "
                f"bucket {bucket!r} and key {key!r}: {exc}"
            ) from exc

    @override
    def exists(self, bucket: str, key: str) -> bool:
        return self._payload_path(bucket, key).is_file()

    def _payload_path(self, bucket: str, key: str) -> pathlib.Path:
        bucket_name = self._valid_bucket(bucket)
        key_parts = self._valid_key_parts(key)
        payload_path = self.storage_root.joinpath(bucket_name, *key_parts).resolve(
            strict=False
        )
        self._ensure_inside_storage_root(path=payload_path, bucket=bucket, key=key)
        return payload_path

    def _ensure_inside_storage_root(
        self,
        *,
        path: pathlib.Path,
        bucket: str,
        key: str,
    ) -> None:
        try:
            path.relative_to(self.storage_root)
        except ValueError as exc:
            raise ValueError(
                "Artifact payload path escapes storage root for "
                f"bucket {bucket!r} and key {key!r}"
            ) from exc

    @staticmethod
    def _valid_bucket(bucket: str) -> str:
        if (
            not bucket
            or bucket in {".", ".."}
            or "/" in bucket
            or "\\" in bucket
            or pathlib.PureWindowsPath(bucket).drive
            or "\0" in bucket
        ):
            raise ValueError(f"Invalid artifact payload bucket: {bucket!r}")
        return bucket

    @staticmethod
    def _valid_key_parts(key: str) -> list[str]:
        if (
            not key
            or "\\" in key
            or pathlib.PureWindowsPath(key).drive
            or "\0" in key
            or key.startswith("/")
        ):
            raise ValueError(f"Invalid artifact payload key: {key!r}")

        key_parts = key.split("/")
        if any(part in {"", ".", ".."} for part in key_parts):
            raise ValueError(f"Invalid artifact payload key: {key!r}")

        return key_parts
