import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


class ObjectStoreError(RuntimeError):
    """Raised when an object store URI cannot be resolved."""


@dataclass(frozen=True)
class LocalS3ObjectStore:
    """Filesystem-backed object store that exposes S3-style URIs.

    This is a prototype adapter: API and worker can share a mounted directory while
    the domain still passes around durable `s3://bucket/key` references.
    """

    root: Path
    bucket: str

    def put_bytes(self, key: str, content: bytes) -> str:
        object_key = self._clean_key(key)
        path = self.root / self.bucket / object_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return f"s3://{self.bucket}/{object_key.as_posix()}"

    def get_bytes(self, uri: str) -> bytes:
        return self.path_for_uri(uri).read_bytes()

    def get_text(self, uri: str, encoding: str = "utf-8") -> str:
        return self.get_bytes(uri).decode(encoding)

    def path_for_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ObjectStoreError(f"Unsupported object URI: {uri}")
        return self.root / self.bucket / self._clean_key(parsed.path.lstrip("/"))

    @staticmethod
    def _clean_key(key: str) -> PurePosixPath:
        object_key = PurePosixPath(key)
        if object_key.is_absolute() or ".." in object_key.parts or str(object_key) in {"", "."}:
            raise ObjectStoreError(f"Unsafe object key: {key}")
        return object_key


def create_local_s3_object_store() -> LocalS3ObjectStore:
    return LocalS3ObjectStore(
        root=Path(os.getenv("NOTARIUS_OBJECT_STORAGE_DIR", ".notarius-objects")),
        bucket=os.getenv("NOTARIUS_OBJECT_BUCKET", "notarius-studio"),
    )
