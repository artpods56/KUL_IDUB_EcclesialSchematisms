"""One-API-owner fence for the initial collaboration deployment."""

import fcntl
import os
from pathlib import Path
from types import TracebackType
from typing import IO, Self


def configured_http_worker_count() -> int:
    """Return the process worker count implied by common ASGI env knobs."""

    for name in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            return int(raw)
        except ValueError as exc:
            raise RuntimeError(
                f"{name} must be an integer worker count; got {raw!r}"
            ) from exc
    return 1


def assert_single_http_worker() -> None:
    worker_count = configured_http_worker_count()
    if worker_count != 1:
        raise RuntimeError(
            "Collaboration requires exactly one API HTTP worker "
            f"(got {worker_count}). Unset WEB_CONCURRENCY/UVICORN_WORKERS or set "
            "them to 1; multi-owner deployment is not supported."
        )


class ApiOwnerLease:
    """Exclusive file lock held for the lifetime of the API process."""

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._handle: IO[str] | None = None

    def acquire(self) -> None:
        assert_single_http_worker()
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                "Another Grafy API owner already holds "
                f"{self._lock_path}. Collaboration assumes one replica and one "
                "worker; stop the other process or leave rooms/executions disabled."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        self._handle = None

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()


__all__ = [
    "ApiOwnerLease",
    "assert_single_http_worker",
    "configured_http_worker_count",
]
