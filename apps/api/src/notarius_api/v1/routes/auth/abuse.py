"""Small single-process abuse controls for the initial one-owner deployment."""

from collections.abc import Callable
from dataclasses import dataclass
import time
from asyncio import Lock


@dataclass
class _Window:
    started_at: float
    count: int = 0


class AuthAbuseControl:
    _max_tracked_keys = 4096

    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        login_start_limit: int = 10,
        callback_limit: int = 20,
        session_failure_limit: int = 30,
        pat_creation_limit: int = 10,
        outstanding_login_limit: int = 2,
        outstanding_login_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_seconds = window_seconds
        self._limits = {
            "login_start": login_start_limit,
            "callback": callback_limit,
            "session_failure": session_failure_limit,
            "pat_creation": pat_creation_limit,
        }
        self._outstanding_login_limit = outstanding_login_limit
        self._outstanding_login_ttl_seconds = outstanding_login_ttl_seconds
        self._clock = clock
        self._windows: dict[tuple[str, str], _Window] = {}
        self._outstanding_logins: dict[str, _Window] = {}
        self._lock = Lock()

    async def allow_login_start(self, browser_key: str) -> bool:
        return await self._allow("login_start", browser_key)

    async def allow_callback(self, browser_key: str) -> bool:
        return await self._allow("callback", browser_key)

    async def allow_session_failure(self, browser_key: str) -> bool:
        return await self._allow("session_failure", browser_key)

    async def allow_pat_creation(self, user_key: str) -> bool:
        return await self._allow("pat_creation", user_key)

    async def reserve_login(self, browser_key: str) -> bool:
        now = self._clock()
        async with self._lock:
            expired_keys = [
                key
                for key, window in self._outstanding_logins.items()
                if now - window.started_at >= self._outstanding_login_ttl_seconds
            ]
            for expired_key in expired_keys:
                del self._outstanding_logins[expired_key]
            if len(self._outstanding_logins) >= self._max_tracked_keys:
                oldest_key = min(
                    self._outstanding_logins,
                    key=lambda key: self._outstanding_logins[key].started_at,
                )
                del self._outstanding_logins[oldest_key]
            current = self._outstanding_logins.get(browser_key)
            if (
                current is None
                or now - current.started_at >= self._outstanding_login_ttl_seconds
            ):
                current = _Window(started_at=now)
                self._outstanding_logins[browser_key] = current
            if current.count >= self._outstanding_login_limit:
                return False
            current.count += 1
            return True

    async def release_login(self, browser_key: str) -> None:
        async with self._lock:
            current = self._outstanding_logins.get(browser_key)
            if current is None or current.count <= 1:
                self._outstanding_logins.pop(browser_key, None)
            else:
                current.count -= 1

    async def _allow(self, kind: str, key: str) -> bool:
        now = self._clock()
        async with self._lock:
            expired_keys = [
                window_key
                for window_key, window in self._windows.items()
                if now - window.started_at >= self._window_seconds
            ]
            for expired_key in expired_keys:
                del self._windows[expired_key]
            window_key = (kind, key)
            if (
                window_key not in self._windows
                and len(self._windows) >= self._max_tracked_keys
            ):
                oldest_key = min(
                    self._windows,
                    key=lambda key: self._windows[key].started_at,
                )
                del self._windows[oldest_key]
            window = self._windows.get(window_key)
            if window is None or now - window.started_at >= self._window_seconds:
                self._windows[window_key] = _Window(started_at=now, count=1)
                return True
            if window.count >= self._limits[kind]:
                return False
            window.count += 1
            return True


def request_browser_key(request: object) -> str:
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return host if isinstance(host, str) and host else "unknown"


__all__ = ["AuthAbuseControl", "request_browser_key"]
