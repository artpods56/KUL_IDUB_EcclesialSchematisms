"""Small single-process abuse controls for the initial one-owner deployment."""

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import time
from asyncio import Lock
from secrets import token_urlsafe
from uuid import UUID

from fastapi import Response


BROWSER_ABUSE_COOKIE = "notarius_browser_abuse"
_BROWSER_ABUSE_COOKIE_MAX_AGE = 24 * 60 * 60


@dataclass
class _Window:
    started_at: float
    count: int = 0


@dataclass
class _LoginReservation:
    expires_at: float


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
        self._windows: dict[tuple[str, bytes], _Window] = {}
        self._outstanding_logins: dict[bytes, dict[UUID, _LoginReservation]] = {}
        self._lock = Lock()

    async def allow_login_start(self, browser_key: str) -> bool:
        return await self._allow("login_start", browser_key)

    async def allow_callback(self, browser_key: str) -> bool:
        return await self._allow("callback", browser_key)

    async def allow_session_failure(self, browser_key: str) -> bool:
        return await self._allow("session_failure", browser_key)

    async def allow_pat_creation(self, user_key: str) -> bool:
        return await self._allow("pat_creation", user_key)

    async def reserve_login(self, browser_key: str, transaction_id: UUID) -> bool:
        now = self._clock()
        browser_digest = _digest(browser_key)
        async with self._lock:
            expired_browser_digests: list[bytes] = []
            for digest, reservations in self._outstanding_logins.items():
                expired_ids = [
                    reservation_id
                    for reservation_id, reservation in reservations.items()
                    if reservation.expires_at <= now
                ]
                for reservation_id in expired_ids:
                    del reservations[reservation_id]
                if not reservations:
                    expired_browser_digests.append(digest)
            for digest in expired_browser_digests:
                del self._outstanding_logins[digest]

            current = self._outstanding_logins.get(browser_digest)
            if (
                current is None
                and len(self._outstanding_logins) >= self._max_tracked_keys
            ):
                return False
            if current is None:
                current = {}
                self._outstanding_logins[browser_digest] = current
            if len(current) >= self._outstanding_login_limit:
                return False
            current[transaction_id] = _LoginReservation(
                expires_at=now + self._outstanding_login_ttl_seconds
            )
            return True

    async def release_login(self, browser_key: str, transaction_id: UUID) -> None:
        browser_digest = _digest(browser_key)
        async with self._lock:
            current = self._outstanding_logins.get(browser_digest)
            if current is None:
                return
            current.pop(transaction_id, None)
            if not current:
                self._outstanding_logins.pop(browser_digest, None)

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
            window_key = (kind, _digest(key))
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
    state = getattr(request, "state", None)
    cached = getattr(state, "auth_browser_key", None)
    if isinstance(cached, str):
        return cached
    cookies = getattr(request, "cookies", {})
    browser_key = cookies.get(BROWSER_ABUSE_COOKIE)
    if not isinstance(browser_key, str) or not browser_key or len(browser_key) > 128:
        path = getattr(getattr(request, "url", None), "path", "")
        if isinstance(path, str) and "/auth/oidc/" not in path:
            client = getattr(request, "client", None)
            host = getattr(client, "host", None)
            browser_key = (
                f"ip:{host}" if isinstance(host, str) and host else "ip:unknown"
            )
        else:
            browser_key = token_urlsafe(32)
    if state is not None:
        state.auth_browser_key = browser_key
    return browser_key


def set_browser_abuse_cookie(
    response: Response, browser_key: str, *, secure: bool
) -> None:
    response.set_cookie(
        BROWSER_ABUSE_COOKIE,
        browser_key,
        max_age=_BROWSER_ABUSE_COOKIE_MAX_AGE,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/api/v1/auth/oidc",
    )


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


__all__ = [
    "AuthAbuseControl",
    "BROWSER_ABUSE_COOKIE",
    "request_browser_key",
    "set_browser_abuse_cookie",
]
