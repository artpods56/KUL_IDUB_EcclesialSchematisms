from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from types import CoroutineType
from typing import Any, Concatenate
from collections.abc import Callable, Coroutine, Awaitable
from functools import wraps
from typing import Self, TypedDict

from pydantic import BaseModel

from notarius.domain.protocols import BaseRequest, BaseResponse


class EngineStats(TypedDict):
    """Statistics tracked by all engines."""

    calls: int
    errors: int


class CachedEngineStats(TypedDict):
    """Extended statistics for cached engines."""

    calls: int
    errors: int
    hits: int
    misses: int
    invalidations: int


class ConfigurableEngine[
    ConfigT: BaseModel,
    RequestT: BaseRequest[Any],
    ResponseT: BaseResponse[Any],
](ABC):
    """Base engine interface generic over config, request, and response types.

    Type Parameters:
        ConfigT: The configuration type for this engine
        RequestT: The request type this engine accepts
        ResponseT: The response type this engine returns

    Stats Tracking:
        All engines track basic statistics (calls, errors). Use the @track_stats
        decorator on the process() method to automatically track these stats.
        Subclasses can extend stats by overriding _init_stats() and _stats type.
    """

    _stats: EngineStats

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure subclasses have their own stats instance."""
        super().__init_subclass__(**kwargs)

    def _init_stats(self) -> None:
        """Initialize stats for this engine instance. Call in __init__."""
        self._stats = _create_base_stats()

    @classmethod
    @abstractmethod
    def from_config(cls, config: ConfigT) -> Self:
        raise NotImplementedError

    @abstractmethod
    def process(self, request: RequestT) -> ResponseT:
        raise NotImplementedError

    async def process_async(self, request: RequestT) -> ResponseT:
        """Async version of process. Override in subclasses that support async."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async processing"
        )

    @property
    def stats(self) -> EngineStats:
        """Get a copy of current engine statistics."""
        return EngineStats(**self._stats)

    def clear_stats(self) -> None:
        """Reset engine statistics."""
        self._stats = _create_base_stats()


def _create_base_stats() -> EngineStats:
    """Create a fresh EngineStats instance."""
    return EngineStats(calls=0, errors=0)


def _create_cached_stats() -> CachedEngineStats:
    """Create a fresh CachedEngineStats instance."""
    return CachedEngineStats(calls=0, errors=0, hits=0, misses=0, invalidations=0)


AnyEngine = ConfigurableEngine[Any, Any, Any]


def track_stats[EngineT: AnyEngine, **P, R](
    func: Callable[Concatenate[EngineT, P], R],
) -> Callable[Concatenate[EngineT, P], R]:
    @wraps(func)
    def wrapper(self: EngineT, *args: P.args, **kwargs: P.kwargs) -> R:
        self._stats["calls"] += 1
        try:
            return func(self, *args, **kwargs)
        except Exception:
            self._stats["errors"] += 1
            raise

    return wrapper


def track_stats_async[EngineT: AnyEngine, **P, R](
    func: Callable[Concatenate[EngineT, P], CoroutineType[Any, Any, R]],
) -> Callable[Concatenate[EngineT, P], CoroutineType[Any, Any, R]]:
    @wraps(func)
    async def wrapper(self: EngineT, *args: P.args, **kwargs: P.kwargs) -> R:
        self._stats["calls"] += 1
        try:
            return await func(self, *args, **kwargs)
        except Exception:
            self._stats["errors"] += 1
            raise

    return wrapper
