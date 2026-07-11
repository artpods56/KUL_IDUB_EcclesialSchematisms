"""Cache decorator for ConfigurableEngine implementations.

This module provides a generic caching wrapper that can be applied to any
ConfigurableEngine, moving caching logic out of use cases and into the
infrastructure layer where it belongs.
"""

import asyncio

from typing import (
    Never,
    Protocol,
    runtime_checkable,
    Any,
    override,
)

from pydantic import BaseModel

from notarius.application.ports.outbound.engine import (
    ConfigurableEngine,
    CachedEngineStats,
    _create_cached_stats,
)
from notarius.domain.protocols import BaseRequest, BaseResponse
from notarius.shared.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class CacheKeyGenerator[RequestT: BaseRequest[Any]](Protocol):
    """Protocol for generating cache keys from requests."""

    def generate_key(self, request: RequestT) -> str:
        """Generate a unique cache key for the given request."""
        ...


@runtime_checkable
class ResponseValidator[ResponseT: BaseResponse[Any]](Protocol):
    """Protocol for validating cached responses before use.

    Implementations can check if a cached response is valid/usable.
    Invalid responses will be treated as cache misses and retried.
    """

    def is_valid(self, response: ResponseT) -> bool:
        """Check if the cached response is valid.

        Args:
            response: The cached response to validate

        Returns:
            True if valid and can be used, False to trigger retry
        """
        ...


@runtime_checkable
class CacheBackend[ResponseT: BaseResponse[Any]](Protocol):
    """Protocol for cache storage backends."""

    def get(self, key: str) -> ResponseT | None:
        """Retrieve cached structured_response by key."""
        ...

    def set(self, key: str, value: ResponseT) -> bool:
        """Store structured_response in cache."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a cached entry by key.

        Args:
            key: The cache key to delete

        Returns:
            True if the key was found and deleted, False otherwise
        """
        ...


class CachedEngine[
    ConfigT: BaseModel,
    RequestT: BaseRequest[object],
    ResponseT: BaseResponse[object],
](ConfigurableEngine[ConfigT, RequestT, ResponseT]):
    """
    Decorator that adds caching capabilities to any ConfigurableEngine.

    This wrapper intercepts the process() method, checks cache before
    delegating to the wrapped engine, and caches the sample.

    Example:
        ```python
        # Create the base engine
        llm_engine_resource = LLMEngine.from_config(config)

        # Wrap it with caching
        cached_engine = CachedEngine(
            engine=llm_engine_resource,
            cache_backend=llm_cache,
            key_generator=llm_key_generator,
            enabled=True
        )

        # Use it exactly like the original engine
        structured_response = cached_engine.process(request)
        ```
    """

    def __init__(
        self,
        engine: ConfigurableEngine[ConfigT, RequestT, ResponseT],
        cache_backend: CacheBackend[ResponseT],
        key_generator: CacheKeyGenerator[RequestT],
        enabled: bool = True,
        response_validator: ResponseValidator[ResponseT] | None = None,
    ):
        """
        Initialize the cached engine wrapper.

        Args:
            engine: The underlying engine to wrap
            cache_backend: Cache storage implementation
            key_generator: Strategy for generating cache keys
            enabled: Whether caching is enabled
            response_validator: Optional validator to check cached responses.
                If validation fails, the cached entry is deleted and retried.
        """
        self._engine = engine
        self._cache = cache_backend
        self._key_generator = key_generator
        self._enabled = enabled
        self._validator = response_validator
        self._stats: CachedEngineStats = _create_cached_stats()

    @classmethod
    @override
    def from_config(cls, config: ConfigT) -> Never:
        """This method should not be called on the wrapper."""
        raise NotImplementedError(
            "CachedEngine should be instantiated with an existing engine, ",
            "not from config directly",
        )

    @override
    def process(self, request: RequestT) -> ResponseT:
        """
        Process request with caching.

        Checks cache first, delegates to wrapped engine on miss,
        and stores sample in cache.
        """
        if not self._enabled:
            return self._engine.process(request)

        try:
            cache_key = self._key_generator.generate_key(request)

            cached_response = self._cache.get(cache_key)
            if cached_response is not None:
                # Validate cached response if validator is provided
                if self._validator is not None and not self._validator.is_valid(
                    cached_response
                ):
                    logger.warning(
                        "Invalid cached response, deleting and retrying",
                        key=cache_key[:16],
                        engine_type=type(self._engine).__name__,
                    )
                    self._cache.delete(cache_key)
                    self._stats["invalidations"] = self._stats.get("invalidations", 0) + 1
                else:
                    self._stats["hits"] += 1
                    logger.debug(
                        "Cache hit",
                        key=cache_key[:16],
                        engine_type=type(self._engine).__name__,
                    )
                    return cached_response

            # Cache miss - process with underlying engine
            self._stats["misses"] += 1
            logger.debug(
                "Cache miss",
                key=cache_key[:16],
                engine_type=type(self._engine).__name__,
            )

            response = self._engine.process(request)

            # Store in cache
            success = self._cache.set(cache_key, response)
            if success:
                logger.debug(
                    "Cached structured_response",
                    key=cache_key[:16],
                    engine_type=type(self._engine).__name__,
                )
            else:
                logger.warning(
                    "Cache set failed",
                    key=cache_key[:16],
                    engine_type=type(self._engine).__name__,
                )

            return response

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(
                "Cache error, falling back to direct processing",
                error=str(e),
                engine_type=type(self._engine).__name__,
            )
            return self._engine.process(request)

    @override
    async def process_async(self, request: RequestT) -> ResponseT:
        """
        Process request with caching (async version).

        Checks cache first, delegates to wrapped engine on miss,
        and stores result in cache.

        Note: Cache operations (get/set) and key generation are offloaded to
        a thread pool via asyncio.to_thread() to avoid blocking the event loop,
        since diskcache performs synchronous disk I/O.
        """
        if not self._enabled:
            return await self._engine.process_async(request)

        try:
            cache_key = await asyncio.to_thread(
                self._key_generator.generate_key, request
            )

            cached_response = await asyncio.to_thread(self._cache.get, cache_key)
            if cached_response is not None:
                # Validate cached response if validator is provided
                is_valid = (
                    self._validator is None
                    or await asyncio.to_thread(
                        self._validator.is_valid, cached_response
                    )
                )
                if not is_valid:
                    logger.warning(
                        "Invalid cached response, deleting and retrying",
                        key=cache_key[:16],
                        engine_type=type(self._engine).__name__,
                    )
                    await asyncio.to_thread(self._cache.delete, cache_key)
                    self._stats["invalidations"] = self._stats.get("invalidations", 0) + 1
                else:
                    self._stats["hits"] += 1
                    logger.debug(
                        "Cache hit",
                        key=cache_key[:16],
                        engine_type=type(self._engine).__name__,
                    )
                    return cached_response

            # Cache miss - process with underlying engine
            self._stats["misses"] += 1
            logger.debug(
                "Cache miss",
                key=cache_key[:16],
                engine_type=type(self._engine).__name__,
            )

            response = await self._engine.process_async(request)

            # Store in cache (offloaded to thread pool)
            success = await asyncio.to_thread(self._cache.set, cache_key, response)
            if success:
                logger.debug(
                    "Cached response",
                    key=cache_key[:16],
                    engine_type=type(self._engine).__name__,
                )
            else:
                logger.warning(
                    "Cache set failed",
                    key=cache_key[:16],
                    engine_type=type(self._engine).__name__,
                )

            return response

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(
                "Cache error, falling back to direct processing",
                error=str(e),
                engine_type=type(self._engine).__name__,
            )
            return await self._engine.process_async(request)

    @property
    @override
    def stats(self) -> CachedEngineStats:
        """Get cache statistics."""
        self._stats["calls"] = self._stats["hits"] + self._stats["misses"]
        return CachedEngineStats(**self._stats)

    @property
    def wrapped_engine(self) -> ConfigurableEngine[ConfigT, RequestT, ResponseT]:
        """Access the underlying engine."""
        return self._engine

    @override
    def clear_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = _create_cached_stats()  # pyright: ignore[reportIncompatibleVariableOverride]
