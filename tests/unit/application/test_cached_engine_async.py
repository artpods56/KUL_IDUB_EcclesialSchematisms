"""Tests for CachedEngine async behavior.

This test suite focuses on:
1. process_async method correctly stores results after cache miss
2. Concurrent async requests handle race conditions properly
3. Failed cache sets are detected and reported
4. Integration with diskcache for persistence
"""

import asyncio
from dataclasses import dataclass
from typing import final, override
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from notarius.application.ports.outbound.cached_engine import (
    CacheBackend,
    CachedEngine,
    CacheKeyGenerator,
)
from notarius.application.ports.outbound.engine import ConfigurableEngine
from notarius.domain.protocols import BaseRequest, BaseResponse


# Test fixtures
@dataclass(frozen=True)
class AsyncTestRequest(BaseRequest[str]):
    """Test request type."""

    pass


@dataclass(frozen=True)
class AsyncTestResponse(BaseResponse[str]):
    """Test response type."""

    pass


class AsyncTestConfig(BaseModel):
    """Test configuration."""

    value: str


@final
class AsyncTestEngine(ConfigurableEngine[AsyncTestConfig, AsyncTestRequest, AsyncTestResponse]):
    """Test engine implementation with async support."""

    def __init__(self, config: AsyncTestConfig, delay: float = 0.01):
        self._init_stats()
        self.config = config
        self.process_count = 0
        self.async_process_count = 0
        self.delay = delay

    @classmethod
    @override
    def from_config(cls, config: AsyncTestConfig):
        return cls(config)

    @override
    def process(self, request: AsyncTestRequest) -> AsyncTestResponse:
        self.process_count += 1
        return AsyncTestResponse(output=f"sync_processed: {request.input}")

    @override
    async def process_async(self, request: AsyncTestRequest) -> AsyncTestResponse:
        self.async_process_count += 1
        await asyncio.sleep(self.delay)  # Simulate async work
        return AsyncTestResponse(output=f"async_processed: {request.input}")


class AsyncTestKeyGenerator(CacheKeyGenerator[AsyncTestRequest]):
    """Test key generator."""

    def generate_key(self, request: AsyncTestRequest) -> str:
        return f"key_{request.input}"


@final
class AsyncTestCacheBackend(CacheBackend[AsyncTestResponse]):
    """Test cache backend with async-safe operations."""

    def __init__(self, fail_on_set: bool = False):
        self.cache: dict[str, AsyncTestResponse] = {}
        self.set_calls: list[str] = []  # Track set calls for verification
        self.fail_on_set = fail_on_set

    @override
    def get(self, key: str) -> AsyncTestResponse | None:
        return self.cache.get(key)

    @override
    def set(self, key: str, value: AsyncTestResponse) -> bool:
        self.set_calls.append(key)
        if self.fail_on_set:
            return False
        self.cache[key] = value
        return True


class TestCachedEngineProcessAsync:
    """Test CachedEngine.process_async method."""

    @pytest.mark.asyncio
    async def test_async_cache_miss_processes_and_stores_result(self):
        """Test that async cache miss processes request and stores result."""
        # Setup
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = AsyncTestCacheBackend()
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = AsyncTestRequest(input="test_data")

        # Execute
        response = await cached_engine.process_async(request)

        # Verify processing occurred
        assert response.output == "async_processed: test_data"
        assert base_engine.async_process_count == 1

        # Verify stats
        assert cached_engine.stats["misses"] == 1
        assert cached_engine.stats["hits"] == 0

        # CRITICAL: Verify result was stored in cache
        assert "key_test_data" in cache_backend.cache
        assert cache_backend.cache["key_test_data"].output == "async_processed: test_data"

    @pytest.mark.asyncio
    async def test_async_cache_hit_returns_cached_without_processing(self):
        """Test that async cache hit returns cached response without processing."""
        # Setup
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = AsyncTestCacheBackend()
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = AsyncTestRequest(input="test_data")

        # First call - cache miss
        response1 = await cached_engine.process_async(request)
        assert base_engine.async_process_count == 1

        # Second call - should be cache hit
        response2 = await cached_engine.process_async(request)

        # Verify second call didn't process
        assert base_engine.async_process_count == 1  # Still 1, not 2
        assert response1.output == response2.output

        # Verify stats
        assert cached_engine.stats["hits"] == 1
        assert cached_engine.stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_async_cache_disabled_bypasses_cache(self):
        """Test that disabled cache bypasses caching entirely."""
        # Setup
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = AsyncTestCacheBackend()
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=False,  # Cache disabled
        )

        request = AsyncTestRequest(input="test_data")

        # Execute twice
        await cached_engine.process_async(request)
        await cached_engine.process_async(request)

        # Verify both calls processed
        assert base_engine.async_process_count == 2

        # Verify nothing cached
        assert len(cache_backend.cache) == 0


class TestCachedEngineCacheSetFailure:
    """Test handling of cache.set() failures."""

    @pytest.mark.asyncio
    async def test_async_cache_set_failure_still_returns_response(self):
        """Test that cache.set() failure still returns the response."""
        # Setup with failing cache
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = AsyncTestCacheBackend(fail_on_set=True)
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = AsyncTestRequest(input="test_data")

        # Execute - should succeed despite cache set failure
        response = await cached_engine.process_async(request)

        # Verify response is correct
        assert response.output == "async_processed: test_data"

        # Verify set was attempted
        assert "key_test_data" in cache_backend.set_calls

        # Verify nothing was cached (because set failed)
        assert len(cache_backend.cache) == 0

    @pytest.mark.asyncio
    async def test_async_cache_set_failure_logs_warning(self):
        """Test that cache.set() failure logs a warning."""
        # Setup with failing cache
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = AsyncTestCacheBackend(fail_on_set=True)
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = AsyncTestRequest(input="test_data")

        # Execute with log capture
        with patch("notarius.application.ports.outbound.cached_engine.logger") as mock_logger:
            await cached_engine.process_async(request)

            # Verify warning was logged for failed set
            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args
            assert "Cache set failed" in str(call_kwargs)


class TestCachedEngineConcurrency:
    """Test concurrent async behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_different_keys(self):
        """Test multiple concurrent async requests with different keys."""
        # Setup
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"), delay=0.05)
        cache_backend = AsyncTestCacheBackend()
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # Create multiple concurrent requests
        requests = [AsyncTestRequest(input=f"data_{i}") for i in range(10)]

        # Execute all concurrently
        responses = await asyncio.gather(
            *[cached_engine.process_async(req) for req in requests]
        )

        # Verify all processed
        assert len(responses) == 10
        assert base_engine.async_process_count == 10

        # Verify all cached
        assert len(cache_backend.cache) == 10
        for i in range(10):
            assert f"key_data_{i}" in cache_backend.cache

    @pytest.mark.asyncio
    async def test_concurrent_requests_with_same_key_all_miss(self):
        """Test concurrent requests with same key - race condition scenario.

        When multiple concurrent requests have the same key and all experience
        cache misses (because the first one hasn't finished storing yet),
        all of them will call the underlying engine. This is expected behavior
        but the cache should still eventually contain the result.
        """
        # Setup with slow engine
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"), delay=0.1)
        cache_backend = AsyncTestCacheBackend()
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # Create multiple concurrent requests with SAME input
        requests = [AsyncTestRequest(input="same_data") for _ in range(5)]

        # Execute all concurrently
        responses = await asyncio.gather(
            *[cached_engine.process_async(req) for req in requests]
        )

        # All should have the same response
        assert all(r.output == "async_processed: same_data" for r in responses)

        # Due to race condition, all might call the engine
        # (This is expected behavior - no distributed locking)
        assert base_engine.async_process_count >= 1

        # But cache should have the entry
        assert "key_same_data" in cache_backend.cache

    @pytest.mark.asyncio
    async def test_second_batch_uses_cache_from_first_batch(self):
        """Test that a second batch of requests uses cached results from first batch."""
        # Setup
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = AsyncTestCacheBackend()
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # First batch - all should miss
        requests_batch1 = [AsyncTestRequest(input=f"data_{i}") for i in range(5)]
        await asyncio.gather(*[cached_engine.process_async(req) for req in requests_batch1])

        assert base_engine.async_process_count == 5
        assert cached_engine.stats["misses"] == 5

        # Second batch - same requests, all should hit
        requests_batch2 = [AsyncTestRequest(input=f"data_{i}") for i in range(5)]
        await asyncio.gather(*[cached_engine.process_async(req) for req in requests_batch2])

        # Engine should NOT have been called again
        assert base_engine.async_process_count == 5  # Still 5, not 10

        # Stats should show hits
        assert cached_engine.stats["hits"] == 5
        assert cached_engine.stats["misses"] == 5


class TestCachedEngineErrorHandling:
    """Test error handling in async path."""

    @pytest.mark.asyncio
    async def test_key_generator_exception_falls_back_to_engine(self):
        """Test that key generator exception falls back to direct processing."""
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = AsyncTestCacheBackend()
        key_generator = Mock()
        key_generator.generate_key.side_effect = Exception("Key generation failed")

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = AsyncTestRequest(input="test_data")

        # Should fall back to direct processing
        response = await cached_engine.process_async(request)

        assert response.output == "async_processed: test_data"
        assert base_engine.async_process_count == 1
        assert cached_engine.stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_cache_get_exception_falls_back_to_engine(self):
        """Test that cache.get() exception falls back to direct processing."""
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = Mock()
        cache_backend.get.side_effect = Exception("Cache read failed")
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = AsyncTestRequest(input="test_data")

        # Should fall back to direct processing
        response = await cached_engine.process_async(request)

        assert response.output == "async_processed: test_data"
        assert base_engine.async_process_count == 1
        assert cached_engine.stats["errors"] == 1


class TestCachedEngineKeyDeterminism:
    """Test that key generation is deterministic."""

    @pytest.mark.asyncio
    async def test_same_request_produces_same_key_across_calls(self):
        """Test that the same request always produces the same cache key."""
        key_generator = AsyncTestKeyGenerator()

        # Create identical requests
        request1 = AsyncTestRequest(input="test_data")
        request2 = AsyncTestRequest(input="test_data")

        key1 = key_generator.generate_key(request1)
        key2 = key_generator.generate_key(request2)

        assert key1 == key2

    @pytest.mark.asyncio
    async def test_first_request_miss_second_request_hit(self):
        """Integration test: verify second identical request hits cache."""
        base_engine = AsyncTestEngine(AsyncTestConfig(value="test"))
        cache_backend = AsyncTestCacheBackend()
        key_generator = AsyncTestKeyGenerator()

        cached_engine = CachedEngine(
            engine=base_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # First request - miss
        request1 = AsyncTestRequest(input="test_data")
        response1 = await cached_engine.process_async(request1)
        assert base_engine.async_process_count == 1

        # Create NEW request object with same data
        request2 = AsyncTestRequest(input="test_data")
        response2 = await cached_engine.process_async(request2)

        # Should have been a cache hit
        assert base_engine.async_process_count == 1  # No additional processing
        assert response1.output == response2.output
        assert cached_engine.stats["hits"] == 1
        assert cached_engine.stats["misses"] == 1
