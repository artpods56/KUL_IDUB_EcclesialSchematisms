"""Integration tests for CachedEngine with actual diskcache backend.

These tests verify the full stack:
- CachedEngine -> LLMCacheBackend -> LLMCache -> diskcache

This is where the actual bug might be - in the serialization/deserialization
or the persistence layer.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Never, final, override

import pytest
from pydantic import BaseModel

from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.application.ports.outbound.engine import ConfigurableEngine
from notarius.domain.entities.completions import BaseProviderResponse
from notarius.domain.entities.messages import ChatMessage, TextContent, ImageContent
from notarius.domain.protocols import BaseRequest, BaseResponse
from notarius.infrastructure.cache.adapters.llm import LLMCache
from notarius.infrastructure.cache.backends.llm import (
    LLMCacheBackend,
    LLMCacheKeyGenerator,
    create_llm_cache_backend,
)
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import (
    CompletionRequest,
    CompletionResult,
)


# Test fixtures for LLM-like responses
class TestSchema(BaseModel):
    """Test structured output schema."""

    text: str
    confidence: float


@dataclass(frozen=True)
class MockLLMResponse(BaseProviderResponse[TestSchema]):
    """Mock LLM provider response."""

    def to_string(self) -> str:
        if self.structured_response:
            return self.structured_response.model_dump_json()
        return self.text_response or ""


class MockLLMConfig(BaseModel):
    """Mock LLM config."""

    model_name: str


@final
class MockLLMEngine(ConfigurableEngine[MockLLMConfig, CompletionRequest, CompletionResult]):
    """Mock LLM engine that simulates API calls."""

    def __init__(self, config: MockLLMConfig):
        self._init_stats()
        self.config = config
        self.call_count = 0
        self.async_call_count = 0

    @classmethod
    @override
    def from_config(cls, config: MockLLMConfig):
        return cls(config)

    @override
    def process(self, request: CompletionRequest) -> CompletionResult:
        self.call_count += 1
        return self._create_response(request)

    @override
    async def process_async(self, request: CompletionRequest) -> CompletionResult:
        self.async_call_count += 1
        await asyncio.sleep(0.01)  # Simulate API latency
        return self._create_response(request)

    def _create_response(self, request: CompletionRequest) -> CompletionResult:
        """Create a mock LLM response."""
        response = MockLLMResponse(
            structured_response=None,
            text_response=f"Response to: {request.input.messages[-1].content[0].text[:50]}",
        )

        # Add assistant response to conversation
        updated_conv = request.input.add(
            ChatMessage(
                role="assistant",
                content=[TextContent(text=response.text_response)],
            )
        )

        return CompletionResult(output=response, conversation=updated_conv)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory."""
    return tmp_path / "test_caches"


@pytest.fixture
def mock_llm_engine() -> MockLLMEngine:
    """Create mock LLM engine."""
    return MockLLMEngine(MockLLMConfig(model_name="test-model"))


@pytest.fixture
def llm_cache(tmp_cache_dir: Path) -> LLMCache:
    """Create LLMCache with temporary directory."""
    return LLMCache(model_name="test-model", caches_dir=tmp_cache_dir)


@pytest.fixture
def key_generator() -> LLMCacheKeyGenerator:
    """Create key generator."""
    return LLMCacheKeyGenerator()


@pytest.fixture
def cache_backend(llm_cache: LLMCache, key_generator: LLMCacheKeyGenerator) -> LLMCacheBackend:
    """Create cache backend."""
    return LLMCacheBackend(cache=llm_cache, key_generator=key_generator)


def create_test_conversation(user_text: str) -> Conversation:
    """Create a test conversation."""
    return Conversation.from_messages([
        ChatMessage(role="system", content=[TextContent(text="You are helpful")]),
        ChatMessage(role="user", content=[TextContent(text=user_text)]),
    ])


def create_image_conversation(user_text: str, image_data: str = "abc123") -> Conversation:
    """Create a conversation with an image."""
    return Conversation.from_messages([
        ChatMessage(role="system", content=[TextContent(text="You are helpful")]),
        ChatMessage(
            role="user",
            content=[
                TextContent(text=user_text),
                ImageContent(image_url=f"data:image/jpeg;base64,{image_data}", detail="high"),
            ],
        ),
    ])


class TestCachedEngineWithDiskcache:
    """Integration tests with actual diskcache backend."""

    @pytest.mark.asyncio
    async def test_async_cache_miss_stores_in_diskcache(
        self,
        mock_llm_engine: MockLLMEngine,
        cache_backend: LLMCacheBackend,
        key_generator: LLMCacheKeyGenerator,
        llm_cache: LLMCache,
    ):
        """Test that async cache miss stores result in diskcache."""
        cached_engine = CachedEngine(
            engine=mock_llm_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request = CompletionRequest(
            input=create_test_conversation("Hello world"),
            structured_output=None,
        )

        # Execute - should miss and store
        response = await cached_engine.process_async(request)

        assert response.output.text_response is not None
        assert mock_llm_engine.async_call_count == 1

        # Verify stored in actual cache
        cache_key = key_generator.generate_key(request)
        cached = llm_cache.get(cache_key)

        assert cached is not None, "Response was NOT stored in diskcache!"
        assert cached.output.text_response == response.output.text_response

    @pytest.mark.asyncio
    async def test_async_second_request_hits_diskcache(
        self,
        mock_llm_engine: MockLLMEngine,
        cache_backend: LLMCacheBackend,
        key_generator: LLMCacheKeyGenerator,
    ):
        """Test that second identical async request hits diskcache."""
        cached_engine = CachedEngine(
            engine=mock_llm_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        request1 = CompletionRequest(
            input=create_test_conversation("Test message"),
            structured_output=None,
        )

        # First call - miss
        response1 = await cached_engine.process_async(request1)
        assert mock_llm_engine.async_call_count == 1

        # Create NEW request with identical content
        request2 = CompletionRequest(
            input=create_test_conversation("Test message"),
            structured_output=None,
        )

        # Second call - should hit
        response2 = await cached_engine.process_async(request2)

        # Engine should NOT have been called again
        assert mock_llm_engine.async_call_count == 1, "Engine was called twice - cache miss!"
        assert response1.output.text_response == response2.output.text_response

    @pytest.mark.asyncio
    async def test_concurrent_async_requests_all_get_cached(
        self,
        mock_llm_engine: MockLLMEngine,
        cache_backend: LLMCacheBackend,
        key_generator: LLMCacheKeyGenerator,
        llm_cache: LLMCache,
    ):
        """Test that concurrent requests with different keys all get cached."""
        cached_engine = CachedEngine(
            engine=mock_llm_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # Create requests with different content
        requests = [
            CompletionRequest(
                input=create_test_conversation(f"Message {i}"),
                structured_output=None,
            )
            for i in range(10)
        ]

        # Execute concurrently
        responses = await asyncio.gather(
            *[cached_engine.process_async(req) for req in requests]
        )

        assert len(responses) == 10
        assert mock_llm_engine.async_call_count == 10

        # Verify ALL were cached
        for i, req in enumerate(requests):
            cache_key = key_generator.generate_key(req)
            cached = llm_cache.get(cache_key)
            assert cached is not None, f"Request {i} was NOT cached!"

    @pytest.mark.asyncio
    async def test_image_conversation_caching(
        self,
        mock_llm_engine: MockLLMEngine,
        cache_backend: LLMCacheBackend,
        key_generator: LLMCacheKeyGenerator,
        llm_cache: LLMCache,
    ):
        """Test caching conversations with images."""
        cached_engine = CachedEngine(
            engine=mock_llm_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        # Create request with image
        request = CompletionRequest(
            input=create_image_conversation("What is in this image?", "base64imagedata"),
            structured_output=None,
        )

        # Execute
        response = await cached_engine.process_async(request)
        assert mock_llm_engine.async_call_count == 1

        # Verify cached
        cache_key = key_generator.generate_key(request)
        cached = llm_cache.get(cache_key)
        assert cached is not None, "Image conversation was NOT cached!"

        # Second request should hit
        request2 = CompletionRequest(
            input=create_image_conversation("What is in this image?", "base64imagedata"),
            structured_output=None,
        )
        await cached_engine.process_async(request2)
        assert mock_llm_engine.async_call_count == 1, "Image request missed cache!"


class TestCacheKeyDeterminism:
    """Test that cache key generation is deterministic."""

    def test_same_text_conversation_same_key(self, key_generator: LLMCacheKeyGenerator):
        """Test identical text conversations produce identical keys."""
        conv1 = create_test_conversation("Hello")
        conv2 = create_test_conversation("Hello")

        req1 = CompletionRequest(input=conv1, structured_output=None)
        req2 = CompletionRequest(input=conv2, structured_output=None)

        key1 = key_generator.generate_key(req1)
        key2 = key_generator.generate_key(req2)

        assert key1 == key2

    def test_same_image_conversation_same_key(self, key_generator: LLMCacheKeyGenerator):
        """Test identical image conversations produce identical keys."""
        conv1 = create_image_conversation("Describe", "imagedata123")
        conv2 = create_image_conversation("Describe", "imagedata123")

        req1 = CompletionRequest(input=conv1, structured_output=None)
        req2 = CompletionRequest(input=conv2, structured_output=None)

        key1 = key_generator.generate_key(req1)
        key2 = key_generator.generate_key(req2)

        assert key1 == key2

    def test_different_image_data_different_key(self, key_generator: LLMCacheKeyGenerator):
        """Test different image data produces different keys."""
        conv1 = create_image_conversation("Describe", "imagedata_A")
        conv2 = create_image_conversation("Describe", "imagedata_B")

        req1 = CompletionRequest(input=conv1, structured_output=None)
        req2 = CompletionRequest(input=conv2, structured_output=None)

        key1 = key_generator.generate_key(req1)
        key2 = key_generator.generate_key(req2)

        assert key1 != key2


class TestCachePersistence:
    """Test that cache persists across engine instances."""

    @pytest.mark.asyncio
    async def test_cache_persists_across_cached_engine_instances(
        self,
        tmp_cache_dir: Path,
    ):
        """Test that cached results persist when creating new CachedEngine instances."""
        # First engine instance
        engine1 = MockLLMEngine(MockLLMConfig(model_name="test"))
        cache1 = LLMCache(model_name="test-model", caches_dir=tmp_cache_dir)
        keygen1 = LLMCacheKeyGenerator()
        backend1 = LLMCacheBackend(cache=cache1, key_generator=keygen1)
        cached_engine1 = CachedEngine(
            engine=engine1,
            cache_backend=backend1,
            key_generator=keygen1,
            enabled=True,
        )

        request = CompletionRequest(
            input=create_test_conversation("Persist test"),
            structured_output=None,
        )

        # Process with first engine
        await cached_engine1.process_async(request)
        assert engine1.async_call_count == 1

        # Create COMPLETELY NEW instances
        engine2 = MockLLMEngine(MockLLMConfig(model_name="test"))
        cache2 = LLMCache(model_name="test-model", caches_dir=tmp_cache_dir)
        keygen2 = LLMCacheKeyGenerator()
        backend2 = LLMCacheBackend(cache=cache2, key_generator=keygen2)
        cached_engine2 = CachedEngine(
            engine=engine2,
            cache_backend=backend2,
            key_generator=keygen2,
            enabled=True,
        )

        # Same request with new engine
        request2 = CompletionRequest(
            input=create_test_conversation("Persist test"),
            structured_output=None,
        )

        # Should hit cache
        await cached_engine2.process_async(request2)

        # Engine2 should NOT have been called
        assert engine2.async_call_count == 0, (
            "Cache did NOT persist across engine instances! "
            "The second engine was called when it should have hit cache."
        )


class TestDiskcacheSetReturnValue:
    """Test that diskcache.set() return value is properly handled."""

    @pytest.mark.asyncio
    async def test_diskcache_set_returns_true_on_success(
        self,
        llm_cache: LLMCache,
        key_generator: LLMCacheKeyGenerator,
    ):
        """Verify diskcache.set() returns True on successful write."""
        conv = create_test_conversation("Test")
        response = MockLLMResponse(structured_response=None, text_response="Test response")
        result = CompletionResult(
            output=response,
            conversation=conv.add(
                ChatMessage(role="assistant", content=[TextContent(text="Test")])
            ),
        )

        success = llm_cache.set("test_key", result)

        assert success is True, "diskcache.set() did not return True!"

        # Verify it was actually stored
        cached = llm_cache.get("test_key")
        assert cached is not None

    @pytest.mark.asyncio
    async def test_verify_cache_entry_count_increases(
        self,
        mock_llm_engine: MockLLMEngine,
        cache_backend: LLMCacheBackend,
        key_generator: LLMCacheKeyGenerator,
        llm_cache: LLMCache,
    ):
        """Test that cache entry count increases after successful caching."""
        cached_engine = CachedEngine(
            engine=mock_llm_engine,
            cache_backend=cache_backend,
            key_generator=key_generator,
            enabled=True,
        )

        initial_count = len(llm_cache)

        # Make 5 unique requests
        for i in range(5):
            request = CompletionRequest(
                input=create_test_conversation(f"Unique message {i}"),
                structured_output=None,
            )
            await cached_engine.process_async(request)

        final_count = len(llm_cache)

        assert final_count == initial_count + 5, (
            f"Expected {initial_count + 5} entries but got {final_count}. "
            "Cache entries are NOT being added!"
        )
