"""LLM cache backend and key generator for cached engine pattern."""

from __future__ import annotations
from notarius.shared.constants import CACHES_DIR
from notarius.infrastructure.persistence.storage.local import LocalFileStorage
import hashlib
import json
from typing import final, override, cast, Any

from pydantic import BaseModel
from structlog import get_logger

from notarius.application.ports.outbound.cached_engine import (
    CacheBackend,
    CacheKeyGenerator,
)
from notarius.infrastructure.cache.adapters.llm import LLMCache
from notarius.infrastructure.cache.storage.utils import (
    get_base64_hash,
    conversation_with_refs,
    conversation_with_images,
)
from notarius.infrastructure.llm.engine_adapter import (
    CompletionResult,
    CompletionRequest,
)
from notarius.infrastructure.persistence.storage.local import ImageRepository
from notarius.shared.logger import Logger

logger = cast(Logger, get_logger(__name__))


def _process_payload_for_hashing(payload: Any) -> Any:
    """Recursively replace large base64 images with their hashes in a dictionary/list.

    This ensures that the cache key generation is fast and the payload is compact,
    while remaining deterministic.
    """
    if isinstance(payload, dict):
        new_dict = {}
        for k, v in payload.items():
            if k == "image_url" and isinstance(v, str) and v.startswith("data:"):
                new_dict[k] = f"hash:{get_base64_hash(v)}"
            else:
                new_dict[k] = _process_payload_for_hashing(v)
        return new_dict
    elif isinstance(payload, list):
        return [_process_payload_for_hashing(item) for item in payload]
    elif isinstance(payload, tuple):
        return tuple(_process_payload_for_hashing(item) for item in payload)
    return payload


@final
class LLMCacheKeyGenerator(CacheKeyGenerator[CompletionRequest[BaseModel]]):
    """Generate deterministic cache keys for LLM completion requests.

    Keys are based on:
    - Conversation messages (content and roles)
    - Whether structured output is requested
    """

    @override
    def generate_key(self, request: CompletionRequest[BaseModel]) -> str:
        """Generate a unique cache key from the request.

        Args:
            request: CompletionRequest containing conversation and config

        Returns:
            SHA-256 hash of the request parameters
        """
        # Serialize conversation to dict
        conversation_dict = request.input.to_dict()

        # Create payload for hashing
        payload = {
            "messages": conversation_dict,
            "structured_output": request.structured_output is not None,
        }

        # Replace massive base64 strings with hashes for efficient key generation
        processed_payload = _process_payload_for_hashing(payload)

        # Generate deterministic hash
        payload_str = json.dumps(processed_payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()


@final
class LLMCacheBackend[T: BaseModel](CacheBackend[CompletionResult[T]]):
    """Cache backend adapter for LLM responses.

    This adapter bridges the CachedEngine protocol with the LLMCache storage,
    using pickle serialization for automatic handling of complex types.

    The cache stores complete CompletionResult objects with images stored
    separately as content-addressable files for efficient deduplication.

    Storage strategy:
    - Conversation images are replaced with references before caching
    - Images are stored in ImageRepository by content hash
    - On retrieval, references are resolved back to base64 images
    """

    def __init__(
        self,
        cache: LLMCache[T],
        key_generator: LLMCacheKeyGenerator,
        image_repository: ImageRepository,
    ):
        """Initialize the cache backend.

        Args:
            cache: LLMCache instance for storage
            key_generator: Key generator for creating cache keys from requests
            image_repository: Repository for storing images by content hash
        """
        self.cache = cache
        self.key_generator = key_generator
        self.image_repository = image_repository

    @override
    def get(self, key: str) -> CompletionResult[T] | None:
        """Retrieve CompletionResult from cache and resolve image refs.

        Args:
            key: Cache key

        Returns:
            Cached CompletionResult with images restored, None if not found
        """
        result = self.cache.get(key)
        if result is None:
            return None

        # Resolve image references back to base64
        conversation = conversation_with_images(
            result.conversation, self.image_repository
        )

        return CompletionResult(
            output=result.output,
            conversation=conversation,
        )

    @override
    def set(self, key: str, value: CompletionResult[T]) -> bool:
        """Store CompletionResult in cache with image refs.

        Args:
            key: Cache key
            value: CompletionResult to cache

        Returns:
            True if cached successfully
        """
        # Replace base64 images with content-addressable references
        conversation = conversation_with_refs(value.conversation, self.image_repository)

        modified_value = CompletionResult(
            output=value.output,
            conversation=conversation,
        )

        return self.cache.set(key, modified_value)


def create_llm_cache_backend[T: BaseModel](
    model_name: str,
    image_repository: ImageRepository,
) -> tuple[LLMCacheBackend[T], LLMCacheKeyGenerator]:
    """Create an LLM cache backend with key generator and image repository.

    This is a convenience factory for setting up LLM caching with automatic
    image deduplication via content-addressable storage.

    Images are automatically stored separately by content hash, reducing
    cache size and enabling deduplication across requests.

    Args:
        model_name: Model name for cache directory namespacing

    Returns:
        Tuple of (cache_backend, key_generator)
    """

    cache = LLMCache[T](model_name=model_name)
    key_generator = LLMCacheKeyGenerator()

    backend = LLMCacheBackend[T](
        cache=cache,
        key_generator=key_generator,
        image_repository=image_repository,
    )

    return backend, key_generator
