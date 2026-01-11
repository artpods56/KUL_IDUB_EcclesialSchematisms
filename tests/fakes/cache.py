"""Fake cache implementations for testing.

This module provides in-memory cache fakes that replace disk/Redis caching.
"""

from typing import Generic, TypeVar
from notarius.domain.protocols import BaseResponse

ResponseT = TypeVar("ResponseT", bound=BaseResponse)


class FakeCacheBackend(Generic[ResponseT]):
    """Fake in-memory cache backend for testing without disk/Redis.

    This fake cache:
    - Implements the CacheBackend[ResponseT] protocol
    - Stores responses in memory (dict)
    - Tracks hits, misses, and operations
    - No actual disk or Redis access

    Example:
        # Basic usage
        cache = FakeCacheBackend[OCRResponse]()
        cache.set("key1", response)
        cached = cache.get("key1")
        assert cached == response

        # Hit/miss tracking
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        assert cache.hit_count == 1
        assert cache.miss_count == 1

        # Operation tracking
        assert len(cache.set_calls) == 1
        assert len(cache.get_calls) == 2
    """

    def __init__(self):
        """Initialize fake cache backend with in-memory dict."""
        self.cache: dict[str, ResponseT] = {}
        self.get_calls: list[str] = []
        self.set_calls: list[tuple[str, ResponseT]] = []
        self.delete_calls: list[str] = []
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> ResponseT | None:
        """Retrieve cached response by key.

        Args:
            key: Cache key

        Returns:
            Cached response or None if not found
        """
        self.get_calls.append(key)

        if key in self.cache:
            self.hit_count += 1
            return self.cache[key]
        else:
            self.miss_count += 1
            return None

    def set(self, key: str, value: ResponseT) -> bool:
        """Store response in cache.

        Args:
            key: Cache key
            value: Response to cache

        Returns:
            True (always succeeds)
        """
        self.set_calls.append((key, value))
        self.cache[key] = value
        return True

    def delete(self, key: str) -> bool:
        """Delete a cached entry by key.

        Args:
            key: Cache key to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
        self.delete_calls.append(key)

        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def reset(self) -> None:
        """Reset cache and all tracking."""
        self.cache.clear()
        self.get_calls.clear()
        self.set_calls.clear()
        self.delete_calls.clear()
        self.hit_count = 0
        self.miss_count = 0

    def __len__(self) -> int:
        """Get number of cached items."""
        return len(self.cache)


class FakeCacheKeyGenerator:
    """Fake cache key generator for testing.

    This generates simple keys based on request data.

    Example:
        generator = FakeCacheKeyGenerator()
        key = generator.generate_key(request)
        # key will be based on request hash
    """

    def generate_key(self, request) -> str:
        """Generate a cache key from request.

        Args:
            request: The request object

        Returns:
            Simple cache key based on request hash
        """
        # Simple implementation - just use hash of request
        # In real tests, you can make this more sophisticated
        return f"cache_key_{hash(str(request))}"


class FakeResponseValidator(Generic[ResponseT]):
    """Fake response validator for testing.

    This validator can be configured to accept/reject responses.

    Example:
        validator = FakeResponseValidator()
        validator.configure_valid(True)  # Accept all
        assert validator.is_valid(response) == True

        validator.configure_valid(False)  # Reject all
        assert validator.is_valid(response) == False
    """

    def __init__(self, always_valid: bool = True):
        """Initialize validator.

        Args:
            always_valid: Whether to always return valid=True
        """
        self.always_valid = always_valid
        self.validation_calls: list[ResponseT] = []

    def is_valid(self, response: ResponseT) -> bool:
        """Check if response is valid.

        Args:
            response: Response to validate

        Returns:
            True if valid, False otherwise
        """
        self.validation_calls.append(response)
        return self.always_valid

    def configure_valid(self, valid: bool) -> None:
        """Configure validation result.

        Args:
            valid: Whether to return valid=True or False
        """
        self.always_valid = valid

    def reset(self) -> None:
        """Reset validation call tracking."""
        self.validation_calls.clear()
