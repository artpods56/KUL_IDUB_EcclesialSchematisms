"""Fake implementations for testing.

This package provides realistic fake implementations of engines, storage, and cache
that replace mocks. Fakes implement the actual interfaces and provide:
- Real behavior without external dependencies
- Call tracking for assertions
- Error simulation capabilities
- Configurable responses

Philosophy:
- Fakes are REAL implementations, not mocks
- Use fakes for anything that would call external resources
- Fakes track all interactions for testing
- Fakes support error scenarios

Example:
    from tests.fakes import FakeOCREngine, FakeImageStorage

    # Use in tests
    engine = FakeOCREngine(default_text="Sample OCR")
    storage = FakeImageStorage()

    # Configure behavior
    engine.should_fail = True
    engine.configure_response(image, custom_response)

    # Assert on calls
    assert len(engine.call_history) == 5
"""

from tests.fakes.engines import FakeOCREngine, FakeLMv3Engine, FakeLLMEngine
from tests.fakes.storage import FakeImageStorage, FakePDFIngestor
from tests.fakes.cache import FakeCacheBackend, FakeCacheKeyGenerator, FakeResponseValidator

__all__ = [
    # Engines
    "FakeOCREngine",
    "FakeLMv3Engine",
    "FakeLLMEngine",
    # Storage
    "FakeImageStorage",
    "FakePDFIngestor",
    # Cache
    "FakeCacheBackend",
    "FakeCacheKeyGenerator",
    "FakeResponseValidator",
]
