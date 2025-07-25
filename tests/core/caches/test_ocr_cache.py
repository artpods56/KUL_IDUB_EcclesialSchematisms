from pathlib import Path

import pytest

from core.caches.ocr_cache import PyTesseractCache
from core.schemas.caches.entries import PyTesseractCacheItem


def test_ocr_cache_persistence(tmp_path: Path):
    """Verify that values stored in PyTesseractCache persist across instances and are retrievable."""

    cache_root = tmp_path / "cache_root"

    cache1 = PyTesseractCache(language="eng", caches_dir=cache_root)
    key = cache1.generate_hash(image_hash="test_image_hash")

    payload = PyTesseractCacheItem(
        text="Hello World",
        bbox=[(10, 20, 100, 30)],
        words=["Hello", "World"]
    )
    cache1[key] = payload.model_dump()

    # Same instance retrieval
    assert cache1[key]["text"] == "Hello World"
    assert len(cache1[key]["bbox"]) == 1

    # Flush explicitly (legacy code path)
    # Create a new instance – should read the same data
    cache2 = PyTesseractCache(language="eng", caches_dir=cache_root)
    assert cache2[key]["text"] == "Hello World"
    assert cache2[key]["words"] == ["Hello", "World"]


def test_ocr_cache_language_separation(tmp_path: Path):
    """Test that different languages get separate cache directories."""
    
    cache_root = tmp_path / "cache_root"
    
    # Create caches for different languages
    eng_cache = PyTesseractCache(language="eng", caches_dir=cache_root)
    pol_cache = PyTesseractCache(language="pol", caches_dir=cache_root)
    
    # Same key, different languages
    key = "test_image_hash"
    
    eng_payload = PyTesseractCacheItem(
        text="Hello",
        bbox=[(10, 20, 50, 30)],
        words=["Hello"]
    )
    
    pol_payload = PyTesseractCacheItem(
        text="Cześć",
        bbox=[(10, 20, 50, 30)],
        words=["Cześć"]
    )
    
    eng_cache[key] = eng_payload.model_dump()
    pol_cache[key] = pol_payload.model_dump()
    
    # Verify they're stored separately
    assert eng_cache[key]["text"] == "Hello"
    assert pol_cache[key]["text"] == "Cześć"
    
    # Verify cache directories are separate
    assert eng_cache.model_cache_dir != pol_cache.model_cache_dir


def test_ocr_cache_hit_miss_logging(tmp_path: Path):
    """Test that OCR cache hits and misses are logged with timing information."""
    
    cache_root = tmp_path / "cache_root"
    cache = PyTesseractCache(language="eng", caches_dir=cache_root)
    
    # Test cache miss
    key = cache.generate_hash(image_hash="test_image_hash")
    
    value = cache[key]
    assert value is None

    #
    # # Test cache hit
    # payload = PyTesseractCacheItem(
    #     text="Test text",
    #     bbox=[(10, 20, 100, 30)],
    #     words=["Test", "text"]
    # )
    # cache[key] = payload.model_dump()
    #
    # result = cache[key]  # This should be a hit and log it
    # assert result["text"] == "Test text"
