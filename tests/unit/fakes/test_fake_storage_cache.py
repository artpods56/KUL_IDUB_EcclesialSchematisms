"""Smoke tests for fake storage and cache."""

import pytest
from PIL import Image
from pathlib import Path

from tests.fakes import FakeImageStorage, FakePDFIngestor, FakeCacheBackend, FakeCacheKeyGenerator, FakeResponseValidator
from tests.factories import OCRResponseFactory


class TestFakeImageStorage:
    """Smoke tests for FakeImageStorage."""

    def test_add_and_get_image(self):
        """Test that images can be added and retrieved."""
        storage = FakeImageStorage()
        image = Image.new("RGB", (800, 600), color="red")

        path = storage.add(image, "test.jpg")
        retrieved = storage.get(path)

        assert isinstance(retrieved, Image.Image)
        assert retrieved.mode == "RGB"

    def test_exists_returns_correct_value(self):
        """Test that exists() returns correct boolean."""
        storage = FakeImageStorage()
        image = Image.new("RGB", (100, 100))

        assert storage.exists("test.jpg") == False

        storage.add(image, "test.jpg")

        assert storage.exists("test.jpg") == True

    def test_load_image_returns_image(self):
        """Test that load_image() works."""
        storage = FakeImageStorage()
        image = Image.new("RGB", (100, 100))

        storage.add(image, "test.jpg")
        loaded = storage.load_image("test.jpg")

        assert isinstance(loaded, Image.Image)

    def test_default_image_configuration(self):
        """Test that default image is returned when image not found."""
        storage = FakeImageStorage()
        default = Image.new("RGB", (200, 200), color="blue")

        storage.configure_default_image(default)
        retrieved = storage.get(Path("nonexistent.jpg"))

        assert isinstance(retrieved, Image.Image)

    def test_tracks_operations(self):
        """Test that all operations are tracked."""
        storage = FakeImageStorage()
        image = Image.new("RGB", (100, 100))

        storage.add(image, "test.jpg")
        storage.get(Path("test.jpg"))
        storage.exists("test.jpg")

        assert len(storage.add_calls) == 1
        assert len(storage.get_calls) == 1
        assert len(storage.exists_calls) == 1

    def test_reset_clears_state(self):
        """Test that reset() clears all state."""
        storage = FakeImageStorage()
        image = Image.new("RGB", (100, 100))

        storage.add(image, "test.jpg")
        storage.reset()

        assert len(storage.images) == 0
        assert len(storage.add_calls) == 0


class TestFakeCacheBackend:
    """Smoke tests for FakeCacheBackend."""

    def test_set_and_get(self):
        """Test that values can be set and retrieved."""
        cache = FakeCacheBackend()
        response = OCRResponseFactory.build()

        cache.set("key1", response)
        retrieved = cache.get("key1")

        assert retrieved == response

    def test_get_returns_none_for_missing_key(self):
        """Test that get() returns None for missing keys."""
        cache = FakeCacheBackend()

        result = cache.get("nonexistent")

        assert result is None

    def test_delete_removes_entry(self):
        """Test that delete() removes entries."""
        cache = FakeCacheBackend()
        response = OCRResponseFactory.build()

        cache.set("key1", response)
        assert cache.delete("key1") == True
        assert cache.get("key1") is None

    def test_delete_returns_false_for_missing_key(self):
        """Test that delete() returns False for missing keys."""
        cache = FakeCacheBackend()

        assert cache.delete("nonexistent") == False

    def test_tracks_hits_and_misses(self):
        """Test that cache tracks hits and misses."""
        cache = FakeCacheBackend()
        response = OCRResponseFactory.build()

        cache.set("key1", response)

        cache.get("key1")  # hit
        cache.get("key2")  # miss
        cache.get("key1")  # hit

        assert cache.hit_count == 2
        assert cache.miss_count == 1

    def test_tracks_operations(self):
        """Test that all operations are tracked."""
        cache = FakeCacheBackend()
        response = OCRResponseFactory.build()

        cache.set("key1", response)
        cache.get("key1")
        cache.delete("key1")

        assert len(cache.set_calls) == 1
        assert len(cache.get_calls) == 1
        assert len(cache.delete_calls) == 1

    def test_len_returns_cache_size(self):
        """Test that len() returns number of cached items."""
        cache = FakeCacheBackend()
        response1 = OCRResponseFactory.build()
        response2 = OCRResponseFactory.build()

        assert len(cache) == 0

        cache.set("key1", response1)
        cache.set("key2", response2)

        assert len(cache) == 2

    def test_reset_clears_state(self):
        """Test that reset() clears all state."""
        cache = FakeCacheBackend()
        response = OCRResponseFactory.build()

        cache.set("key1", response)
        cache.get("key1")

        cache.reset()

        assert len(cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
        assert len(cache.set_calls) == 0


class TestFakeCacheKeyGenerator:
    """Smoke tests for FakeCacheKeyGenerator."""

    def test_generates_keys(self):
        """Test that key generator generates keys."""
        generator = FakeCacheKeyGenerator()
        from tests.factories import OCRRequestFactory

        request = OCRRequestFactory.build()
        key = generator.generate_key(request)

        assert isinstance(key, str)
        assert key.startswith("cache_key_")

    def test_same_request_generates_same_key(self):
        """Test that same request generates same key."""
        generator = FakeCacheKeyGenerator()
        from tests.factories import OCRRequestFactory

        request = OCRRequestFactory.build()
        key1 = generator.generate_key(request)
        key2 = generator.generate_key(request)

        assert key1 == key2


class TestFakeResponseValidator:
    """Smoke tests for FakeResponseValidator."""

    def test_always_valid_by_default(self):
        """Test that validator returns True by default."""
        validator = FakeResponseValidator()
        response = OCRResponseFactory.build()

        assert validator.is_valid(response) == True

    def test_can_be_configured_invalid(self):
        """Test that validator can be configured to return False."""
        validator = FakeResponseValidator()
        response = OCRResponseFactory.build()

        validator.configure_valid(False)

        assert validator.is_valid(response) == False

    def test_tracks_validation_calls(self):
        """Test that validator tracks all validation calls."""
        validator = FakeResponseValidator()
        response1 = OCRResponseFactory.build()
        response2 = OCRResponseFactory.build()

        validator.is_valid(response1)
        validator.is_valid(response2)

        assert len(validator.validation_calls) == 2


class TestFakePDFIngestor:
    """Smoke tests for FakePDFIngestor."""

    def test_ingest_returns_pages(self):
        """Test that ingest() returns list of (text, image) tuples."""
        ingestor = FakePDFIngestor()
        pdf_path = Path("test.pdf")

        pages = ingestor.ingest(pdf_path)

        assert isinstance(pages, list)
        assert len(pages) == 2  # Default has 2 pages
        assert all(isinstance(page, tuple) for page in pages)
        assert all(len(page) == 2 for page in pages)

    def test_ingest_with_custom_content(self):
        """Test that ingest() returns custom page content."""
        custom_content = ["Page 1", "Page 2", "Page 3"]
        ingestor = FakePDFIngestor(pages_content=custom_content)
        pdf_path = Path("test.pdf")

        pages = ingestor.ingest(pdf_path)

        assert len(pages) == 3
        for i, (text, image) in enumerate(pages):
            assert text == custom_content[i]
            assert isinstance(image, Image.Image)

    def test_tracks_ingest_calls(self):
        """Test that ingest() calls are tracked."""
        ingestor = FakePDFIngestor()
        pdf1 = Path("doc1.pdf")
        pdf2 = Path("doc2.pdf")

        ingestor.ingest(pdf1)
        ingestor.ingest(pdf2)

        assert len(ingestor.ingest_calls) == 2
        assert ingestor.ingest_calls[0] == pdf1
        assert ingestor.ingest_calls[1] == pdf2

    def test_returns_images(self):
        """Test that ingest() returns valid PIL images."""
        ingestor = FakePDFIngestor()
        pdf_path = Path("test.pdf")

        pages = ingestor.ingest(pdf_path)

        for text, image in pages:
            assert isinstance(image, Image.Image)
            assert image.size == (800, 600)
            assert image.mode == "RGB"
