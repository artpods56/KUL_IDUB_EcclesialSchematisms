"""Tests for image encoding determinism.

This test suite verifies that encoding the same image multiple times
produces identical base64 strings, which is critical for cache key
determinism.

If these tests fail, it explains why cache misses occur for previously
cached images.
"""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

from notarius.infrastructure.llm.utils import encode_image_to_base64
from notarius.infrastructure.cache.backends.llm import LLMCacheKeyGenerator
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import CompletionRequest
from notarius.domain.entities.messages import ChatMessage, TextContent, ImageContent


class TestImageEncodingDeterminism:
    """Test that image encoding is deterministic."""

    def test_same_pil_image_object_same_base64(self):
        """Test that encoding the same PIL image object twice gives same result."""
        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")

        encoded1 = encode_image_to_base64(img)
        encoded2 = encode_image_to_base64(img)

        assert encoded1 == encoded2, "Same PIL image object produced different base64!"

    def test_recreated_identical_image_same_base64(self):
        """Test that recreating an identical image gives same base64."""
        img1 = Image.new("RGB", (100, 100), color="blue")
        img2 = Image.new("RGB", (100, 100), color="blue")

        encoded1 = encode_image_to_base64(img1)
        encoded2 = encode_image_to_base64(img2)

        assert encoded1 == encoded2, "Identical images produced different base64!"

    def test_image_from_numpy_array_deterministic(self):
        """Test that images from numpy arrays encode deterministically."""
        # Create deterministic numpy array
        np.random.seed(42)
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        img1 = Image.fromarray(arr, mode="RGB")
        img2 = Image.fromarray(arr.copy(), mode="RGB")

        encoded1 = encode_image_to_base64(img1)
        encoded2 = encode_image_to_base64(img2)

        assert encoded1 == encoded2, "Images from same array produced different base64!"

    def test_image_load_from_bytes_deterministic(self):
        """Test that loading an image from bytes is deterministic."""
        # Create image and save to bytes
        original = Image.new("RGB", (50, 50), color="green")
        buffer = BytesIO()
        original.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        # Load twice from same bytes
        img1 = Image.open(BytesIO(png_bytes))
        img2 = Image.open(BytesIO(png_bytes))

        encoded1 = encode_image_to_base64(img1)
        encoded2 = encode_image_to_base64(img2)

        assert encoded1 == encoded2, "Images from same bytes produced different base64!"

    def test_rgb_vs_rgba_conversion_deterministic(self):
        """Test that RGBA to RGB conversion is deterministic."""
        # Create RGBA image
        rgba_img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 255))

        # encode_image_to_base64 converts to RGB internally
        encoded1 = encode_image_to_base64(rgba_img)

        # Create another RGBA image
        rgba_img2 = Image.new("RGBA", (50, 50), color=(255, 0, 0, 255))
        encoded2 = encode_image_to_base64(rgba_img2)

        assert encoded1 == encoded2, "RGBA images produced different base64 after conversion!"

    def test_grayscale_to_rgb_conversion_deterministic(self):
        """Test that grayscale to RGB conversion is deterministic."""
        gray_img1 = Image.new("L", (50, 50), color=128)
        gray_img2 = Image.new("L", (50, 50), color=128)

        # Convert to RGB first (like the use case does)
        rgb1 = gray_img1.convert("RGB")
        rgb2 = gray_img2.convert("RGB")

        encoded1 = encode_image_to_base64(rgb1)
        encoded2 = encode_image_to_base64(rgb2)

        assert encoded1 == encoded2, "Grayscale images produced different base64!"


class TestCacheKeyWithImages:
    """Test cache key generation with images."""

    def test_same_image_same_cache_key(self):
        """Test that identical images produce identical cache keys."""
        key_gen = LLMCacheKeyGenerator()

        # Create two identical images
        img1 = Image.new("RGB", (50, 50), color="purple")
        img2 = Image.new("RGB", (50, 50), color="purple")

        base64_1 = encode_image_to_base64(img1)
        base64_2 = encode_image_to_base64(img2)

        conv1 = Conversation.from_messages([
            ChatMessage(role="user", content=[
                TextContent(text="Describe this"),
                ImageContent(image_url=f"data:image/jpeg;base64,{base64_1}"),
            ])
        ])

        conv2 = Conversation.from_messages([
            ChatMessage(role="user", content=[
                TextContent(text="Describe this"),
                ImageContent(image_url=f"data:image/jpeg;base64,{base64_2}"),
            ])
        ])

        req1 = CompletionRequest(input=conv1, structured_output=None)
        req2 = CompletionRequest(input=conv2, structured_output=None)

        key1 = key_gen.generate_key(req1)
        key2 = key_gen.generate_key(req2)

        assert key1 == key2, "Identical images produced different cache keys!"

    def test_different_images_different_cache_keys(self):
        """Test that different images produce different cache keys."""
        key_gen = LLMCacheKeyGenerator()

        img1 = Image.new("RGB", (50, 50), color="red")
        img2 = Image.new("RGB", (50, 50), color="blue")

        base64_1 = encode_image_to_base64(img1)
        base64_2 = encode_image_to_base64(img2)

        conv1 = Conversation.from_messages([
            ChatMessage(role="user", content=[
                TextContent(text="Describe this"),
                ImageContent(image_url=f"data:image/jpeg;base64,{base64_1}"),
            ])
        ])

        conv2 = Conversation.from_messages([
            ChatMessage(role="user", content=[
                TextContent(text="Describe this"),
                ImageContent(image_url=f"data:image/jpeg;base64,{base64_2}"),
            ])
        ])

        req1 = CompletionRequest(input=conv1, structured_output=None)
        req2 = CompletionRequest(input=conv2, structured_output=None)

        key1 = key_gen.generate_key(req1)
        key2 = key_gen.generate_key(req2)

        assert key1 != key2, "Different images should produce different cache keys!"


class TestRealWorldImageScenario:
    """Test scenarios that match real-world usage."""

    def test_simulate_multiple_runs_same_image(self, tmp_path: Path):
        """Simulate loading same image file in multiple runs."""
        # Create and save a test image
        img_path = tmp_path / "test_image.png"
        original = Image.new("RGB", (100, 100), color="orange")
        original.save(img_path, format="PNG")

        # Simulate first run
        img_run1 = Image.open(img_path).convert("RGB")
        base64_run1 = encode_image_to_base64(img_run1)

        # Close and reopen (simulating new process)
        img_run1.close()

        # Simulate second run
        img_run2 = Image.open(img_path).convert("RGB")
        base64_run2 = encode_image_to_base64(img_run2)

        assert base64_run1 == base64_run2, (
            "Same image file produced different base64 in different 'runs'! "
            "This would cause cache misses."
        )

    def test_cache_key_across_simulated_runs(self, tmp_path: Path):
        """Test that cache keys match across simulated runs."""
        key_gen = LLMCacheKeyGenerator()

        # Create and save test image
        img_path = tmp_path / "test_image.png"
        original = Image.new("RGB", (100, 100), color="cyan")
        original.save(img_path, format="PNG")

        # First "run"
        img1 = Image.open(img_path).convert("RGB")
        base64_1 = encode_image_to_base64(img1)
        conv1 = Conversation.from_messages([
            ChatMessage(role="system", content=[TextContent(text="You are helpful")]),
            ChatMessage(role="user", content=[
                TextContent(text="OCR this"),
                ImageContent(image_url=f"data:image/jpeg;base64,{base64_1}"),
            ])
        ])
        req1 = CompletionRequest(input=conv1, structured_output=None)
        key1 = key_gen.generate_key(req1)
        img1.close()

        # Second "run" (fresh load)
        img2 = Image.open(img_path).convert("RGB")
        base64_2 = encode_image_to_base64(img2)
        conv2 = Conversation.from_messages([
            ChatMessage(role="system", content=[TextContent(text="You are helpful")]),
            ChatMessage(role="user", content=[
                TextContent(text="OCR this"),
                ImageContent(image_url=f"data:image/jpeg;base64,{base64_2}"),
            ])
        ])
        req2 = CompletionRequest(input=conv2, structured_output=None)
        key2 = key_gen.generate_key(req2)
        img2.close()

        assert key1 == key2, (
            f"Cache keys differ across runs!\n"
            f"Run 1 key: {key1[:16]}...\n"
            f"Run 2 key: {key2[:16]}...\n"
            f"Base64 match: {base64_1 == base64_2}"
        )


class TestJPEGEncodingConsistency:
    """Test JPEG encoding consistency which is used in encode_image_to_base64."""

    def test_jpeg_encoding_same_quality_deterministic(self):
        """Test that JPEG encoding with default quality is deterministic."""
        img = Image.new("RGB", (100, 100), color="yellow")

        # Encode twice
        buf1 = BytesIO()
        img.save(buf1, format="JPEG")
        bytes1 = buf1.getvalue()

        buf2 = BytesIO()
        img.save(buf2, format="JPEG")
        bytes2 = buf2.getvalue()

        assert bytes1 == bytes2, "JPEG encoding is not deterministic!"

    def test_encode_image_to_base64_uses_jpeg(self):
        """Verify that encode_image_to_base64 produces JPEG output."""
        img = Image.new("RGB", (50, 50), color="white")
        base64_str = encode_image_to_base64(img)

        # Decode and check format
        import base64
        decoded = base64.b64decode(base64_str)

        # JPEG magic bytes
        assert decoded[:2] == b'\xff\xd8', "Output is not JPEG format!"
