"""Diagnostic tests to identify cache key issues in LLM OCR use case.

This test simulates the exact flow of add_llm_ocr_to_dataset.py
to verify cache key determinism.
"""

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Never

import pytest
from PIL import Image

from notarius.domain.entities.messages import ChatMessage, TextContent, ImageContent
from notarius.infrastructure.cache.backends.llm import LLMCacheKeyGenerator
from notarius.infrastructure.llm.conversation import Conversation
from notarius.infrastructure.llm.engine_adapter import CompletionRequest
from notarius.infrastructure.llm.prompt_manager import Jinja2PromptRenderer
from notarius.infrastructure.llm.utils import (
    construct_text_message,
    construct_image_message,
    encode_image_to_base64,
)


class TestLLMOcrCacheKeyDiagnostic:
    """Diagnostic tests for LLM OCR cache key generation."""

    @pytest.fixture
    def prompt_renderer(self) -> Jinja2PromptRenderer:
        """Create prompt renderer."""
        return Jinja2PromptRenderer()

    @pytest.fixture
    def key_generator(self) -> LLMCacheKeyGenerator:
        """Create key generator."""
        return LLMCacheKeyGenerator()

    @pytest.fixture
    def test_image(self, tmp_path: Path) -> Path:
        """Create a test image file."""
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test_image.png"
        img.save(img_path, format="PNG")
        return img_path

    def test_cache_key_components(
        self,
        prompt_renderer: Jinja2PromptRenderer,
        key_generator: LLMCacheKeyGenerator,
        test_image: Path,
    ):
        """Show all components that affect the cache key."""
        # Render prompts exactly like the use case does
        system_prompt = prompt_renderer.render_prompt(
            template_name="tasks/ocr/system.j2", context={}
        )
        user_prompt = prompt_renderer.render_prompt(
            template_name="tasks/ocr/user.j2", context={}
        )

        # Load and encode image exactly like the use case does
        image = Image.open(test_image).convert("RGB")
        base64_image = encode_image_to_base64(image)

        # Construct messages exactly like the use case does
        system_message = construct_text_message(text=system_prompt, role="system")
        user_message = construct_image_message(
            pil_image=image, text=user_prompt, role="user"
        )

        # Create conversation
        conversation = Conversation().add(system_message).add(user_message)

        # Create request
        request = CompletionRequest[Never](
            input=conversation,
            structured_output=None,
        )

        # Generate key
        cache_key = key_generator.generate_key(request)

        # Show what's being hashed
        conv_dict = conversation.to_dict()
        payload = {
            "messages": conv_dict,
            "structured_output": request.structured_output is not None,
        }

        print("\n" + "=" * 80)
        print("CACHE KEY DIAGNOSTIC")
        print("=" * 80)
        print(f"\nCache key: {cache_key}")
        print(f"\nSystem prompt length: {len(system_prompt)} chars")
        print(f"System prompt hash: {hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}")
        print(f"\nUser prompt length: {len(user_prompt)} chars")
        print(f"User prompt hash: {hashlib.sha256(user_prompt.encode()).hexdigest()[:16]}")
        print(f"\nBase64 image length: {len(base64_image)} chars")
        print(f"Base64 image hash: {hashlib.sha256(base64_image.encode()).hexdigest()[:16]}")
        print(f"\nmax_history_length: {conv_dict.get('max_history_length')}")
        print(f"structured_output is not None: {request.structured_output is not None}")
        print(f"\nNumber of messages: {len(conv_dict['messages'])}")
        print("=" * 80)

    def test_cache_key_determinism_same_image_file(
        self,
        prompt_renderer: Jinja2PromptRenderer,
        key_generator: LLMCacheKeyGenerator,
        test_image: Path,
    ):
        """Test that loading the same image file twice produces the same key."""
        keys = []

        for run in range(2):
            # Simulate fresh load like a new Dagster run would
            system_prompt = prompt_renderer.render_prompt(
                template_name="tasks/ocr/system.j2", context={}
            )
            user_prompt = prompt_renderer.render_prompt(
                template_name="tasks/ocr/user.j2", context={}
            )

            # Fresh image load
            image = Image.open(test_image).convert("RGB")

            system_message = construct_text_message(text=system_prompt, role="system")
            user_message = construct_image_message(
                pil_image=image, text=user_prompt, role="user"
            )

            conversation = Conversation().add(system_message).add(user_message)
            request = CompletionRequest[Never](
                input=conversation,
                structured_output=None,
            )

            key = key_generator.generate_key(request)
            keys.append(key)
            image.close()

        assert keys[0] == keys[1], (
            f"Cache keys differ across runs!\n"
            f"Run 1: {keys[0]}\n"
            f"Run 2: {keys[1]}"
        )

    def test_prompt_whitespace_sensitivity(
        self,
        key_generator: LLMCacheKeyGenerator,
    ):
        """Test how sensitive the key is to prompt whitespace."""
        prompt1 = "Hello world"
        prompt2 = "Hello world "  # trailing space
        prompt3 = "Hello  world"  # double space

        keys = []
        for prompt in [prompt1, prompt2, prompt3]:
            conv = Conversation.from_messages([
                ChatMessage(role="user", content=[TextContent(text=prompt)])
            ])
            request = CompletionRequest[Never](input=conv, structured_output=None)
            keys.append(key_generator.generate_key(request))

        print("\n" + "=" * 80)
        print("WHITESPACE SENSITIVITY TEST")
        print("=" * 80)
        print(f"'Hello world':   {keys[0][:16]}...")
        print(f"'Hello world ':  {keys[1][:16]}...")
        print(f"'Hello  world':  {keys[2][:16]}...")
        print("=" * 80)

        # All should be different
        assert len(set(keys)) == 3, "Different whitespace should produce different keys"

    def test_conversation_max_history_length_sensitivity(
        self,
        key_generator: LLMCacheKeyGenerator,
    ):
        """Test if max_history_length affects the cache key."""
        msg = ChatMessage(role="user", content=[TextContent(text="Hello")])

        conv1 = Conversation(messages=(msg,), max_history_length=None)
        conv2 = Conversation(messages=(msg,), max_history_length=10)

        request1 = CompletionRequest[Never](input=conv1, structured_output=None)
        request2 = CompletionRequest[Never](input=conv2, structured_output=None)

        key1 = key_generator.generate_key(request1)
        key2 = key_generator.generate_key(request2)

        print("\n" + "=" * 80)
        print("MAX_HISTORY_LENGTH SENSITIVITY TEST")
        print("=" * 80)
        print(f"max_history_length=None: {key1[:16]}...")
        print(f"max_history_length=10:   {key2[:16]}...")
        print("=" * 80)

        assert key1 != key2, (
            "max_history_length DOES affect cache key! "
            "If this value changes between runs, all cache entries become invalid."
        )

    def test_image_detail_sensitivity(
        self,
        key_generator: LLMCacheKeyGenerator,
    ):
        """Test if image detail level affects the cache key."""
        base64_img = "ABC123"

        conv1 = Conversation.from_messages([
            ChatMessage(role="user", content=[
                ImageContent(image_url=f"data:image/jpeg;base64,{base64_img}", detail="auto")
            ])
        ])
        conv2 = Conversation.from_messages([
            ChatMessage(role="user", content=[
                ImageContent(image_url=f"data:image/jpeg;base64,{base64_img}", detail="high")
            ])
        ])

        request1 = CompletionRequest[Never](input=conv1, structured_output=None)
        request2 = CompletionRequest[Never](input=conv2, structured_output=None)

        key1 = key_generator.generate_key(request1)
        key2 = key_generator.generate_key(request2)

        print("\n" + "=" * 80)
        print("IMAGE DETAIL SENSITIVITY TEST")
        print("=" * 80)
        print(f"detail='auto': {key1[:16]}...")
        print(f"detail='high': {key2[:16]}...")
        print("=" * 80)

        assert key1 != key2, (
            "Image detail level DOES affect cache key! "
            "If detail changes between runs, cache entries become invalid."
        )
