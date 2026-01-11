"""Tests for message factories."""

import pytest

from tests.factories.messages import (
    TextContentFactory,
    ImageContentFactory,
    ChatMessageFactory,
    ConversationFactory,
)
from notarius.domain.entities.messages import TextContent, ImageContent, ChatMessage
from notarius.infrastructure.llm.conversation import Conversation


class TestTextContentFactory:
    """Tests for TextContentFactory."""

    def test_build_creates_text_content(self):
        """Test that build() creates TextContent with default text."""
        content = TextContentFactory.build()

        assert isinstance(content, TextContent)
        assert content.text is not None
        assert content.type == "input_text"

    def test_build_with_custom_text(self):
        """Test that build() accepts custom text."""
        content = TextContentFactory.build(text="Custom text")

        assert content.text == "Custom text"

    def test_build_with_ocr(self):
        """Test that build_with_ocr() creates OCR-like text."""
        content = TextContentFactory.build_with_ocr()

        assert "OCR" in content.text or "Deanery" in content.text


class TestImageContentFactory:
    """Tests for ImageContentFactory."""

    def test_build_creates_image_content(self):
        """Test that build() creates ImageContent with default URL."""
        content = ImageContentFactory.build()

        assert isinstance(content, ImageContent)
        assert content.image_url is not None
        assert content.detail == "auto"
        assert content.type == "input_image"

    def test_build_with_custom_url(self):
        """Test that build() accepts custom URL."""
        url = "https://example.com/image.jpg"
        content = ImageContentFactory.build(image_url=url)

        assert content.image_url == url

    def test_build_with_url_method(self):
        """Test that build_with_url() creates content with URL."""
        url = "https://example.com/image.jpg"
        content = ImageContentFactory.build_with_url(url)

        assert content.image_url == url

    def test_build_with_detail_level(self):
        """Test that build() accepts detail level."""
        content = ImageContentFactory.build(detail="high")

        assert content.detail == "high"


class TestChatMessageFactory:
    """Tests for ChatMessageFactory."""

    def test_build_creates_message_with_defaults(self):
        """Test that build() creates ChatMessage with defaults."""
        msg = ChatMessageFactory.build()

        assert isinstance(msg, ChatMessage)
        assert msg.role == "user"
        assert len(msg.content) > 0

    def test_build_with_text_shorthand(self):
        """Test that build() accepts text shorthand."""
        msg = ChatMessageFactory.build(text="Hello")

        assert len(msg.content) == 1
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "Hello"

    def test_build_with_custom_content(self):
        """Test that build() accepts custom content list."""
        content = [
            TextContent(text="What's in this image?"),
            ImageContent(image_url="data:image/jpeg;base64,abc123")
        ]
        msg = ChatMessageFactory.build(content=content)

        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ImageContent)

    def test_build_user_message(self):
        """Test that build_user_message() creates user message."""
        msg = ChatMessageFactory.build_user_message("Hello")

        assert msg.role == "user"
        assert msg.content[0].text == "Hello"

    def test_build_assistant_message(self):
        """Test that build_assistant_message() creates assistant message."""
        msg = ChatMessageFactory.build_assistant_message("Hi there!")

        assert msg.role == "assistant"
        assert msg.content[0].text == "Hi there!"

    def test_build_system_message(self):
        """Test that build_system_message() creates system message."""
        msg = ChatMessageFactory.build_system_message("You are helpful")

        assert msg.role == "system"
        assert msg.content[0].text == "You are helpful"

    def test_build_multimodal_message(self):
        """Test that build_multimodal_message() creates message with text and image."""
        msg = ChatMessageFactory.build_multimodal_message(
            text="What's this?",
            image_url="data:image/jpeg;base64,abc"
        )

        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ImageContent)


class TestConversationFactory:
    """Tests for ConversationFactory."""

    def test_build_creates_conversation_with_defaults(self):
        """Test that build() creates Conversation with default messages."""
        conv = ConversationFactory.build()

        assert isinstance(conv, Conversation)
        assert len(conv.messages) == 2  # default message_count

    def test_build_with_message_count(self):
        """Test that build() respects message_count."""
        conv = ConversationFactory.build(message_count=5)

        assert len(conv.messages) == 5

    def test_build_with_custom_messages(self):
        """Test that build() accepts custom messages."""
        messages = [
            ChatMessageFactory.build_user_message("Hi"),
            ChatMessageFactory.build_assistant_message("Hello"),
        ]
        conv = ConversationFactory.build(messages=messages)

        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"

    def test_build_empty(self):
        """Test that build_empty() creates empty conversation."""
        conv = ConversationFactory.build_empty()

        assert len(conv.messages) == 0

    def test_build_with_system_prompt(self):
        """Test that build_with_system_prompt() creates conversation with system message."""
        conv = ConversationFactory.build_with_system_prompt("You are helpful")

        assert len(conv.messages) == 1
        assert conv.messages[0].role == "system"

    def test_build_user_assistant_exchange(self):
        """Test that build_user_assistant_exchange() creates proper exchange."""
        conv = ConversationFactory.build_user_assistant_exchange(
            user_text="Question",
            assistant_text="Answer"
        )

        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[1].role == "assistant"

    def test_build_multimodal_conversation(self):
        """Test that build_multimodal_conversation() creates conversation with multimodal message."""
        conv = ConversationFactory.build_multimodal_conversation(
            text="What's this?",
            image_url="data:image/jpeg;base64,abc"
        )

        assert len(conv.messages) == 1
        assert len(conv.messages[0].content) == 2
