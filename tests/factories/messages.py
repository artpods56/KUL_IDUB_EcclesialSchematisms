"""Factories for creating LLM message objects.

This module provides factories for creating message entities used in
LLM interactions, including text content, image content, chat messages,
and conversations.
"""

from tests.factories.base import BaseFactory
from notarius.domain.entities.messages import (
    TextContent,
    ImageContent,
    ChatMessage,
    ContentPart,
)
from notarius.infrastructure.llm.conversation import Conversation


class TextContentFactory(BaseFactory[TextContent]):
    """Factory for creating TextContent instances."""

    @classmethod
    def build(cls, text: str | None = None, **kwargs) -> TextContent:
        """Build a TextContent instance.

        Args:
            text: The text content
            **kwargs: Additional fields (type is fixed as "input_text")

        Returns:
            A new TextContent instance

        Example:
            content = TextContentFactory.build()
            content = TextContentFactory.build(text="Custom text")
        """
        return TextContent(
            text=text or "Sample text content from factory",
            **kwargs
        )

    @classmethod
    def build_with_ocr(cls, text: str | None = None) -> TextContent:
        """Build TextContent with OCR-like text.

        Args:
            text: Optional OCR text

        Returns:
            A TextContent instance with OCR-like text

        Example:
            content = TextContentFactory.build_with_ocr()
        """
        return cls.build(
            text=text or "Sample OCR text: Deanery: Test. Parish: Sample."
        )


class ImageContentFactory(BaseFactory[ImageContent]):
    """Factory for creating ImageContent instances."""

    _counter = 0

    @classmethod
    def build(
        cls,
        image_url: str | None = None,
        detail: str = "auto",
        **kwargs
    ) -> ImageContent:
        """Build an ImageContent instance.

        Args:
            image_url: URL or base64 data URI of the image
            detail: Detail level ("auto", "low", or "high")
            **kwargs: Additional fields (type is fixed as "input_image")

        Returns:
            A new ImageContent instance

        Example:
            content = ImageContentFactory.build()
            content = ImageContentFactory.build(
                image_url="data:image/jpeg;base64,/9j/4AAQ...",
                detail="high"
            )
        """
        cls._counter += 1

        return ImageContent(
            image_url=image_url or f"data:image/jpeg;base64,fake_image_{cls._counter}",
            detail=detail,  # type: ignore
            **kwargs
        )

    @classmethod
    def build_with_url(cls, url: str) -> ImageContent:
        """Build ImageContent with a specific URL.

        Args:
            url: The image URL

        Returns:
            An ImageContent instance with the specified URL

        Example:
            content = ImageContentFactory.build_with_url("https://example.com/image.jpg")
        """
        return cls.build(image_url=url)


class ChatMessageFactory(BaseFactory[ChatMessage]):
    """Factory for creating ChatMessage instances."""

    @classmethod
    def build(
        cls,
        role: str = "user",
        content: list[ContentPart] | None = None,
        text: str | None = None,
        **kwargs
    ) -> ChatMessage:
        """Build a ChatMessage instance.

        Args:
            role: Message role ("user", "assistant", "system", "developer")
            content: List of content parts (if None, creates single text content)
            text: Shorthand for creating single TextContent (ignored if content provided)
            **kwargs: Additional fields

        Returns:
            A new ChatMessage instance

        Example:
            msg = ChatMessageFactory.build()
            msg = ChatMessageFactory.build(role="assistant", text="Response")
            msg = ChatMessageFactory.build(
                role="user",
                content=[
                    TextContent(text="What's in this image?"),
                    ImageContent(image_url="data:image/...")
                ]
            )
        """
        if content is None:
            content = [TextContentFactory.build(text=text)]

        return ChatMessage(
            role=role,  # type: ignore
            content=content,
            **kwargs
        )

    @classmethod
    def build_user_message(cls, text: str | None = None) -> ChatMessage:
        """Build a user message with text content.

        Args:
            text: The message text

        Returns:
            A ChatMessage with role="user"

        Example:
            msg = ChatMessageFactory.build_user_message("Hello")
        """
        return cls.build(role="user", text=text)

    @classmethod
    def build_assistant_message(cls, text: str | None = None) -> ChatMessage:
        """Build an assistant message with text content.

        Args:
            text: The message text

        Returns:
            A ChatMessage with role="assistant"

        Example:
            msg = ChatMessageFactory.build_assistant_message("Hi there!")
        """
        return cls.build(role="assistant", text=text)

    @classmethod
    def build_system_message(cls, text: str | None = None) -> ChatMessage:
        """Build a system message with text content.

        Args:
            text: The message text

        Returns:
            A ChatMessage with role="system"

        Example:
            msg = ChatMessageFactory.build_system_message("You are a helpful assistant")
        """
        return cls.build(role="system", text=text)

    @classmethod
    def build_multimodal_message(
        cls,
        text: str | None = None,
        image_url: str | None = None,
        role: str = "user"
    ) -> ChatMessage:
        """Build a multimodal message with both text and image.

        Args:
            text: The text content
            image_url: The image URL or data URI
            role: Message role

        Returns:
            A ChatMessage with both text and image content

        Example:
            msg = ChatMessageFactory.build_multimodal_message(
                text="What's in this image?",
                image_url="data:image/jpeg;base64,..."
            )
        """
        content = [
            TextContentFactory.build(text=text),
            ImageContentFactory.build(image_url=image_url)
        ]
        return cls.build(role=role, content=content)


class ConversationFactory(BaseFactory[Conversation]):
    """Factory for creating Conversation instances."""

    @classmethod
    def build(
        cls,
        messages: list[ChatMessage] | None = None,
        message_count: int = 2,
        max_history_length: int | None = None,
        **kwargs
    ) -> Conversation:
        """Build a Conversation instance.

        Args:
            messages: List of messages (auto-generated if not provided)
            message_count: Number of messages to generate if messages not provided
            max_history_length: Maximum history length
            **kwargs: Additional fields

        Returns:
            A new Conversation instance

        Example:
            conv = ConversationFactory.build()
            conv = ConversationFactory.build(message_count=5)
            conv = ConversationFactory.build(messages=[msg1, msg2])
        """
        if messages is None:
            messages = [
                ChatMessageFactory.build_user_message(f"User message {i}")
                if i % 2 == 0
                else ChatMessageFactory.build_assistant_message(f"Assistant message {i}")
                for i in range(message_count)
            ]

        return Conversation(
            messages=tuple(messages),
            max_history_length=max_history_length,
            **kwargs
        )

    @classmethod
    def build_empty(cls) -> Conversation:
        """Build an empty conversation with no messages.

        Returns:
            A Conversation with no messages

        Example:
            conv = ConversationFactory.build_empty()
        """
        return cls.build(messages=[])

    @classmethod
    def build_with_system_prompt(cls, system_prompt: str) -> Conversation:
        """Build a conversation starting with a system message.

        Args:
            system_prompt: The system prompt text

        Returns:
            A Conversation with a system message

        Example:
            conv = ConversationFactory.build_with_system_prompt(
                "You are a helpful assistant"
            )
        """
        messages = [ChatMessageFactory.build_system_message(system_prompt)]
        return cls.build(messages=messages)

    @classmethod
    def build_user_assistant_exchange(
        cls,
        user_text: str = "User question",
        assistant_text: str = "Assistant answer"
    ) -> Conversation:
        """Build a simple user-assistant exchange.

        Args:
            user_text: User message text
            assistant_text: Assistant response text

        Returns:
            A Conversation with one user message and one assistant response

        Example:
            conv = ConversationFactory.build_user_assistant_exchange(
                user_text="What is 2+2?",
                assistant_text="4"
            )
        """
        messages = [
            ChatMessageFactory.build_user_message(user_text),
            ChatMessageFactory.build_assistant_message(assistant_text),
        ]
        return cls.build(messages=messages)

    @classmethod
    def build_multimodal_conversation(
        cls,
        text: str | None = None,
        image_url: str | None = None
    ) -> Conversation:
        """Build a conversation with a multimodal user message.

        Args:
            text: Text content
            image_url: Image URL

        Returns:
            A Conversation with a multimodal message

        Example:
            conv = ConversationFactory.build_multimodal_conversation(
                text="What's in this image?",
                image_url="data:image/..."
            )
        """
        messages = [
            ChatMessageFactory.build_multimodal_message(
                text=text,
                image_url=image_url
            )
        ]
        return cls.build(messages=messages)
