"""Domain entities for LLM messages.

These domain types are provider-agnostic and represent the core message
structure used throughout the application. Provider-specific adapters
translate between these domain types and their native formats.

Supports both simple text messages and multimodal messages (text + images).
"""

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TextContent:
    """Text content part of a multimodal message."""

    text: str
    type: Literal["input_text"] = "input_text"


@dataclass(frozen=True)
class ImageContent:
    """Image content part of a multimodal message."""

    image_url: str
    detail: Literal["auto", "low", "high"] = "auto"
    type: Literal["input_image"] = "input_image"


# Union of all content types
ContentPart = TextContent | ImageContent

MessageContent = list[ContentPart]


@dataclass(frozen=True)
class ChatMessage:
    """A message in a input.

    Supports both simple text messages and multimodal messages with images.

    Attributes:
        role: The role of the message sender
        content: Either a simple text string or a list of content parts (text, images)

    Examples:
        # Simple text message
        ChatMessage(role="user", content="Hello!")

        # Multimodal message with text and image
        ChatMessage(
            role="user",
            content=[
                TextContent(text="What's in this image?"),
                ImageContent(image_url="data:image/jpeg;base64,...")
            ]
        )
    """

    role: Literal["user", "system", "developer", "assistant"]
    content: MessageContent


ChatMessageList = list[ChatMessage] | tuple[ChatMessage, ...]


def strip_images_from_message(message: ChatMessage) -> ChatMessage:
    """Create a new message with image content removed.

    Useful for maintaining conversation history without accumulating
    large base64 image payloads.

    Args:
        message: Original message potentially containing images

    Returns:
        New ChatMessage with only text content preserved
    """
    text_only_content = [
        part for part in message.content if isinstance(part, TextContent)
    ]
    return ChatMessage(role=message.role, content=text_only_content)


def strip_next_page_ocr_from_message(message: ChatMessage) -> ChatMessage:
    """Create a new message with NEXT_PAGE_TEXT sections removed from text content.

    When processing pages sequentially, each user message includes:
    - CURRENT_PAGE_TEXT: OCR of the current page (useful context to keep)
    - NEXT_PAGE_TEXT: OCR lookahead for the next page (becomes redundant)

    When page N+1 is processed, its CURRENT_PAGE_TEXT contains what was
    NEXT_PAGE_TEXT in page N's message. To avoid duplication and reduce
    context size, we strip NEXT_PAGE_TEXT from historical messages while
    preserving CURRENT_PAGE_TEXT as efficient "memory" of previous pages.

    This allows the LLM to reference content from all previously processed
    pages via their OCR text, without keeping expensive images or duplicate text.

    Context management strategy:
    - Keep CURRENT_PAGE_TEXT: Provides compressed view of what was on each page
    - Strip NEXT_PAGE_TEXT: Prevents duplication (becomes next CURRENT_PAGE_TEXT)
    - Keep assistant responses: Structured outputs with extracted context
    - Strip images: Reduces payload size

    Args:
        message: Original message potentially containing NEXT_PAGE_TEXT sections

    Returns:
        New ChatMessage with NEXT_PAGE_TEXT sections removed from text content
    """
    if message.role != "user":
        return message

    cleaned_content = []
    for part in message.content:
        if isinstance(part, TextContent):
            # Remove NEXT_PAGE_TEXT section using regex on rendered template
            # Note: [\s\S] matches any character (whitespace or non-whitespace)
            # more explicitly than . with DOTALL flag
            cleaned_text = re.sub(
                r"\s*<NEXT_PAGE_TEXT>[\s\S]*?</NEXT_PAGE_TEXT>",
                "",
                part.text,
            )
            # Clean up excessive whitespace left behind
            cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
            cleaned_content.append(TextContent(text=cleaned_text))
        else:
            # Keep non-text content as-is (though images should be stripped separately)
            cleaned_content.append(part)

    return ChatMessage(role=message.role, content=cleaned_content)
