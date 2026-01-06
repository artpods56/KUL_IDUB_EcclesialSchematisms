"""Tests for strip_next_page_ocr_from_message function."""

import pytest

from notarius.domain.entities.messages import (
    ChatMessage,
    TextContent,
    ImageContent,
    strip_next_page_ocr_from_message,
)


class TestStripNextPageOcr:
    """Test suite for strip_next_page_ocr_from_message."""

    def test_strips_next_page_text_section(self):
        """Should remove NEXT_PAGE_TEXT section from user message."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<INSTRUCTION>Find the Latin source text</INSTRUCTION>

<CURRENT_PAGE__TEXT>
This is page 1 OCR text that should be KEPT
</CURRENT_PAGE__TEXT>

<NEXT_PAGE_TEXT>
This is page 2 OCR text that should be REMOVED
</NEXT_PAGE_TEXT>

<PARSED_GROUND_TRUTH>
{"entries": []}
</PARSED_GROUND_TRUTH>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # NEXT_PAGE_TEXT should be removed
        assert "<NEXT_PAGE_TEXT>" not in cleaned.content[0].text
        assert "page 2 OCR text that should be REMOVED" not in cleaned.content[0].text

        # Other sections should be preserved
        assert "<CURRENT_PAGE__TEXT>" in cleaned.content[0].text
        assert "page 1 OCR text that should be KEPT" in cleaned.content[0].text
        assert "<INSTRUCTION>" in cleaned.content[0].text
        assert "<PARSED_GROUND_TRUTH>" in cleaned.content[0].text

    def test_preserves_current_page_text(self):
        """Should keep CURRENT_PAGE_TEXT section intact."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<CURRENT_PAGE__TEXT>
Important OCR content
Multi-line text
With special chars: <>&"'
</CURRENT_PAGE__TEXT>

<NEXT_PAGE_TEXT>
Remove this
</NEXT_PAGE_TEXT>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # Current page text fully preserved
        assert "Important OCR content" in cleaned.content[0].text
        assert "Multi-line text" in cleaned.content[0].text
        assert "With special chars: <>&\"'" in cleaned.content[0].text

    def test_handles_message_without_next_page_text(self):
        """Should handle messages that don't have NEXT_PAGE_TEXT section."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<INSTRUCTION>Process this page</INSTRUCTION>

<CURRENT_PAGE__TEXT>
Only current page OCR here
</CURRENT_PAGE__TEXT>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # Content should be unchanged
        assert cleaned.content[0].text == message.content[0].text

    def test_does_not_modify_assistant_messages(self):
        """Should return assistant messages unchanged."""
        message = ChatMessage(
            role="assistant",
            content=[
                TextContent(
                    text="""<NEXT_PAGE_TEXT>
This should NOT be stripped from assistant messages
</NEXT_PAGE_TEXT>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # Assistant message unchanged
        assert cleaned.content[0].text == message.content[0].text
        assert "<NEXT_PAGE_TEXT>" in cleaned.content[0].text

    def test_does_not_modify_system_messages(self):
        """Should return system messages unchanged."""
        message = ChatMessage(
            role="system",
            content=[
                TextContent(text="""System prompt with <NEXT_PAGE_TEXT> reference""")
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        assert cleaned.content[0].text == message.content[0].text

    def test_handles_multiple_text_content_parts(self):
        """Should process all text content parts."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<NEXT_PAGE_TEXT>Remove from part 1</NEXT_PAGE_TEXT>"""
                ),
                TextContent(
                    text="""<CURRENT_PAGE__TEXT>Keep this</CURRENT_PAGE__TEXT>"""
                ),
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # First part should have NEXT_PAGE_TEXT removed
        assert "<NEXT_PAGE_TEXT>" not in cleaned.content[0].text
        assert "Remove from part 1" not in cleaned.content[0].text

        # Second part should be unchanged
        assert "<CURRENT_PAGE__TEXT>" in cleaned.content[1].text
        assert "Keep this" in cleaned.content[1].text

    def test_preserves_image_content(self):
        """Should preserve image content parts (though they should be stripped separately)."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(text="""<NEXT_PAGE_TEXT>Remove this</NEXT_PAGE_TEXT>"""),
                ImageContent(image_url="data:image/jpeg;base64,abc123"),
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # Text cleaned
        assert "<NEXT_PAGE_TEXT>" not in cleaned.content[0].text

        # Image preserved
        assert isinstance(cleaned.content[1], ImageContent)
        assert cleaned.content[1].image_url == "data:image/jpeg;base64,abc123"

    def test_cleans_up_excessive_whitespace(self):
        """Should clean up excessive newlines left after removal."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<INSTRUCTION>Test</INSTRUCTION>


<NEXT_PAGE_TEXT>
Long content
Multiple lines
To be removed
</NEXT_PAGE_TEXT>


<CURRENT_PAGE__TEXT>Content</CURRENT_PAGE__TEXT>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in cleaned.content[0].text

    def test_handles_multiline_next_page_text(self):
        """Should handle NEXT_PAGE_TEXT with multiple lines and complex content."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<CURRENT_PAGE__TEXT>
Current page content
</CURRENT_PAGE__TEXT>

<NEXT_PAGE_TEXT>
Line 1 of next page
Line 2 with special chars: <>{}[]
Line 3 with numbers: 12345

Empty line above
</NEXT_PAGE_TEXT>

<PARSED_GROUND_TRUTH>
{"key": "value"}
</PARSED_GROUND_TRUTH>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # All NEXT_PAGE_TEXT content removed
        assert "<NEXT_PAGE_TEXT>" not in cleaned.content[0].text
        assert "Line 1 of next page" not in cleaned.content[0].text
        assert "special chars: <>{}[]" not in cleaned.content[0].text

        # Other sections preserved
        assert "<CURRENT_PAGE__TEXT>" in cleaned.content[0].text
        assert "<PARSED_GROUND_TRUTH>" in cleaned.content[0].text

    def test_returns_new_message_instance(self):
        """Should return a new ChatMessage instance, not modify the original."""
        original = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<NEXT_PAGE_TEXT>Original content</NEXT_PAGE_TEXT>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(original)

        # Original unchanged
        assert "<NEXT_PAGE_TEXT>" in original.content[0].text
        assert "Original content" in original.content[0].text

        # Cleaned is different
        assert "<NEXT_PAGE_TEXT>" not in cleaned.content[0].text
        assert "Original content" not in cleaned.content[0].text

        # Different instances
        assert cleaned is not original
        assert cleaned.content[0] is not original.content[0]

    def test_handles_edge_case_empty_next_page_text(self):
        """Should handle NEXT_PAGE_TEXT with empty content."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<CURRENT_PAGE__TEXT>Content</CURRENT_PAGE__TEXT>

<NEXT_PAGE_TEXT>
</NEXT_PAGE_TEXT>

<INSTRUCTION>Test</INSTRUCTION>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        assert "<NEXT_PAGE_TEXT>" not in cleaned.content[0].text
        assert "<CURRENT_PAGE__TEXT>" in cleaned.content[0].text

    def test_handles_leading_whitespace_before_tag(self):
        """Should handle whitespace before NEXT_PAGE_TEXT tag."""
        message = ChatMessage(
            role="user",
            content=[
                TextContent(
                    text="""<CURRENT_PAGE__TEXT>Content</CURRENT_PAGE__TEXT>

    <NEXT_PAGE_TEXT>
    Remove this with leading spaces
    </NEXT_PAGE_TEXT>

<INSTRUCTION>Test</INSTRUCTION>"""
                )
            ],
        )

        cleaned = strip_next_page_ocr_from_message(message)

        # Should remove the tag and its leading whitespace
        assert "<NEXT_PAGE_TEXT>" not in cleaned.content[0].text
        assert "Remove this with leading spaces" not in cleaned.content[0].text
