from notarius_core.domain.models.messages import (
    ChatMessage,
    ImageContent,
    TextContent,
    strip_images_from_message,
    strip_next_page_ocr_from_message,
)


def test_strip_images_from_message_keeps_text() -> None:
    message = ChatMessage(
        role="user",
        content=[TextContent(text="hello"), ImageContent(image_url="data:image/jpeg;base64,x")],
    )

    stripped = strip_images_from_message(message)

    assert stripped.content == [TextContent(text="hello")]


def test_strip_next_page_ocr_from_message_removes_tagged_section() -> None:
    message = ChatMessage(
        role="user",
        content=[TextContent(text="a<NEXT_PAGE_TEXT>remove me</NEXT_PAGE_TEXT>b")],
    )

    stripped = strip_next_page_ocr_from_message(message)

    assert stripped.content == [TextContent(text="ab")]

