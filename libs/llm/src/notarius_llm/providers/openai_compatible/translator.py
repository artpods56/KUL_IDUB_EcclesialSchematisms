from functools import singledispatch

from openai.types.responses import (
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseInputContentParam,
)
from openai.types.responses.response_input_param import Message

from notarius_core.domain.models.messages import (
    ContentPart,
    MessageContent,
    ChatMessageList,
    TextContent,
    ImageContent,
)


@singledispatch
def translate(part: ContentPart) -> ResponseInputContentParam:
    raise ValueError(f"No translator for {type(part)}")


@translate.register
def _(part: TextContent) -> ResponseInputTextParam:
    return ResponseInputTextParam(text=part.text, type=part.type)


@translate.register
def _(part: ImageContent) -> ResponseInputImageParam:
    return ResponseInputImageParam(
        image_url=part.image_url,
        detail=part.detail,
        type=part.type,
    )


def content_to_openai(content: MessageContent):
    return [translate(part) for part in content]


def messages_to_openai(
    messages: ChatMessageList,
) -> ResponseInputParam:
    return [
        Message(
            role=message.role,
            content=content_to_openai(message.content),
        )
        for message in messages
    ]
