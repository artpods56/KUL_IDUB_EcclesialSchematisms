import pytest
from openai.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)

from core.models.llm.utils import messages_to_string


@pytest.fixture
def user_message() -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": "user prompt"}


@pytest.fixture
def system_message() -> ChatCompletionSystemMessageParam:
    return {"role": "system", "content": "system prompt"}


@pytest.fixture
def user_message_with_image():
    text_part: ChatCompletionContentPartTextParam = {
        "type": "text",
        "text": "user prompt",
    }
    image_part: ChatCompletionContentPartImageParam = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,base64_image"},
    }
    message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": [text_part, image_part],
    }
    return message


@pytest.fixture
def list_of_messages(
    system_message, user_message, user_message_with_image
) -> list[ChatCompletionUserMessageParam]:
    return [system_message, user_message, user_message_with_image]


class TestMessagesToString:

    def test_user_messages_to_string(
        self, user_message: ChatCompletionUserMessageParam
    ):

        parsed_message = messages_to_string([user_message])

        assert parsed_message == "user: user prompt"

    def test_system_message_to_string(
        self, system_message: ChatCompletionSystemMessageParam
    ):

        parsed_message = messages_to_string([system_message])

        assert parsed_message == "system: system prompt"

    def test_user_message_with_image_to_string(
        self, user_message_with_image: ChatCompletionUserMessageParam
    ):
        parsed_message = messages_to_string([user_message_with_image])

        assert parsed_message == "user: user prompt"

    def test_messages_to_string(self, list_of_messages):

        parsed_messages = messages_to_string(list_of_messages)

        assert isinstance(parsed_messages, str)

        prompts = ["system: system prompt", "user: user prompt", "user: user prompt"]

        for prompt in prompts:

            assert prompt in parsed_messages
