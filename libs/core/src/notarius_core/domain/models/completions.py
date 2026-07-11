import abc
from dataclasses import dataclass

from pydantic import BaseModel

from notarius_core.domain.models.messages import ChatMessage, TextContent


@dataclass(frozen=True)
class BaseProviderResponse[T: BaseModel](abc.ABC):
    structured_response: T | None
    text_response: str | None

    def to_message(self) -> ChatMessage:
        """Convert the output to a ChatMessage for input history.

        Returns:
            ChatMessage with role="assistant" and text content
        """

        if self.structured_response:
            return ChatMessage(
                role="assistant",
                content=[
                    TextContent(text=self.structured_response.model_dump_json() or "")
                ],
            )
        else:
            return ChatMessage(
                role="assistant", content=[TextContent(text=self.text_response or "")]
            )
