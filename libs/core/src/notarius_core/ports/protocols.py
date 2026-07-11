from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from PIL import Image

from notarius_core.application.sequence_state import SequenceState
from notarius_core.domain.models.conversation import Conversation
from notarius_core.domain.models.dataset import BaseDataItem
from notarius_core.domain.models.messages import ChatMessage

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ItemT = TypeVar("ItemT", bound=BaseDataItem, contravariant=True)

class FileStreamProtocol(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...
    def write(self, data: bytes, /) -> int: ...
    def close(self) -> None: ...

@dataclass(frozen=True)
class BaseRequest(Generic[InputT]):
    input: InputT


@dataclass(frozen=True)
class BaseResponse(Generic[OutputT]):
    output: OutputT


class ContextProvider(Protocol[ItemT]):
    def get_context(
        self,
        items: Sequence[ItemT],
        sequence_state: SequenceState,
    ) -> dict[str, Any]: ...

    def get_context_keys(self) -> list[str]: ...


class MessageBuilder(Protocol):
    @property
    def task_name(self) -> str: ...

    def construct_template_name(self, template_name: str) -> str: ...

    def build_system_message(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> ChatMessage: ...

    def build_user_message(
        self,
        template_name: str,
        context: dict[str, Any],
        image: Image.Image | None,
    ) -> ChatMessage: ...


class ContextStrategy(Protocol):
    def initialize_state(self) -> Conversation: ...

    def prepare_state(
        self,
        state: SequenceState,
        context: dict[str, Any],
        image: Image.Image | None = None,
    ) -> SequenceState: ...

    def update_state(self, state: SequenceState) -> SequenceState: ...
