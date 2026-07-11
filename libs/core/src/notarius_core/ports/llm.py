from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from notarius_core.domain.models.completions import BaseProviderResponse
from notarius_core.domain.models.conversation import Conversation
from notarius_core.ports.protocols import BaseRequest, BaseResponse


@dataclass(frozen=True)
class CompletionRequest[T: BaseModel](BaseRequest[Conversation]):
    input: Conversation
    structured_output: type[T] | None = None


@dataclass(frozen=True)
class CompletionResult[T: BaseModel](BaseResponse[BaseProviderResponse[T]]):
    output: BaseProviderResponse[T]
    conversation: Conversation
    structured_output_expected: bool = False

    @property
    def updated_conversation(self) -> Conversation:
        return self.conversation.add(self.output.to_message())


@runtime_checkable
class LLMCompletionEngine(Protocol):
    def process[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]: ...

    async def process_async[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]: ...

    @property
    def stats(self) -> Any: ...

