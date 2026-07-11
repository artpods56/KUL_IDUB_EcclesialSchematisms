from abc import ABC, abstractmethod

from pydantic import BaseModel

from notarius_core.domain.models.completions import BaseProviderResponse
from notarius_core.domain.models.messages import ChatMessageList
from notarius_llm.config import ClientConfig


class LLMProvider[TClient](ABC):
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client: TClient = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> TClient:
        raise NotImplementedError

    @abstractmethod
    def generate_response[ResponseT: BaseModel](
        self,
        messages: ChatMessageList,
        text_format: type[ResponseT] | None = None,
    ) -> BaseProviderResponse[ResponseT]:
        raise NotImplementedError

    async def generate_response_async[ResponseT: BaseModel](
        self,
        messages: ChatMessageList,
        text_format: type[ResponseT] | None = None,
    ) -> BaseProviderResponse[ResponseT]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async generation"
        )

