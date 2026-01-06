from abc import ABC, abstractmethod

from pydantic import BaseModel

from notarius.domain.entities.completions import BaseProviderResponse
from notarius.domain.entities.messages import ChatMessageList
from notarius.schemas.configs.llm_model_config import ClientConfig


class LLMProvider[TClient](ABC):
    """Abstract base class for all LLM provider clients.

    This port defines the interface for LLM providers using domain types,
    allowing multiple provider implementations (OpenAI, Anthropic, Google, etc.)
    without coupling to any specific vendor's SDK types.
    """

    def __init__(self, config: ClientConfig):
        self.config: ClientConfig = config
        self.client: TClient = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> TClient:
        """Initialize the specific LLM client (e.g., OpenAI)."""
        raise NotImplementedError

    @abstractmethod
    def generate_response[ResponseT: BaseModel](
        self,
        messages: ChatMessageList,
        text_format: type[ResponseT] | None = None,
    ) -> BaseProviderResponse[ResponseT]:
        """Generate a output from the LLM given a list of messages.

        Args:
            messages: Domain message list (provider-agnostic)
            text_format: Optional Pydantic model for structured output

        Returns:
            Provider-specific output wrapped in BaseProviderResponse
        """
        raise NotImplementedError()

    async def generate_response_async[ResponseT: BaseModel](
        self,
        messages: ChatMessageList,
        text_format: type[ResponseT] | None = None,
    ) -> BaseProviderResponse[ResponseT]:
        """Async version of generate_response. Override in subclasses that support async.

        Args:
            messages: Domain message list (provider-agnostic)
            text_format: Optional Pydantic model for structured output

        Returns:
            Provider-specific output wrapped in BaseProviderResponse
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async generation"
        )
