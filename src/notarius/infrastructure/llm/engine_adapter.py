"""Clean LLM Engine adapter using refactored components."""

from dataclasses import dataclass
from typing import Self, final, override, Protocol, runtime_checkable, cast, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)
from pydantic import BaseModel

from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.application.ports.outbound.engine import (
    ConfigurableEngine,
    track_stats,
    track_stats_async,
)
from notarius.domain.entities.completions import BaseProviderResponse
from notarius.domain.protocols import BaseRequest, BaseResponse

from notarius.infrastructure.llm.conversation import (
    Conversation,
)
from notarius.infrastructure.llm.providers.factory import llm_provider_factory
from notarius.schemas.configs import LLMEngineConfig
from notarius.schemas.configs.llm_model_config import ClientConfig, BackendType
from notarius.shared.constants import MAX_LLM_RETRIES


@dataclass(frozen=True)
class CompletionRequest[T: BaseModel](BaseRequest[Conversation]):
    """Configuration for a single LLM request."""

    input: Conversation
    structured_output: type[T] | None = None


@dataclass(frozen=True)
class CompletionResult[T: BaseModel](BaseResponse[BaseProviderResponse[T]]):
    """Result of an LLM completion request.

    The input automatically includes the assistant's output.
    """

    output: BaseProviderResponse[T]
    conversation: Conversation

    @property
    def updated_conversation(self) -> Conversation:
        """Get the input with the assistant's output added.

        This is useful for multi-turn conversations where you want to
        maintain the full history including the assistant's replies.
        """
        return self.conversation.add(self.output.to_message())


@runtime_checkable
class LLMCompletionEngine(Protocol):
    """Protocol for engines that support generic LLM completions."""

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


@final
class LLMEngine(
    ConfigurableEngine[
        LLMEngineConfig,
        CompletionRequest[BaseModel],
        CompletionResult[BaseModel],
    ]
):
    """Engine for interacting with LLM providers using domain types."""

    def __init__(self, config: LLMEngineConfig):
        self._init_stats()
        self.config = config
        self.provider = llm_provider_factory(config)

    def update_client(self, client_config: ClientConfig):
        self.config.clients[self.used_backend] = client_config
        self.provider = llm_provider_factory(self.config)

    @property
    def used_backend(self) -> BackendType:
        return self.config.backend.type

    @property
    def used_model(self) -> str:
        client_config = self.get_client_config(self.used_backend)
        return client_config.model

    def get_client_config(
        self, backend_type: BackendType | None = None
    ) -> ClientConfig:
        client_config = self.config.clients.get(self.used_backend or backend_type or "")
        if client_config is None:
            raise ValueError(f"No client config found for backend {self.used_backend}")
        return client_config

    @classmethod
    @override
    def from_config(cls, config: LLMEngineConfig) -> Self:
        return cls(config=config)

    @override
    @track_stats
    @retry(
        stop=stop_after_attempt(MAX_LLM_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def process[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        response = self.provider.generate_response(
            request.input.messages, text_format=request.structured_output
        )

        return CompletionResult[T](
            output=response,
            conversation=request.input,
        )

    @override
    @retry(
        stop=stop_after_attempt(MAX_LLM_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    @track_stats_async
    async def process_async[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        response = await self.provider.generate_response_async(
            request.input.messages, text_format=request.structured_output
        )

        return CompletionResult[T](
            output=response,
            conversation=request.input,
        )


# CachedLLMEngine = CachedEngine[
#     LLMEngineConfig, CompletionRequest[Any], CompletionResult[Any]
# ]


@final
class CachedLLMEngine(
    CachedEngine[
        LLMEngineConfig, CompletionRequest[BaseModel], CompletionResult[BaseModel]
    ]
):
    """Cached version of LLMEngine that preserves generic completion types."""

    @override
    def process[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        return cast(CompletionResult[T], super().process(request))

    @override
    async def process_async[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        return cast(CompletionResult[T], await super().process_async(request))
