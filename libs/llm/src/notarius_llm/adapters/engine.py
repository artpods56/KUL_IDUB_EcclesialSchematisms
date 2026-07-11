from typing import Any, Self, final, override

from pydantic import BaseModel
from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential

from notarius_core.ports.llm import CompletionRequest, CompletionResult
from notarius_llm.config import BackendType, ClientConfig, LLMEngineConfig
from notarius_llm.providers.factory import llm_provider_factory
from notarius_shared.constants import MAX_LLM_RETRIES
from notarius_shared.logger import get_logger

logger = get_logger(__name__)


class EngineStats(dict[str, int]):
    pass


def _should_retry_none_structured(result: CompletionResult[BaseModel]) -> bool:
    if result.structured_output_expected and result.output.structured_response is None:
        logger.warning("Structured output parsing failed, will retry")
        return True
    return False


@final
class StructuredResponseValidator:
    def is_valid(self, response: CompletionResult[BaseModel]) -> bool:
        return response.output.structured_response is not None


@final
class LLMEngine:
    def __init__(self, config: LLMEngineConfig):
        self.config = config
        self.provider = llm_provider_factory(config)
        self._stats = EngineStats(calls=0, errors=0)

    @classmethod
    def from_config(cls, config: LLMEngineConfig) -> Self:
        return cls(config=config)

    @property
    def used_backend(self) -> BackendType:
        return self.config.backend.type

    @property
    def used_model(self) -> str:
        return self.get_client_config(self.used_backend).model

    @property
    def stats(self) -> Any:
        return dict(self._stats)

    def update_client(self, client_config: ClientConfig) -> None:
        self.config.clients[self.used_backend] = client_config
        self.provider = llm_provider_factory(self.config)

    def get_client_config(
        self,
        backend_type: BackendType | None = None,
    ) -> ClientConfig:
        selected = backend_type or self.used_backend
        client_config = self.config.clients.get(selected)
        if client_config is None:
            raise ValueError(f"No client config found for backend {selected}")
        return client_config

    @override
    @retry(
        stop=stop_after_attempt(MAX_LLM_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_result(_should_retry_none_structured),
        reraise=True,
    )
    def process[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        self._stats["calls"] += 1
        try:
            response = self.provider.generate_response(
                request.input.messages,
                text_format=request.structured_output,
            )
            return CompletionResult[T](
                output=response,
                conversation=request.input,
                structured_output_expected=request.structured_output is not None,
            )
        except Exception:
            self._stats["errors"] += 1
            raise

    @override
    @retry(
        stop=stop_after_attempt(MAX_LLM_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_result(_should_retry_none_structured),
        reraise=True,
    )
    async def process_async[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        self._stats["calls"] += 1
        try:
            response = await self.provider.generate_response_async(
                request.input.messages,
                text_format=request.structured_output,
            )
            return CompletionResult[T](
                output=response,
                conversation=request.input,
                structured_output_expected=request.structured_output is not None,
            )
        except Exception:
            self._stats["errors"] += 1
            raise

