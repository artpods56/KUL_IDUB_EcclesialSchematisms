from typing import Any

from notarius_llm.config import BackendType, LLMEngineConfig
from notarius_llm.ports import LLMProvider
from notarius_llm.providers.openai_compatible.adapter import OpenAICompatibleProvider

PROVIDER_MAP: dict[BackendType, type[LLMProvider[Any]]] = {
    "openai": OpenAICompatibleProvider,
    "lm_studio": OpenAICompatibleProvider,
    "openrouter": OpenAICompatibleProvider,
    "llama": OpenAICompatibleProvider,
}


def llm_provider_factory(config: LLMEngineConfig) -> LLMProvider[Any]:
    backend_type = config.backend.type
    client_config = config.clients.get(backend_type)
    if client_config is None:
        raise ValueError(f"No provider configuration found for backend: {backend_type}")
    provider_class = PROVIDER_MAP.get(backend_type)
    if provider_class is None:
        raise ValueError(f"Unsupported backend type: {backend_type}")
    return provider_class(client_config)

