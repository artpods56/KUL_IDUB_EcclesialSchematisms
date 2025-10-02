# core/models/llm/factory.py

from omegaconf import DictConfig
from openai import OpenAI

from core.caches.llm_cache import LLMCache
from core.models.llm.providers import LLMProvider, OpenAIProvider

PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "lm_studio": OpenAIProvider,
    "openrouter": OpenAIProvider,
    "llama": OpenAIProvider,
}

LLMClient = OpenAI


def llm_provider_factory(config: DictConfig) -> tuple[LLMProvider[LLMClient], LLMCache]:
    """
    Factory function to create an instance of an LLM provider.

    Args:
        config (DictConfig): Config dictionary to instantiate an the LLM model.

    Returns:
        An instance of a class that inherits from LLMProvider.
    """
    api_type = config.predictor.get("api_type", "openai")
    provider_config = config.interfaces.get(api_type)

    if provider_config is None:
        raise ValueError(f"No provider configuration found for api_type: {api_type}")

    provider_class = PROVIDER_MAP.get(api_type.lower())
    if not provider_class:
        raise ValueError(f"Unsupported API type: {api_type}")

    return provider_class(provider_config), LLMCache(
        model_name=provider_config.get("model", "llm_model")
    )
