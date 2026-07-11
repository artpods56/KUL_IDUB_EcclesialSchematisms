from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerationParams(BaseModel):
    max_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


BackendType = Literal["llama", "lm_studio", "openai", "mistral", "openrouter"]


class ClientConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: BackendType = "openai"
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env_var: str = "OPENAI_API_KEY"
    structured_output: bool = False
    template_dir: str = "prompts"
    params: GenerationParams = Field(default_factory=GenerationParams)


def default_clients() -> dict[str, ClientConfig]:
    return {
        "openai": ClientConfig(backend="openai"),
        "openrouter": ClientConfig(
            backend="openrouter",
            model="google/gemini-2.5-flash-preview",
            base_url="https://openrouter.ai/api/v1",
            api_key_env_var="OPENROUTER_API_KEY",
        ),
        "lm_studio": ClientConfig(
            backend="lm_studio",
            model="local-model",
            base_url="http://localhost:1234/v1",
            api_key_env_var="LM_STUDIO_KEY",
        ),
        "llama": ClientConfig(
            backend="llama",
            model="local-llama",
            base_url="http://localhost:8080/v1",
            api_key_env_var="LLAMA_API_KEY",
        ),
    }


class BackendSelection(BaseModel):
    type: BackendType = "openrouter"
    max_retries: int = 5


class LLMEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: BackendSelection = Field(default_factory=BackendSelection)
    clients: dict[str, ClientConfig] = Field(default_factory=default_clients)

