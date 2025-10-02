from typing import Dict, Literal

from pydantic import BaseModel, Field, ConfigDict

from core.config.constants import ConfigType, ModelsConfigSubtype
from core.config.registry import register_config


class ModelApiKwargs(BaseModel):
    max_tokens: int = Field(
        default=4096,
        description="The maximum number of tokens to generate"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Controls randomness in the output"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Controls diversity via nucleus sampling"
    )


class BaseLLMModelConfig(BaseModel):
    model: str = Field(default="gpt-3.5-turbo", description="Model description")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the client"
    )
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable that specifies the API key to use"
    )
    structured_output: bool = Field(
        default=False,
        description="Whether to use structured output"
    )
    template_dir: str = Field(
        default="prompts",
        description="Path to template directory that is passed to the PromptManager"
    )
    api_kwargs: ModelApiKwargs = Field(
        default_factory=ModelApiKwargs,
        description="Model API kwargs"
    )


API_TYPES = Literal["llama", "lm_studio", "openai", "mistral", "openrouter"]

class PredictorConfig(BaseModel):
    api_type: API_TYPES = Field(default="openai", description="Type of API to use (e.g., 'lm_studio', 'openai')")
    template_dir: str = Field(
        default="prompts",
        description="Path to template directory that is passed to the PromptManager"
    )
    max_retries: int = Field(
        default=5,
        description="The maximum number of retry attempts to make before giving up"
    )


@register_config(ConfigType.MODELS, ModelsConfigSubtype.LLM)
class LLMModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictor: PredictorConfig = Field(default_factory=PredictorConfig, description="Predictor configuration")
    interfaces: Dict[API_TYPES, BaseLLMModelConfig] = Field(default_factory=dict, description="Interface configurations by description")
