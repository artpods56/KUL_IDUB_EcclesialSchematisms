import logging
import os
from abc import ABC, abstractmethod

from PIL import Image
from omegaconf import DictConfig
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)

from core.models.llm.utils import encode_image_to_base64, make_all_properties_required
from core.schemas.data.schematism import SchematismPage

logger = logging.getLogger(__name__)


class LLMProvider[TClient](ABC):
    """Abstract base class for all LLM provider interfaces."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = config.get("model")
        if not self.model:
            raise ValueError("Model must be specified in the provider config.")
        self.api_kwargs = dict(config.get("api_kwargs", {}))
        self.client: TClient = self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> TClient:
        """Initialize the specific LLM client (e.g., OpenAI)."""
        pass

    @abstractmethod
    def generate_response(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Generate a response from the LLM given a list of messages."""
        pass

    def construct_system_message(self, prompt: str) -> ChatCompletionSystemMessageParam:
        """Construct system message from template."""
        return {"role": "system", "content": prompt}

    def construct_user_text_message(
        self, prompt: str
    ) -> ChatCompletionUserMessageParam:
        """Construct text-only message from template."""
        return {
            "role": "user",
            "content": prompt,
        }

    def construct_user_image_message(
        self,
        pil_image: Image.Image,
        prompt: str,
    ) -> ChatCompletionUserMessageParam:
        """Construct user message with image from template."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        base64_image = encode_image_to_base64(pil_image)

        text_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": prompt,
        }

        image_part: ChatCompletionContentPartImageParam = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
        }

        message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": [text_part, image_part],
        }

        return message


# --- OpenAI Provider ---


class OpenAIProvider(LLMProvider[OpenAI]):
    """Concrete implementation for OpenAI-compatible APIs."""

    def _initialize_client(self) -> OpenAI:
        """Initialize the OpenAI client."""
        api_key_env_var = self.config.get("api_key_env_var")
        if not api_key_env_var:
            raise ValueError(
                "api_key_env_var must be set in the OpenAI provider config."
            )
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env_var} is not set.")

        return OpenAI(api_key=api_key, base_url=self.config.get("base_url"))

    def generate_response(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Generate response from an OpenAI-compatible model."""

        schema_dict = SchematismPage.model_json_schema()

        response_format: ResponseFormatJSONSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": "schematism_page",
                "description": "Schema for SchematismPage response",
                "schema": make_all_properties_required(schema_dict),
                "strict": True,
            },
        }

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                **self.api_kwargs,
                response_format=response_format,
            )
            if not response.choices:
                raise ValueError("No choices returned from the API response.")

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response from OpenAI API: {e}")
            raise
