import json
from typing import Dict, Any, Optional

from PIL import Image
from omegaconf import DictConfig
from openai.types.chat import ChatCompletionMessageParam
from pydantic_core import ValidationError
from structlog import get_logger

from core.caches.utils import get_image_hash, get_text_hash
from core.models.base import ConfigurableModel
from core.models.llm.factory import llm_provider_factory
from core.models.llm.prompt_manager import PromptManager
from core.models.llm.utils import messages_to_string
from core.schemas.data.cache import LLMCacheItem

logger = get_logger(__name__)


class LLMModel(ConfigurableModel):
    """LLM model wrapper with unified predict interface."""

    def __init__(
        self,
        config: DictConfig,
        enable_cache: bool = True,
        test_connection: bool = True,
        retries: int = 5,
    ):
        self.provider, self.cache = llm_provider_factory(config)
        self.prompt_manager = PromptManager(config.predictor.get("template_dir"))
        self.enable_cache = enable_cache
        self.test_connection = test_connection
        self.retries = retries

        self.last_messages: list[ChatCompletionMessageParam] = []

    @classmethod
    def from_config(cls, config: DictConfig) -> "LLMModel":
        return cls(config=config)

    def get_parsed_messages(self) -> str:
        parsed = messages_to_string(self.last_messages)
        return parsed

    def _predict(self, messages) -> dict[str, Any] | None:

        self.messages = messages

        for _ in range(self.retries):

            try:
                response = self.provider.generate_response(messages)
                logger.info("Generated response", response=response)
                json_response = json.loads(response)
                return json_response
            except ValidationError as e:
                logger.warning(f"Failed to generate response: {e}")
                raise

    def predict(
        self,
        context: dict[str, Any] | None = None,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        system_prompt: str = "system.j2",
        user_prompt: str = "user.j2",
        invalidate_cache: bool = False,
    ) -> tuple[Dict[str, Any], str]:
        """Generate predictions_data using the LLM model.

        Supports three modes:
        - Image-only: Provide only `image` parameter
        - Text-only: Provide only `text` parameter
        - Multimodal: Provide both `image` and `text` parameters

        Args:
            image: PIL Image object for vision processing
            text: OCR text string for text processing
            context: Additional context dictionary
            system_prompt: System prompt template name
            user_prompt: User prompt template name
            invalidate_cache: Invalidate cache

        Returns:
            Dictionary with structured prediction results

        Raises:
            ValueError: If neither image nor text is provided
        """
        if not context:
            context = {}

        system_prompt = self.prompt_manager.render_prompt(system_prompt, context)
        user_prompt = self.prompt_manager.render_prompt(user_prompt, context)

        messages: list[ChatCompletionMessageParam] = [
            self.provider.construct_system_message(system_prompt)
        ]
        if image is not None:
            messages.append(
                self.provider.construct_user_image_message(image, user_prompt)
            )
        else:
            messages.append(self.provider.construct_user_text_message(user_prompt))

        if image is None and text is None:
            raise ValueError("At least one of 'image' or 'text' must be provided")

        hints = context.get("hints", None)
        schematism = context.get("schematism", None)
        filename = context.get("filename", None)

        parsed_messages = messages_to_string(messages)

        if self.enable_cache:
            hash_key = self.cache.generate_hash(
                image_hash=get_image_hash(image) if image is not None else None,
                text_hash=get_text_hash(text),
                messages_hash=get_text_hash(parsed_messages),
                hints=hints,
            )

            if invalidate_cache:
                self.cache.delete(hash_key)

            try:
                cache_item_data = self.cache.get(key=hash_key)
                if cache_item_data is not None:
                    cache_item = LLMCacheItem(**cache_item_data)
                    return cache_item.response, parsed_messages
            except ValidationError:
                self.cache.delete(key=hash_key)

            response = self._predict(messages)

            cache_item_data = {
                "response": response,
                "hints": hints,
            }
            cache_item = LLMCacheItem(**cache_item_data)

            self.cache.set(
                key=hash_key,
                value=cache_item.model_dump(),
                schematism=schematism,
                filename=filename,
            )

            return response, parsed_messages

        else:
            return self._predict(messages), parsed_messages
