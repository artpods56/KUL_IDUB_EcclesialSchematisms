import json

from typing import Dict, Any, Optional, cast
from PIL import Image
from pydantic_core import ValidationError

from core.models.llm.structured import PageData
from omegaconf import DictConfig

from core.models.llm.interface import LLMInterface
from core.models.llm.prompt_manager import PromptManager
from core.caches.llm_cache import LLMCache
from core.schemas.caches.entries import LLMCacheItem
from core.caches.utils import get_image_hash, get_text_hash
class LLMModel:
    """LLM model wrapper with unified predict interface."""
    
    def __init__(self, config: DictConfig, enable_cache: bool = True, test_connection: bool = True):
        self.config = config


        # Get the interface configuration based on api_type
        api_type = config.predictor.get("api_type", "openai")
        interface_config = config.interfaces.get(api_type)
        
        if interface_config is None:
            raise ValueError(f"No interface configuration found for api_type: {api_type}")
        

        self.enable_cache = enable_cache
        if self.enable_cache:
            self.cache = LLMCache(
                model_name=interface_config.get("model", "llm_cache").replace("/", "_")
            )
        # Create prompt manager with template directory from config
        template_dir = interface_config.get("template_dir", "prompts")
        prompt_manager = PromptManager(template_dir)
        
        # Initialize the interface with the specific interface config and prompt manager
        self.interface = LLMInterface(interface_config, prompt_manager, api_type, test_connection=test_connection)
        
        self.messages = []


    def _predict(self, image: Optional[Image.Image], text: Optional[str], **kwargs) -> Dict[str, Any]:
        """Generate prediction using the LLM interface.

        Args:
            image: PIL Image object for vision requests, or None for text-only
            text: OCR text string for text requests, or None for image-only
            **kwargs: Additional context passed to prompt templates

        Returns:
            Dictionary with structured prediction results
        """
        context_full = {"ocr_text": text, **kwargs}

        system_prompt = kwargs.get("system_prompt", "system.j2")
        user_prompt = kwargs.get("user_prompt", "user.j2")

        if image is not None:
            response, messages = self.interface.generate_vision_response(
                pil_image=image,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context=context_full,
            )
        else:
            response, messages = self.interface.generate_text_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context=context_full,
            )

        self.messages = messages
        
        # If structured output is enabled, response should already be JSON
        if self.config.interfaces.get(self.config.predictor.api_type, {}).get("structured_output", False):
            try:
                if response is None:
                    return {"raw_response": None, "error": "No response received"}
                parsed_response = json.loads(response) if isinstance(response, str) else response
                if isinstance(parsed_response, dict):
                    return parsed_response
                else:
                    return {"raw_response": parsed_response, "error": "Response is not a dictionary"}
            except json.JSONDecodeError:
                # Fallback if parsing fails
                return {"raw_response": response, "error": "Failed to parse structured output"}
        else:
            # For unstructured output, wrap in a standard format
            return {"raw_response": response}

    def predict(
            self,
            image: Optional[Image.Image] = None,
            text: Optional[str] = None,
            **kwargs,
    ) -> Dict[str, Any]:
        """Generate predictions using the LLM model.

        Supports three modes:
        - Image-only: Provide only `image` parameter
        - Text-only: Provide only `text` parameter
        - Multimodal: Provide both `image` and `text` parameters

        Args:
            image: PIL Image object for vision processing
            text: OCR text string for text processing
            **kwargs: Additional context (e.g., hints from other models)

        Returns:
            Dictionary with structured prediction results

        Raises:
            ValueError: If neither image nor text is provided
        """
        if image is None and text is None:
            raise ValueError("At least one of 'image' or 'text' must be provided")

        hints = kwargs.get("hints", None)
        schematism = kwargs.get("schematism", None)
        filename = kwargs.get("filename", None)

        # Check cache first if enabled
        if self.enable_cache:
            hash_key = self.cache.generate_hash(
                image_hash=get_image_hash(image) if image is not None else None,
                text_hash=get_text_hash(text),
                hints=hints,
            )

            try:
                cache_item_data = cast(
                    Optional[dict], self.cache.get(key=hash_key)
                )
                if cache_item_data is not None:
                    cache_item = LLMCacheItem(**cache_item_data)
                    return cache_item.response.model_dump()
            except ValidationError as e:
                self.cache.delete(key=hash_key)

            # If cache miss or validation error, compute fresh
            response = self._predict(image, text, **kwargs)

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

            return response

        else:
            return self._predict(image, text, **kwargs)

