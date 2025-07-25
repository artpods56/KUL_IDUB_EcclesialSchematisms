import os
from typing import Optional, Union, Dict, Any

import logging

from mistralai import Mistral
from openai import OpenAI
from omegaconf import DictConfig
from PIL.Image import Image

from core.models.llm.prompt_manager import PromptManager
from core.models.llm.utils import encode_image_to_base64

from core.models.llm.structured import PageData

from openai.types.shared.chat_model import ChatModel

CLIENT_TYPE = Union[Mistral, OpenAI]
LLM_MODELS = Union[ChatModel, str]
logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Core LLM interface for making API calls to different LLM providers.
    Handles client initialization and basic message generation.
    """

    def __init__(self, interface_config: DictConfig, prompt_manager: PromptManager, api_type: str,
                 api_key: Optional[str] = None, test_connection: bool = True):
        self.interface_config = interface_config
        self.prompt_manager = prompt_manager
        self.api_type = api_type
        self.client: CLIENT_TYPE = self._initialize_client(api_key)
        self.api_kwargs = dict(self.interface_config.get("api_kwargs", {}))
        self.model: LLM_MODELS = self.interface_config.get("model", None)
        if not self.model:
            raise ValueError("Model name must be specified in the config")

        if test_connection:
            self._test_connection()

    def _initialize_client(self, api_key: Optional[str] = None) -> CLIENT_TYPE:
        """Initialize the LLM client based on the specified API type."""
        logger.info("Initializing LLM client...")

        match self.api_type.lower():
            case "mistral":
                if api_key is None:
                    api_key = os.environ.get("MISTRAL_API_KEY")
                    if not api_key:
                        raise ValueError("MISTRAL_API_KEY must be set in the environment or passed as an argument.")
                return Mistral(api_key=api_key)

            case "openai" | "lm_studio" | "openrouter":
                if api_key is None:
                    api_key_env_var = self.interface_config.get("api_key_env_var", None)
                    if api_key_env_var is None:
                        raise ValueError("api_key_env_var must be set in the config or passed as an argument.")
                    api_key = os.environ.get(api_key_env_var)
                    if not api_key:
                        raise ValueError(f"{api_key_env_var} must be set in the environment or passed as an argument.")
                return OpenAI(
                    api_key=api_key,
                    base_url=self.interface_config.get("base_url", None)
                )

            case _:
                raise ValueError(f"Unsupported API type: {self.api_type}")

    def _test_connection(self):
        """Test client connection after initialization."""
        try:
            logger.info("Testing connection to LLM client...")

            # Attempt to make a simple API call to verify connection
            if isinstance(self.client, Mistral):
                self.client.chat.complete(messages=[{"role": "user", "content": "Hi! This is a connection test."}],
                                          model=self.model
                                                **self.api_kwargs)
                logger.info("Successfully connected to Mistral client.")
            elif isinstance(self.client, OpenAI):
                self.client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hi! This is a connection test."}],
                    model=self.model,
                    **self.api_kwargs)
                logger.info("Successfully connected to OpenAI client.")
            else:
                raise ValueError(f"Unsupported client type: {type(self.client)}")
        except Exception as e:
            logger.error(f"Failed to connect to LLM client: {e}")
            raise e

    def _construct_system_message(self, prompt_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Construct system message from template."""
        return {
            "role": "system",
            "content": self.prompt_manager.render_prompt(prompt_name, **(context or {}))
        }

    def _construct_user_message_with_image(self, pil_image: Image, prompt_name: str,
                                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Construct user message with image from template."""
        # Ensure image is in RGB format for compatibility
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        base64_image = encode_image_to_base64(pil_image)
        if isinstance(self.client, Mistral):
            image_struct = f"data:image/jpeg;base64,{base64_image}"
        elif isinstance(self.client, OpenAI):
            image_struct = {"url": f"data:image/jpeg;base64,{base64_image}"}

        else:
            raise ValueError(f"Unsupported client type: {type(self.client)}")
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.prompt_manager.render_prompt(prompt_name, **(context or {}))
                },
                {
                    "type": "image_url",
                    "image_url": image_struct
                }
            ]
        }

    def _construct_text_message(self, prompt_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Construct text-only message from template."""
        return {
            "role": "user",
            "content": self.prompt_manager.render_prompt(prompt_name, **(context or {}))
        }

    def generate_response(self, messages):
        """Generate response from LLM given messages."""

        if self.interface_config.get("structured_output", False):
            import copy
            # Get the base schema
            base_schema = copy.deepcopy(PageData.model_json_schema())

            # Add additionalProperties: false for OpenRouter compatibility
            if self.api_type.lower() in ["openrouter", "openai", "lm_studio"]:
                # Ensure all object properties have additionalProperties: false
                def add_additional_properties_false(schema):
                    if isinstance(schema, dict):
                        if schema.get("type") == "object":
                            schema["additionalProperties"] = False
                        # Recursively process nested objects
                        for key, value in schema.items():
                            if key in ["properties", "items", "allOf", "anyOf", "oneOf"]:
                                if isinstance(value, dict):
                                    add_additional_properties_false(value)
                                elif isinstance(value, list):
                                    for item in value:
                                        if isinstance(item, dict):
                                            add_additional_properties_false(item)
                        # Also process $defs if present
                        if "$defs" in schema:
                            for def_name, def_schema in schema["$defs"].items():
                                if isinstance(def_schema, dict):
                                    add_additional_properties_false(def_schema)
                    return schema

                add_additional_properties_false(base_schema)
                # logger.info(f"Structured output enabled for {self.api_type}, using modified schema with additionalProperties: false")
            else:
                pass
                # logger.info(f"Structured output enabled for {self.api_type}, using original schema")

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "PageData",
                    "schema": base_schema,
                }
            }
            self.api_kwargs["response_format"] = response_format
        try:
            if isinstance(self.client, Mistral):

                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    **self.api_kwargs,
                )
                # return response.choices[0].message.content if response.choices else None
                if not response.choices:
                    raise ValueError("No choices returned from Mistral response")
                else:
                    return response.choices[0].message.content

            elif isinstance(self.client, OpenAI):

                response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    **self.api_kwargs,
                )

                if not response.choices:
                    raise ValueError("No choices returned from OpenAI response")
                else:
                    return response.choices[0].message.content

            else:
                raise ValueError(f"Unsupported client type: {type(self.client)}")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None

    def generate_vision_response(self, pil_image: Image, system_prompt: str, user_prompt: str,
                                 context: Optional[Dict[str, Any]] = None):
        """Generate response from vision-capable LLM.

        Returns response and the messages used for generation.
        """
        messages = [
            self._construct_system_message(system_prompt, context),
            self._construct_user_message_with_image(pil_image, user_prompt, context)
        ]
        return self.generate_response(messages), messages

    def generate_text_response(self, system_prompt: str, user_prompt: str, context: Optional[Dict[str, Any]] = None):
        """Generate text-only response from LLM.

        Returns response and the messages used for generation.
        """
        messages = [
            self._construct_system_message(system_prompt, context),
            self._construct_text_message(user_prompt, context)
        ]
        return self.generate_response(messages), messages




