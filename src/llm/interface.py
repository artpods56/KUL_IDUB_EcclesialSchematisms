import os
from typing import Optional, Union, Dict, Any, List, cast
import json

import logging

from mistralai import Mistral
from openai import OpenAI
from datasets import Dataset
from omegaconf import DictConfig
from PIL.Image import Image

from llm.prompt_manager import PromptManager
from llm.utils import encode_image_to_base64

from llm.structured import PageData
CLIENT_TYPE = Union[Mistral, OpenAI]

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Core LLM interface for making API calls to different LLM providers.
    Handles client initialization and basic message generation.
    """

    def __init__(self, interface_config: DictConfig, api_key: Optional[str] = None):
        self.interface_config = interface_config
        self.prompt_manager = PromptManager(template_dir=interface_config.get("template_dir", "prompts"))
        self.client: CLIENT_TYPE = self._initialize_client(api_key)
        self._test_connection()

    def _initialize_client(self, api_key: Optional[str] = None) -> CLIENT_TYPE:
        """Initialize the LLM client based on the specified API type."""
        logger.info("Initializing LLM client...")
        
        api_type = self.interface_config.get("api_type", "openai").lower()
        
        match api_type:
            case "mistral":
                if api_key is None:
                    api_key = os.environ.get("MISTRAL_API_KEY")
                    if not api_key:
                        raise ValueError("MISTRAL_API_KEY must be set in the environment or passed as an argument.")
                return Mistral(api_key=api_key)
            
            case "openai":
                if api_key is None:
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY must be set in the environment or passed as an argument.")
                return OpenAI(
                    api_key=api_key, 
                    base_url=self.interface_config.get("base_url", None)
                )
            
            case _:
                raise ValueError(f"Unsupported API type: {api_type}")

    def _test_connection(self):
        """Test client connection after initialization."""
        try:
            logger.info("Testing connection to LLM client...")
            api_kwargs = self.interface_config.get("api_kwargs", {}) 

            # Attempt to make a simple API call to verify connection
            if isinstance(self.client, Mistral):
                self.client.chat.complete(messages=[{"role": "user", "content": "Hi! This is a connection test."}],
                                          model=self.interface_config.get("model", None),
                                          **api_kwargs)
                logger.info("Successfully connected to Mistral client.")
            elif isinstance(self.client, OpenAI):
                self.client.chat.completions.create(messages=[{"role": "user", "content": "Hi! This is a connection test."}],
                                                    model=self.interface_config.get("model", None),
                                                    **api_kwargs)
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

    def _construct_user_message_with_image(self, pil_image: Image, prompt_name: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Construct user message with image from template."""
        base64_image = encode_image_to_base64(pil_image)
        if isinstance(self.client,Mistral):
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
                    "image_url":  image_struct 
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
        logger.info(f"Messages for generation: {json.dumps(messages, indent=2)}")
        api_kwargs = dict(self.interface_config.get("api_kwargs", {}))
        model = self.interface_config.get("model", None)
        logger.info(f"Using model: {model}")
        
        if self.interface_config.get("structured_output", False):
            logger.info(f"Structured output enabled, setting response format to JSON schema using {PageData.model_json_schema()}.")
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "PageData",
                    "schema": PageData.model_json_schema(),
                    "strict": True,
                    }
                }
            api_kwargs["response_format"] = response_format
        try:
            if isinstance(self.client, Mistral):
                
                response = self.client.chat.complete(
                    model=model,
                    messages=messages,
                    **api_kwargs,
                )
                # return response.choices[0].message.content if response.choices else None
                if not response.choices:
                    raise ValueError("No choices returned from Mistral response")
                else:
                    return response.choices[0].message.content
                
            elif isinstance(self.client, OpenAI):

                response = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    **api_kwargs,
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

    def generate_vision_response(self, pil_image: Image, system_prompt: str, user_prompt: str, context: Optional[Dict[str, Any]] = None):
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




