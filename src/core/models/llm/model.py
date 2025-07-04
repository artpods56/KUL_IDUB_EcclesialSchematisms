import json

from typing import Dict, Any, Optional
from PIL import Image
from omegaconf import DictConfig

from core.models.llm.interface import LLMInterface
from core.models.llm.prompt_manager import PromptManager
from core.caches.llm_cache import LLMCache, LLMCacheItem

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
        



    def _predict(self, pil_image: Image.Image, hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Predict on PIL image and return JSON results.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Dictionary with structured prediction results
        """
        response, messages = self.interface.generate_vision_response(
            pil_image=pil_image,
            system_prompt="system.j2",
            user_prompt="user.j2",
            context={"hints": hints}
        )
        
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

    def predict(self, pil_image: Image.Image, hints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.enable_cache:
            hash_key = self.cache.generate_hash(
                image_hash=self.cache.get_image_hash(pil_image),
                hints=hints,
                messages=[
                    self.interface._construct_system_message("system.j2", context={"hints": hints}),
                    self.interface._construct_user_message_with_image(pil_image, "user.j2", context={"hints": hints})
                ]
            )


            
            try:
                cached_result = self.cache[hash_key]
            except KeyError:
                cached_result = None

            if cached_result:
                return cached_result["response"]
            else:
                result = self._predict(pil_image, hints)
                self.cache[hash_key] = LLMCacheItem(
                    response=result,
                    hints=hints
                ).model_dump()
                
                self.cache.save_cache()
                return result
        else:
            return self._predict(pil_image, hints)