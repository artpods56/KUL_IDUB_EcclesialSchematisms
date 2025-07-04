"""Model factory for unified model interface."""

from typing import Dict, Any, Union
from PIL import Image

from core.models.lmv3.model import LMv3Model
from core.models.llm.model import LLMModel


class ModelFactory:
    """Factory class for creating models with unified predict interface."""
    
    AVAILABLE_MODELS = {
        "lmv3": LMv3Model,
        "llm": LLMModel,
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> Union[LMv3Model, LLMModel]:
        """Create a model instance with unified predict interface.
        
        Args:
            model_type: Type of model to create ("lmv3" or "llm")
            config: Configuration dictionary for the model
            
        Returns:
            Model instance with predict(pil_image) -> Dict method
            
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type not in cls.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Available models: {list(cls.AVAILABLE_MODELS.keys())}"
            )
        
        model_class = cls.AVAILABLE_MODELS[model_type]
        return model_class(config)
    
    @classmethod
    def list_models(cls) -> list:
        """List all available model types."""
        return list(cls.AVAILABLE_MODELS.keys())


class UnifiedPredictor:
    """Unified predictor that wraps multiple models with the same interface."""
    
    def __init__(self, model_configs: Dict[str, Dict[str, Any]]):
        """Initialize with multiple model configurations.
        
        Args:
            model_configs: Dictionary mapping model names to their configs
                          e.g., {"lmv3_main": {"type": "lmv3", "config": {...}}}
        """
        self.models = {}
        
        for model_name, model_info in model_configs.items():
            model_type = model_info["type"]
            config = model_info["config"]
            self.models[model_name] = ModelFactory.create_model(model_type, config)
    
    def predict(self, model_name: str, pil_image: Image.Image) -> Dict[str, Any]:
        """Predict using specified model.
        
        Args:
            model_name: Name of the model to use
            pil_image: PIL Image object
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            ValueError: If model_name is not found
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self.models.keys())}"
            )
        
        return self.models[model_name].predict(pil_image)
    
    def predict_all(self, pil_image: Image.Image) -> Dict[str, Dict[str, Any]]:
        """Run prediction on all loaded models.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Dictionary mapping model names to their prediction results
        """
        results = {}
        for model_name, model in self.models.items():
            try:
                results[model_name] = model.predict(pil_image)
            except Exception as e:
                results[model_name] = {"error": str(e)}
        
        return results
    
    def list_loaded_models(self) -> list:
        """List all loaded model names."""
        return list(self.models.keys())
