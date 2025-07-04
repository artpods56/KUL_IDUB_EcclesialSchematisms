"""Base model interface for all ML models in the AI Osrodek pipeline."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
from transformers import AutoProcessor


class BaseModel(ABC):
    """Base class for all ML models in the AI Osrodek pipeline."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.processor = None
        self.is_trained = False
    
    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None) -> None:
        """Load the model from checkpoint or pretrained weights."""
        pass
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """Preprocess inputs for the model."""
        pass
    
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Make predictions using the model."""
        pass
    
    @abstractmethod
    def postprocess(self, outputs: Any) -> Any:
        """Postprocess model outputs."""
        pass
    
    def save_model(self, save_path: str) -> None:
        """Save the model to the specified path."""
        if self.model is not None:
            self.model.save_pretrained(save_path)
        if self.processor is not None:
            self.processor.save_pretrained(save_path)
    
    def to_device(self, device: Union[str, torch.device]) -> None:
        """Move model to specified device."""
        if self.model is not None:
            self.model.to(device)


class BaseDocumentModel(BaseModel):
    """Base class for document processing models."""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name, config)
        self.id2label = {}
        self.label2id = {}
    
    def set_labels(self, id2label: Dict[int, str], label2id: Dict[str, int]) -> None:
        """Set label mappings for the model."""
        self.id2label = id2label
        self.label2id = label2id
    
    @abstractmethod
    def extract_entities(self, image_path: str, texts: list, boxes: list) -> Dict[str, Any]:
        """Extract entities from document image."""
        pass
