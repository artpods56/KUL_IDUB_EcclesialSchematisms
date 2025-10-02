from pydantic import BaseModel
from typing import Dict, Optional, Tuple, Any, List

from core.schemas.data.schematism import SchematismPage

class BaseCacheItem(BaseModel):
    metadata: Optional[Dict[str,Any]] = None  # Store metadata as a dictionary for flexibility

class LLMCacheItem(BaseCacheItem):
    """Cache item model for LLM models.
    """
    response: Dict[str, Any]
    hints: Optional[Dict[str, Any]] = None

class LMv3CacheItem(BaseCacheItem):
    """Cache item model for LMv3 models.
    """
    raw_predictions: Tuple[list, list, list]
    structured_predictions: SchematismPage

class BaseOcrCacheItem(BaseCacheItem):
    """Base cache item model for OCR models.
    """
    text: str

class PyTesseractCacheItem(BaseOcrCacheItem):
    """Cache item model for PyTesseract OCR models.
    """
    bbox: list[Tuple[int, int, int, int]]
    words: list[str]
