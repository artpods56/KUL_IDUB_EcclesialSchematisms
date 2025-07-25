from pydantic import BaseModel
from typing import Dict, Optional, Tuple, List

from core.models.llm.structured import PageData

class BaseCacheItem(BaseModel):
    metadata: Optional[Dict] = None  # Store metadata as a dictionary for flexibility

class LLMCacheItem(BaseCacheItem):
    """Cache item model for LLM models.
    """
    response: PageData
    hints: Optional[Dict] = None

class LMv3CacheItem(BaseCacheItem):
    """Cache item model for LMv3 models.
    """
    raw_predictions: Tuple[List, List, List]
    structured_predictions: PageData

class BaseOcrCacheItem(BaseCacheItem):
    """Base cache item model for OCR models.
    """
    text: str

class PyTesseractCacheItem(BaseOcrCacheItem):
    """Cache item model for PyTesseract OCR models.
    """
    bbox: List[Tuple[int, int, int, int]]
    words: List[str]
