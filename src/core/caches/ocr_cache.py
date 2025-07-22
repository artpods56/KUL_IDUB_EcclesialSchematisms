
from core.caches.base_cache import BaseCache
from pydantic import BaseModel
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

from structlog import get_logger

class BaseOcrCacheItem(BaseModel):
    text: str

class PyTesseractCacheItem(BaseOcrCacheItem):
    bbox: List[Tuple[int, int, int, int]]
    words: List[str]

class BaseOcrCache(BaseCache):
    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__(cache_dir)

    def normalize_kwargs(self, **kwargs) -> Dict[str, Any]:
        return {
            "image_hash": kwargs.get("image_hash"),
        }

class PyTesseractCache(BaseOcrCache):
    def __init__(self, language: str = "lat+pol+rus", cache_dir: Optional[Path] = None):
        """Cache wrapper for PyTesseract OCR results.

        Args:
            language: Languages string passed to Tesseract. Used only for logging purposes so that
                separate language setups create their own dedicated cache directory.
            cache_dir: Optional path overriding the default cache directory (taken from the
                ``CACHE_DIR`` environment variable).
        """
        self.language = language
        self.logger = get_logger(__name__).bind(language=language)

        super().__init__(cache_dir)


    def normalize_kwargs(self, **kwargs) -> Dict[str, Any]:
        # We cache OCR results (text, words, bboxes) once per *image* and *language*.
        # The output format requested by the caller (text_only vs. full) is **not** part of
        # the cache key so that we avoid storing duplicate entries for the same image.
        return {
            "image_hash": kwargs.get("image_hash"),
            "language": self.language,
        }