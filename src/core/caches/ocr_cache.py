
from core.caches.base_cache import BaseCache
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path

from structlog import get_logger

class BaseOcrCache(BaseCache):
    """Base cache model for OCR models.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        super().__init__(cache_dir)

    def normalize_kwargs(self, **kwargs) -> Dict[str, Any]:
        return {
            "image_hash": kwargs.get("image_hash"),
        }
        
class PyTesseractCache(BaseOcrCache):
    def __init__(self, language: str = "lat+pol+rus", caches_dir: Optional[Path] = None):
        """Cache wrapper for PyTesseract OCR results.

        Args:
            language: Languages string passed to Tesseract. Used only for logging purposes so that
                separate language setups create their own dedicated cache directory.
            caches_dir: Optional path overriding the default cache directory (taken from the
                ``CACHE_DIR`` environment variable).
        """

        self.logger = get_logger(__name__).bind(language=language)
        self.language = language

        super().__init__(caches_dir)
        self._setup_cache(
            caches_dir=self._caches_dir,
            cache_type=self.__class__.__name__,
            cache_name=language,
        )

    def normalize_kwargs(self, **kwargs) -> Dict[str, Any]:
        return {
            "image_hash": kwargs.get("image_hash"),
            "language": self.language,
        }