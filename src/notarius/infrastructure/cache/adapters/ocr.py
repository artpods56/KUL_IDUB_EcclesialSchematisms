"""OCR cache adapter using pickle for automatic serialization."""

import pickle
from pathlib import Path
from typing import TypedDict, override, final, cast

from structlog import get_logger

from notarius.application.ports.outbound.cache import BaseCache
from notarius.infrastructure.ocr.engine_adapter import OCRResponse
from notarius.shared.logger import Logger


logger: Logger = get_logger(__name__)


class OCRCacheKeyParams(TypedDict, total=False):
    """Type definition for OCR cache key parameters."""

    image_hash: str


@final
class PyTesseractCache(BaseCache[OCRResponse]):
    """Type-safe cache for PyTesseract OCR sample using pickle serialization.

    Pickle automatically handles the Pydantic models:
    - PyTesseractCacheItem
    - PyTesseractContent
    - BBox structures

    No manual serialization/deserialization needed!
    """

    _item_type = OCRResponse

    def __init__(
        self,
        language: str = "lat+pol+rus",
        caches_dir: Path | None = None,
    ):
        """Initialize PyTesseract cache.

        Args:
            language: Languages string passed to Tesseract. Used to create
                     separate cache directories for different language setups.
            caches_dir: Optional path overriding the default cache directory.
        """
        self.language = language
        super().__init__(
            cache_name=language,
            caches_dir=caches_dir,
            size_limit=2 * 1024 * 1024 * 1024,
        )

    @property
    @override
    def cache_type(self):
        return "PyTesseractCache"

    @override
    def get(self, key: str) -> OCRResponse | None:
        """Retrieve PyTesseractCacheItem from cache.

        Args:
            key: Cache key

        Returns:
            PyTesseractCacheItem if found, None otherwise
        """
        try:
            raw_data = self.cache.get(key)
            if raw_data is None:
                return None

            # Diskcache with Disk backend handles unpickling automatically
            return cast(OCRResponse, raw_data)

        except (pickle.PickleError, AttributeError, ImportError) as e:
            logger.warning(
                "cache_deserialization_failed",
                key=key[:16],
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    @override
    def set(self, key: str, value: OCRResponse) -> bool:
        """Cache a PyTesseractCacheItem.

        Args:
            key: Cache key
            value: PyTesseractCacheItem to cache

        Returns:
            True if cached successfully
        """
        try:
            # Diskcache with Disk backend handles pickling automatically
            return self.cache.set(key, value)
        except (pickle.PickleError, TypeError) as e:
            logger.error(
                "cache_serialization_failed",
                key=key[:16],
                error=str(e),
            )
            return False
