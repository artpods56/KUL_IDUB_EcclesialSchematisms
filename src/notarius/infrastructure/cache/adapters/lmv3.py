"""LayoutLMv3 cache adapter using pickle for automatic serialization."""

import pickle
from pathlib import Path
from typing import TypedDict, override, final, cast

from structlog import get_logger

from notarius.application.ports.outbound.cache import BaseCache
from notarius.infrastructure.ml_models.lmv3.engine_adapter import LMv3Response
from notarius.shared.logger import Logger


logger: Logger = get_logger(__name__)


class LMv3CacheKeyParams(TypedDict, total=False):
    """Type definition for LMv3 cache key parameters."""

    image_hash: str


@final
class LMv3Cache(BaseCache[LMv3Response]):
    """Type-safe cache for LayoutLMv3 model predictions using pickle serialization.

    Pickle automatically handles the complex structures:
    - LMv3Response (dataclass)
    - SchematismPage (Pydantic model)
    - Nested structures (entries, BBox, etc.)

    No manual serialization/deserialization needed!
    """

    _item_type = LMv3Response

    def __init__(self, checkpoint: str, caches_dir: Path | None = None):
        """Initialize LMv3 cache.

        Args:
            checkpoint: Model checkpoint identifier for cache namespacing.
                       Different checkpoints use separate cache directories.
            caches_dir: Optional custom cache directory path.
        """
        self.checkpoint = checkpoint
        super().__init__(
            cache_name=checkpoint,
            caches_dir=caches_dir,
            size_limit=2 * 1024 * 1024 * 1024,
        )

    @property
    @override
    def cache_type(self):
        return "LMv3Cache"

    @override
    def get(self, key: str) -> LMv3Response | None:
        """Retrieve LMv3Response from cache.

        Args:
            key: Cache key

        Returns:
            LMv3Response if found, None otherwise
        """
        try:
            raw_data = self.cache.get(key)
            if raw_data is None:
                return None

            # Diskcache with Disk backend handles unpickling automatically
            return cast(LMv3Response, raw_data)

        except (pickle.PickleError, AttributeError, ImportError) as e:
            logger.warning(
                "cache_deserialization_failed",
                key=key[:16],
                error=str(e),
                error_type=type(e).__name__,
                checkpoint=self.checkpoint,
            )
            return None

    @override
    def set(self, key: str, value: LMv3Response) -> bool:
        """Cache a LMv3Response.

        Args:
            key: Cache key
            value: LMv3Response to cache

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
                checkpoint=self.checkpoint,
            )
            return False
