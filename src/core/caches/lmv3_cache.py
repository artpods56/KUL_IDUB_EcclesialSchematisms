from core.caches.base_cache import BaseCache
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

from structlog import get_logger

class LMv3Cache(BaseCache):
    def __init__(self, checkpoint: str, caches_dir: Optional[Path] = None):
        self.logger = get_logger(__name__).bind(checkpoint=checkpoint)

        self.checkpoint = checkpoint

        super().__init__(caches_dir)

        self._setup_cache(
            caches_dir = self._caches_dir,
            cache_type = self.__class__.__name__,
            cache_name = checkpoint
        )

    def normalize_kwargs(seslf, **kwargs) -> Dict[str, Any]:
        return {
            "image_hash": kwargs.get("image_hash"),
            "structured_predictions": kwargs.get("structured_predictions")
        }
