from core.caches.base_cache import BaseCache
from pydantic import BaseModel
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

from structlog import get_logger

class LMv3CacheItem(BaseModel):
    raw_preds: Optional[Tuple[List, List, List]]
    structured_preds: Optional[Dict]

class LMv3Cache(BaseCache):
    def __init__(self, checkpoint: str, cache_dir: Optional[Path] = None):

        self.logger = get_logger(__name__).bind(checkpoint=checkpoint)
        super().__init__(cache_dir)

    def normalize_kwargs(self, **kwargs) -> Dict[str, Any]:
        return {
            "image_hash": kwargs.get("image_hash"),
            "structured_preds": kwargs.get("structured_preds")
        }