from core.caches.base_cache import BaseCache
from pydantic import BaseModel
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from logging import getLogger
logger = getLogger(__name__)

class LMv3CacheItem(BaseModel):
    raw_preds: Optional[Tuple[List, List, List]]
    structured_preds: Optional[Dict]

class LMv3Cache(BaseCache):
    def __init__(self, cache_dir: Optional[Path] = None):
        super(LMv3Cache, self).__init__(cache_dir)

    def normalize_kwargs(self, **kwargs):
        return {
            "image_hash": kwargs.get("image_hash"),
            "structured_preds": kwargs.get("structured_preds")
        }