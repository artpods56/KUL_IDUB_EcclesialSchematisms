import json

from pydantic import BaseModel
from typing import Dict, Optional, Union
from pathlib import Path

from logging import getLogger
logger = getLogger(__name__)

from core.caches.base_cache import BaseCache


class LLMCacheItem(BaseModel):
    response: Union[str, Dict]
    hints: Optional[Dict]

class LLMCache(BaseCache):
    def __init__(self, model_name: str, cache_dir: Optional[Path] = None):
        self.model_name = model_name
        super(LLMCache, self).__init__(cache_dir)

    def normalize_kwargs(self, **kwargs):
        return {
            "image_hash": kwargs.get("image_hash"),
            "messages": json.dumps(kwargs.get("messages"), ensure_ascii=False),
            "hints": json.dumps(kwargs.get("hints"), ensure_ascii=False)
        }

    def _setup_cache(self):
        if not self.model_cache_dir.exists():
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.model_cache_dir / f"{self.model_name}.json"
        self.cache = self.load_cache()
        self.save_cache()