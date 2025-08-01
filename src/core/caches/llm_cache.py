import json
from typing import Dict, Optional, Union
from pathlib import Path

from structlog import get_logger
from core.caches.base_cache import BaseCache


class LLMCache(BaseCache):
    def __init__(self, model_name: str, caches_dir: Optional[Path] = None):
        self.logger = get_logger(__name__).bind(model_name=model_name)

        self.model_name = self._parse_model_name(model_name)

        super().__init__(caches_dir)

        self._setup_cache(
            caches_dir = self._caches_dir,
            cache_type = self.__class__.__name__,
            cache_name = self.model_name,
        )

    def _parse_model_name(self, model_name: str) -> str:
        """
        Parses the model name to ensure it is in a valid format.
        e.g., /models/gemma-3-27b-it-Q4_K_M.gguf -> gemma-3-27b-it-Q4_K_M
        """
        return model_name.split("/")[-1].split(".")[0]

    def normalize_kwargs(self, **kwargs):
        return {
            "image_hash": kwargs.get("image_hash"),
            "text_hash": kwargs.get("text_hash"),
            "messages": json.dumps(kwargs.get("messages"), ensure_ascii=False),
            "hints": json.dumps(kwargs.get("hints"), ensure_ascii=False)
        }