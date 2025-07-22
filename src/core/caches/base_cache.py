from abc import abstractmethod
from pydantic import BaseModel
from typing import Dict, Optional, Tuple, List, Any
import json
import hashlib
from pathlib import Path
import os
from PIL import Image

from structlog.typing import FilteringBoundLogger

class BaseCache:
    """Base class for all cache implementations.
    """

    logger: FilteringBoundLogger

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache: Dict[str, Any] = {}
        self.cache_version = "v1"
        self._cache_loaded = False

        self.cache_dir = os.getenv("CACHE_DIR", None) if cache_dir is None else cache_dir

        if not self.cache_dir:
            raise ValueError("CACHE_DIR environment variable is not set")
        else:
            self.model_cache_dir = self.cache_dir / Path(self.__class__.__name__)
            self._setup_cache()
            if not self._cache_loaded:
                self.logger.info(f"Cache initialized with {len(self.cache)} entries")

    @abstractmethod
    def normalize_kwargs(self, **kwargs) -> Dict[str, Any]:
        pass

    def _setup_cache(self):
        if not self.model_cache_dir.exists():
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.model_cache_dir / f"{self.__class__.__name__.lower()}.json"
        self.cache = self.load_cache()
        # Only save on initial setup if cache is empty (new cache) and cache file doesn't exist
        if not self.cache and not self.cache_file.exists():
            self.save_cache()

    def load_cache(self):
        """Load existing cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    if not self._cache_loaded:
                        self.logger.info(f"Loaded cache from {self.cache_file}")
                        self._cache_loaded = True
                    return cache
            except Exception as e:
                raise e
        else:
            if not self._cache_loaded:
                self.logger.debug(f"No existing cache found, creating new cache at {self.cache_file}")
                self._cache_loaded = True
            return {}

    def save_cache(self, silent: bool = False):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                # Handle both Pydantic models and dictionaries
                cache_dict = {}
                for k, v in self.cache.items():
                    if hasattr(v, 'model_dump'):
                        cache_dict[k] = v.model_dump()
                    else:
                        cache_dict[k] = v
                json.dump(cache_dict, f, indent=4, ensure_ascii=False)
            if not silent:
                self.logger.debug(f"Cache saved to {self.cache_file}")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")


    def get_image_hash(self, pil_image: Image.Image) -> str:
        """Generate a hash for the image to use as cache key."""
        # Convert to bytes for hashing
        img_bytes = pil_image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()

    # ---------------------------------------------------------------------
    # New helper for text-only or mixed requests
    # ---------------------------------------------------------------------
    def get_text_hash(self, text: str | None) -> str | None:
        """Return a deterministic SHA-256 hash for *text*.

        Returns None if *text* is ``None`` so callers can pass the value
        directly into ``generate_hash`` without extra conditionals.
        """
        if text is None:
            return None
        return hashlib.sha256(text.encode()).hexdigest()

    def generate_hash(self, **kwargs):
        relevant = self.normalize_kwargs(**kwargs)
        key_data = relevant
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self.cache[key]
    
    def __setitem__(self, key: str, value: Dict[str, Any]):
        self.cache[key] = value
    
    def __len__(self) -> int:
        return len(self.cache)