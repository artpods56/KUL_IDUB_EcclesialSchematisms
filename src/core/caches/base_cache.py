from abc import abstractmethod
from ast import Constant
from datetime import datetime
from typing_extensions import reveal_type
from pydantic import BaseModel, Field
from typing import ByteString, Dict, Optional, Sized, Tuple, List, Any, Union, cast
import json
import hashlib
from pathlib import Path
import os
from PIL import Image
import time

from structlog.typing import FilteringBoundLogger
from diskcache import Cache as DCache
from diskcache import JSONDisk

class BaseCache:
    """Base class for all cache implementations.
    """

    logger: FilteringBoundLogger

    def __init__(self, caches_dir: Optional[Path] = None):

        self.cache: DCache
        self._cache_loaded = False

        env_caches_dir = os.getenv("CACHE_DIR", None)
        if env_caches_dir is None:
            raise ValueError("CACHE_DIR environment variable is not set.")

        self._caches_dir = caches_dir if caches_dir is not None else Path(env_caches_dir)

    @abstractmethod
    def normalize_kwargs(self, **kwargs) -> Dict[str, Any]:
        pass

    def _setup_cache(self, caches_dir: Path, cache_type: str, cache_name: str):
        """Initialise the on-disk cache directory and diskcache backend."""

        self.model_cache_dir = caches_dir / cache_type / cache_name
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache = DCache(
            directory=str(self.model_cache_dir),
            disk=JSONDisk,
        )

        # pass methods to our class
        self.get = self.cache.get
        self.delete = self.cache.delete

        self.logger.info(f"{cache_type} cache initialised at {self.model_cache_dir} with {len(self.cache)} entries")

        self._cache_loaded = True

    def generate_hash(self, **kwargs):
        key_data = self.normalize_kwargs(**kwargs)
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def __getitem__(self, key: str):
        return self.cache.get(key)

    def __setitem__(self, key: str, value: Dict[str, Any]):
        """Insert *value* under *key* and attach an optional tag for schematism."""
        self.cache.set(key, value)

    def set(self, key: str, value: Dict[str, Any], schematism: Optional[str] = None, filename: Optional[str] = None):

        if schematism is None:
            schematism = "null"
        if filename is None:
            filename = "null"

        tag = f"{schematism}:{filename}"
        self.cache.set(key, value, tag=tag)

    def __len__(self) -> int:
        return len(
            cast(Sized,self.cache)
        )