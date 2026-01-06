import abc
from pathlib import Path
from typing import Any, Protocol, Literal, TypeVar

from diskcache import Cache as DCache, Disk
from pydantic import BaseModel
from structlog import get_logger

from notarius.shared.constants import CACHES_DIR
from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)

ContentT = TypeVar("ContentT", bound=BaseModel)


class BaseCacheItem[ContentT](BaseModel):
    """Base class for cache item models.

    This is kept for backwards compatibility with OCR and LMv3 caches.
    New caches (like LLMCache) use pickle directly without this wrapper.
    """

    content: ContentT


class CacheProtocol(Protocol):
    """Protocol for cache implementations supporting pickle serialization."""

    def get(
        self,
        key: str,
        default: object | None = None,
        read: bool = False,
        expire_time: bool = False,
        tag: bool = False,
        retry: bool = False,
    ) -> object | None: ...

    def set(
        self,
        key: str,
        value: object,
        expire: int | None = None,
        read: bool = False,
        tag: str | None = None,
        retry: bool = False,
    ) -> bool: ...

    def delete(self, key: str, retry: bool = False) -> bool: ...

    def __len__(self) -> int | Any: ...


SupportedCacheTypes = Literal["LLMCache", "LMv3Cache", "PyTesseractCache"]


class BaseCache[ItemT](abc.ABC):
    """Base class for all cache implementations."""

    _item_type: type[ItemT]

    def __init__(
        self,
        cache_name: str,
        caches_dir: Path | None,
        size_limit: int = 1024 * 1024 * 1024,  # Default 1GB
    ):
        if caches_dir and not caches_dir.is_absolute():
            raise ValueError("Cache directory path must be absolute")

        self.caches_dir: Path = CACHES_DIR if caches_dir is None else caches_dir
        self.cache_name: str = cache_name

        self.cache_path.mkdir(parents=True, exist_ok=True)

        self.cache: CacheProtocol = DCache(
            directory=str(self.cache_path),
            disk=Disk,  # Use pickle for automatic serialization
            size_limit=size_limit,
        )

        logger.debug(
            f"{self.cache_type} cache initialised at {self.cache_path} with {len(self)} entries and {size_limit / (1024**3):.1f}GB limit"
        )

    @property
    def cache_path(self) -> Path:
        return self.caches_dir / self.cache_type / self.cache_name

    @property
    @abc.abstractmethod
    def cache_type(self) -> SupportedCacheTypes:
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, key: str) -> ItemT | None:
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, key: str, value: ItemT) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        return self.cache.delete(key)

    def __len__(self) -> int:
        return len(self.cache)
