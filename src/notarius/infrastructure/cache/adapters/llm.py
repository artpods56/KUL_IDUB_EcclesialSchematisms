"""LLM cache adapter using pickle for automatic serialization."""

import pickle
from pathlib import Path
from typing import final, override, cast

from pydantic import BaseModel
from structlog import get_logger

from notarius.application.ports.outbound.cache import BaseCache
from notarius.infrastructure.llm.engine_adapter import CompletionResult
from notarius.infrastructure.llm.utils import parse_model_name
from notarius.shared.logger import Logger

logger = cast(Logger, get_logger(__name__))


@final
class LLMCache[T: BaseModel](BaseCache[CompletionResult[T]]):
    """Type-safe cache for LLM responses using pickle serialization.

    Pickle automatically handles the complex nested structure of:
    - CompletionResult (dataclass)
    - BaseProviderResponse (dataclass)
    - Conversation (dataclass)
    - Pydantic models (BaseModel)

    No manual serialization/deserialization needed!

    Note: The generic type T is erased at runtime (pickle doesn't preserve it),
    but it helps with static type checking.
    """

    _item_type: type = CompletionResult

    def __init__(self, model_name: str, caches_dir: Path | None = None):
        # Increased size limit to 10GB for multimodal LLM cache
        super().__init__(
            cache_name=parse_model_name(model_name),
            caches_dir=caches_dir,
            size_limit=10 * 1024 * 1024 * 1024,
        )

    @property
    @override
    def cache_type(self):
        return "LLMCache"

    @override
    def get(self, key: str) -> CompletionResult[T] | None:
        """Retrieve CompletionResult from cache.

        Args:
            key: Cache key

        Returns:
            CompletionResult if found, None otherwise
        """
        try:
            raw_data = self.cache.get(key)
            if raw_data is None:
                return None

            # Diskcache with Disk backend handles unpickling automatically
            # Note: Generic type T is erased at runtime
            return cast(CompletionResult[T], raw_data)

        except (pickle.PickleError, AttributeError, ImportError) as e:
            logger.warning(
                "cache_deserialization_failed",
                key=key,
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    @override
    def set(self, key: str, value: CompletionResult[T]) -> bool:
        """Store CompletionResult in cache.

        Args:
            key: Cache key
            value: CompletionResult to cache

        Returns:
            True if stored successfully
        """
        try:
            return self.cache.set(key, value)
        except (pickle.PickleError, TypeError) as e:
            logger.error(
                "cache_serialization_failed",
                key=key,
                error=str(e),
            )
            return False
