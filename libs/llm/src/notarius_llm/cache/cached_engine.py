from typing import Any

from pydantic import BaseModel

from notarius_core.ports.llm import CompletionRequest, CompletionResult, LLMCompletionEngine


class CachedLLMEngine:
    """Small cache decorator for tests and local runs."""

    def __init__(self, engine: LLMCompletionEngine):
        self._engine = engine
        self._cache: dict[str, CompletionResult[Any]] = {}

    @property
    def stats(self) -> Any:
        return self._engine.stats

    def process[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        key = repr(request)
        if key not in self._cache:
            self._cache[key] = self._engine.process(request)
        return self._cache[key]  # type: ignore[return-value]

    async def process_async[T: BaseModel](
        self,
        request: CompletionRequest[T],
    ) -> CompletionResult[T]:
        key = repr(request)
        if key not in self._cache:
            self._cache[key] = await self._engine.process_async(request)
        return self._cache[key]  # type: ignore[return-value]

