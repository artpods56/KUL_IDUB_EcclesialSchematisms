from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompiledRecipe:
    config: dict[str, Any]
    model_metadata: dict[str, Any] = field(
        default_factory=lambda: {"provider": "fake", "model": "deterministic"}
    )

    def extract(
        self,
        text: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "text": text,
            "metadata": metadata,
            "length": len(text or ""),
        }


class RecipeCompiler:
    """Bridge Studio recipe configuration to core runtime components."""

    def compile(self, config: dict[str, Any]) -> CompiledRecipe:
        return CompiledRecipe(config=config)

