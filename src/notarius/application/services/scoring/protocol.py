from typing import Protocol, TypeVar, runtime_checkable

import pandas as pd


T_Metrics = TypeVar("T_Metrics")


@runtime_checkable
class Scorer(Protocol):
    """Protocol for comparing two string values."""

    def score(self, a: str | None, b: str | None) -> float:
        """Return score between 0.0 and 1.0"""
        ...

    @property
    def name(self) -> str:
        """Scorer identifier for aggregation keys"""
        ...


class StylingStrategy(Protocol):
    """Protocol for styling scores."""

    def get_style(self, score: float) -> str:
        """Return CSS style string for a given score."""
        ...

    def get_neutral_style(self) -> str:
        """Return CSS style for neutral/empty cases (e.g., both values missing)."""
        ...


class EvaluationService(Protocol[T_Metrics]):
    """Protocol for evaluation services.

    Generic over the metrics type returned (SimilarityMetrics or ClassificationMetrics).
    """

    def evaluate_field(
        self, df: pd.DataFrame, columns: tuple[str, str]
    ) -> T_Metrics:
        """Evaluate a single field (column pair) and return metrics."""
        ...

    def evaluate(
        self, df: pd.DataFrame, columns_to_compare: dict[str, tuple[str, str]]
    ) -> list[T_Metrics]:
        """Evaluate all fields and return list of metrics."""
        ...
