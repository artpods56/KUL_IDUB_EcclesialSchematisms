import abc
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import final, override

import numpy as np
import pandas as pd

from notarius_schematisms.evaluation.protocol import Scorer

ColumnPairs = Iterable[tuple[str, str]] | Mapping[str, tuple[str, str]]


def _iter_column_pairs(columns_to_compare: ColumnPairs) -> Iterable[tuple[str, str]]:
    if isinstance(columns_to_compare, Mapping):
        return columns_to_compare.values()
    return columns_to_compare


@dataclass(frozen=True)
class FieldEvaluationMetrics:
    field: str


@dataclass(frozen=True)
class DistanceMetrics(FieldEvaluationMetrics):
    average_distance: float
    max_distance: float
    min_distance: float


@dataclass(frozen=True)
class SimilarityMetrics(FieldEvaluationMetrics):
    average_similarity: float
    max_similarity: float
    min_similarity: float
    sample_count: int = 0

    @staticmethod
    def from_distance_metrics(metrics: DistanceMetrics, sample_count: int = 0):
        return SimilarityMetrics(
            field=metrics.field,
            average_similarity=1 - metrics.average_distance,
            max_similarity=1 - metrics.max_distance,
            min_similarity=1 - metrics.min_distance,
            sample_count=sample_count,
        )


@dataclass(frozen=True)
class ClassificationMetrics(FieldEvaluationMetrics):
    tp: int
    fp: int
    fn: int
    tn: int = 0

    @property
    def precision(self) -> float:
        val = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        return round(val, 3)

    @property
    def recall(self) -> float:
        val = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return round(val, 3)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        val = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(val, 3)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        val = (self.tp + self.tn) / total if total else 1.0
        return round(val, 3)


@dataclass(frozen=True)
class MetricAggregate[T: FieldEvaluationMetrics](abc.ABC):
    metrics: list[T]

    def __iter__(self):
        return iter(self.metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    def __getitem__(self, index: int) -> T:
        return self.metrics[index]

    @abc.abstractmethod
    def to_dataframe(self) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ClassificationAggregate(MetricAggregate[ClassificationMetrics]):

    @override
    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "field": m.field,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "accuracy": m.accuracy,
            }
            for m in self.metrics
        ]

        df = pd.DataFrame(rows)

        avg_row = {
            "field": "Average",
            "precision": df["precision"].mean(),
            "recall": df["recall"].mean(),
            "f1": df["f1"].mean(),
            "accuracy": df["accuracy"].mean(),
        }

        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        return df


@dataclass(frozen=True)
class SimilarityAggregate(MetricAggregate[SimilarityMetrics]):

    @override
    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "field": m.field,
                "average_similarity": m.average_similarity,
                "max_similarity": m.max_similarity,
                "min_similarity": m.min_similarity,
            }
            for m in self.metrics
        ]

        df = pd.DataFrame(rows)

        avg_row = {
            "field": "Average",
            "average_similarity": df["average_similarity"].mean(),
            "max_similarity": df["max_similarity"].mean(),
            "min_similarity": df["min_similarity"].mean(),
        }

        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        return df


@final
class SimilarityEvaluationService:
    """Evaluates text similarity for source dataset generation.

    This service computes aggregate similarity metrics (avg, min, max) for
    comparing generated text against expected text. Appropriate for source
    dataset evaluation where you want to measure how similar generated Latin
    text is to the expected source.
    """

    def __init__(self, scorer: Scorer):
        self._scorer = scorer

    def evaluate_field(
        self, df: pd.DataFrame, columns: tuple[str, str]
    ) -> SimilarityMetrics:
        """Compute similarity metrics for a column pair.

        Args:
            df: DataFrame with data
            columns: Tuple of (gt_column, pred_column)
        """
        col_gt, col_pred = columns
        scores = [self._scorer.score(a, b) for a, b in zip(df[col_gt], df[col_pred])]

        field_name = col_gt.removesuffix("_a").removesuffix("_gt").removesuffix("_b")

        return SimilarityMetrics(
            field=field_name,
            average_similarity=float(np.mean(scores)),
            max_similarity=float(np.max(scores)),
            min_similarity=float(np.min(scores)),
            sample_count=len(scores),
        )

    def evaluate(self, df: pd.DataFrame, columns_to_compare: ColumnPairs) -> SimilarityAggregate:
        """Evaluate all fields and return list of similarity metrics."""
        return SimilarityAggregate(
            metrics=[
                self.evaluate_field(df, columns)
                for columns in _iter_column_pairs(columns_to_compare)
            ]
        )


@final
class ClassificationEvaluationService:
    """Evaluates extraction quality using confusion matrix metrics.

    This service computes precision, recall, and F1 with correct Information
    Extraction semantics:
    - TP: Ground truth exists, prediction exists, and prediction is correct
    - FP: Prediction exists but GT doesn't (hallucination) OR both exist but incorrect
    - FN: GT exists but prediction doesn't (miss) OR both exist but incorrect
    - TN: Neither GT nor prediction exists

    Key: incorrect-but-present predictions count as BOTH FP and FN.
    """

    def __init__(self, scorer: Scorer, threshold: float = 1.0):
        self._scorer = scorer
        self._threshold = threshold

    def evaluate_field(
        self, df: pd.DataFrame, columns: tuple[str, str]
    ) -> ClassificationMetrics:
        """Compute confusion matrix metrics for a column pair."""
        col_gt, col_pred = columns

        # Presence detection (binary)
        pred_present = df[col_pred].notna() & (df[col_pred] != "")
        gt_present = df[col_gt].notna() & (df[col_gt] != "")

        # Correctness (conditional on both present)
        scores = pd.Series(
            [self._scorer.score(a, b) for a, b in zip(df[col_pred], df[col_gt])],
            index=df.index,
        )
        correct = scores >= self._threshold

        # Confusion matrix (standard IE semantics)
        # Key insight: incorrect-but-present counts as BOTH FP and FN
        tp = int((gt_present & pred_present & correct).sum())
        fp = int(
            (
                (~gt_present & pred_present) | (gt_present & pred_present & ~correct)
            ).sum()
        )
        fn = int(
            (
                (gt_present & ~pred_present) | (gt_present & pred_present & ~correct)
            ).sum()
        )
        tn = int((~gt_present & ~pred_present).sum())

        field_name = col_gt.removesuffix("_a").removesuffix("_gt").removesuffix("_b")
        return ClassificationMetrics(
            field=field_name,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
        )

    def evaluate(
        self, df: pd.DataFrame, columns_to_compare: ColumnPairs
    ) -> ClassificationAggregate:
        """Evaluate all fields and return list of classification metrics."""
        return ClassificationAggregate(
            [
                self.evaluate_field(df, columns)
                for columns in _iter_column_pairs(columns_to_compare)
            ]
        )
