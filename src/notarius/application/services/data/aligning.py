from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import final, Protocol, override

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from notarius.domain.entities.schematism import SchematismEntry


class Aligner(Protocol):

    def align_entries(
        self, seq_a: Sequence[SchematismEntry], seq_b: Sequence[SchematismEntry]
    ) -> tuple[Sequence[SchematismEntry], Sequence[SchematismEntry]]: ...


@final
class HungarianAligner(Aligner):
    """Aligns seq_b to seq_a using the Hungarian algorithm, preserving seq_a's order."""

    def __init__(self, weights: dict[str, float], threshold: float = 0.5) -> None:
        self._weights = weights
        self._threshold = threshold

    @staticmethod
    def _similarity(a: str | None, b: str | None) -> float:
        a = a or ""
        b = b or ""
        if a == b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    def _score(self, entry_a: SchematismEntry, entry_b: SchematismEntry) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for field, weight in self._weights.items():
            val_a = getattr(entry_a, field, None)
            val_b = getattr(entry_b, field, None)
            weighted_sum += self._similarity(val_a, val_b) * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0

    def _build_score_matrix(
        self, seq_a: Sequence[SchematismEntry], seq_b: Sequence[SchematismEntry]
    ) -> NDArray[np.floating]:
        n_a, n_b = len(seq_a), len(seq_b)
        scores = np.zeros((n_a, n_b))
        for i, a in enumerate(seq_a):
            for j, b in enumerate(seq_b):
                scores[i, j] = self._score(a, b)
        return scores

    @staticmethod
    def _build_cost_matrix(scores: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert score matrix to square cost matrix for Hungarian algorithm."""

        n_a, n_b = scores.shape
        n_a: int
        n_b: int
        size = max(n_a, n_b)
        cost = np.zeros((size, size))
        cost[:n_a, :n_b] = -scores
        return cost

    def _extract_assignment(
        self,
        row_ind: NDArray[np.intp],
        col_ind: NDArray[np.intp],
        scores: NDArray[np.floating],
    ) -> tuple[dict[int, int], set[int]]:
        """Extract valid assignments above threshold from Hungarian result."""

        n_a, n_b = scores.shape
        n_a: int
        n_b: int
        a_to_b: dict[int, int] = {}
        matched_b: set[int] = set()

        valid = (row_ind < n_a) & (col_ind < n_b)
        for row, col in zip(row_ind[valid], col_ind[valid]):
            if scores[row, col] >= self._threshold:
                a_to_b[int(row)] = int(col)
                matched_b.add(int(col))

        return a_to_b, matched_b

    def _reconstruct(
        self,
        seq_a: Sequence[SchematismEntry],
        seq_b: Sequence[SchematismEntry],
        a_to_b: dict[int, int],
        matched_b: set[int],
    ) -> tuple[list[SchematismEntry], list[SchematismEntry]]:
        """Reconstruct aligned sequences preserving seq_a's order."""
        result_a: list[SchematismEntry] = []
        result_b: list[SchematismEntry] = []

        for i, entry_a in enumerate(seq_a):
            result_a.append(entry_a)
            if i in a_to_b:
                result_b.append(seq_b[a_to_b[i]])
            else:
                result_b.append(SchematismEntry())

        for j, entry_b in enumerate(seq_b):
            if j not in matched_b:
                result_a.append(SchematismEntry())
                result_b.append(entry_b)

        return result_a, result_b

    @override
    def align_entries(
        self, seq_a: Sequence[SchematismEntry], seq_b: Sequence[SchematismEntry]
    ) -> tuple[list[SchematismEntry], list[SchematismEntry]]:
        """Align seq_b to seq_a, preserving seq_a's original order."""
        n_a, n_b = len(seq_a), len(seq_b)

        if n_a == 0 and n_b == 0:
            return [], []
        if n_a == 0:
            return [SchematismEntry()] * n_b, list(seq_b)
        if n_b == 0:
            return list(seq_a), [SchematismEntry()] * n_a

        scores = self._build_score_matrix(seq_a, seq_b)
        cost = self._build_cost_matrix(scores)
        row_ind, col_ind = optimize.linear_sum_assignment(cost)
        a_to_b, matched_b = self._extract_assignment(row_ind, col_ind, scores)

        return self._reconstruct(seq_a, seq_b, a_to_b, matched_b)


@final
class GreedyAligner(Aligner):
    """Aligns seq_b to seq_a using greedy matching, preserving seq_a's order."""

    def __init__(
        self,
        weights: dict[str, float],
        threshold: float = 0.5,
        position_weight: float = 0.0,
    ) -> None:
        self._weights = weights
        self._threshold = threshold
        self._position_weight = position_weight

    @staticmethod
    def _similarity(a: str | None, b: str | None) -> float:
        a = a or ""
        b = b or ""
        if a == b:
            return 1.0
        return SequenceMatcher(None, a, b).ratio()

    def _score(self, entry_a: SchematismEntry, entry_b: SchematismEntry) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for field, weight in self._weights.items():
            val_a = getattr(entry_a, field, None)
            val_b = getattr(entry_b, field, None)
            weighted_sum += self._similarity(val_a, val_b) * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0

    def _score_with_position(
        self,
        entry_a: SchematismEntry,
        entry_b: SchematismEntry,
        pos_a: float,
        pos_b: float,
    ) -> float:
        base_score = self._score(entry_a, entry_b)
        if self._position_weight > 0:
            position_similarity = 1.0 - abs(pos_a - pos_b)
            return base_score + position_similarity * self._position_weight
        return base_score

    def _find_best_match(
        self,
        entry_a: SchematismEntry,
        pos_a: float,
        seq_b: Sequence[SchematismEntry],
        available: set[int],
        n_b: int,
    ) -> int | None:
        best_idx: int | None = None
        best_score = 0.0

        for j in available:
            pos_b = j / n_b if n_b > 1 else 0.0
            score = self._score_with_position(entry_a, seq_b[j], pos_a, pos_b)
            if score >= self._threshold and score > best_score:
                best_score = score
                best_idx = j

        return best_idx

    @override
    def align_entries(
        self, seq_a: Sequence[SchematismEntry], seq_b: Sequence[SchematismEntry]
    ) -> tuple[list[SchematismEntry], list[SchematismEntry]]:
        """Align seq_b to seq_a using greedy matching, preserving seq_a's order."""
        n_a, n_b = len(seq_a), len(seq_b)

        if n_a == 0 and n_b == 0:
            return [], []
        if n_a == 0:
            return [SchematismEntry()] * n_b, list(seq_b)
        if n_b == 0:
            return list(seq_a), [SchematismEntry()] * n_a

        result_a: list[SchematismEntry] = []
        result_b: list[SchematismEntry] = []
        available = set(range(n_b))

        for i, entry_a in enumerate(seq_a):
            pos_a = i / n_a if n_a > 1 else 0.0
            best_idx = self._find_best_match(entry_a, pos_a, seq_b, available, n_b)

            result_a.append(entry_a)
            if best_idx is not None:
                result_b.append(seq_b[best_idx])
                available.remove(best_idx)
            else:
                result_b.append(SchematismEntry())

        for j in sorted(available):
            result_a.append(SchematismEntry())
            result_b.append(seq_b[j])

        return result_a, result_b
