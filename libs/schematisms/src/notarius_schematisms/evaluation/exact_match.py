from typing import override

from rapidfuzz.distance import Levenshtein
from notarius_schematisms.evaluation.protocol import Scorer


class ExactMatchScorer(Scorer):

    @property
    @override
    def name(self) -> str:
        return "exact_match"

    @override
    def score(self, a: str | None, b: str | None) -> float:
        a_normalized = (a or "").strip()
        b_normalized = (b or "").strip()

        if a_normalized == b_normalized:
            return 1.0

        if Levenshtein.distance(a_normalized, b_normalized) == 1:
            return 0.5

        return 0.0
