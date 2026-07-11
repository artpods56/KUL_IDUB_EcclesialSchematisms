from typing import override

from rapidfuzz.distance import Levenshtein
from notarius_schematisms.evaluation.protocol import Scorer

import unicodedata


def normalize(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = " ".join(s.split())
    return s


class NormalizedLevenshteinDistanceScorer(Scorer):

    @property
    @override
    def name(self) -> str:
        return "levenshtein"

    @override
    def score(self, a: str | None, b: str | None) -> float:
        a_normalized = normalize(a)
        b_normalized = normalize(b)

        return Levenshtein.normalized_similarity(a_normalized, b_normalized)
