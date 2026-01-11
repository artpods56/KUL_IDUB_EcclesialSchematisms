import pytest
from notarius.application.services.scoring import (
    NormalizedLevenshteinDistanceScorer,
    ExactMatchScorer,
)


class TestLevenshteinScorer:
    def test_exact_match(self):
        scorer = NormalizedLevenshteinDistanceScorer()
        assert scorer.score("test", "test") == 1.0

    def test_both_empty(self):
        scorer = NormalizedLevenshteinDistanceScorer()
        assert scorer.score(None, None) == 1.0
        assert scorer.score("", "") == 1.0
        assert scorer.score("  ", "  ") == 1.0

    def test_one_empty(self):
        scorer = NormalizedLevenshteinDistanceScorer()
        assert scorer.score("test", None) == 0.0
        assert scorer.score(None, "test") == 0.0
        assert scorer.score("test", "") == 0.0

    def test_partial_match(self):
        scorer = NormalizedLevenshteinDistanceScorer()
        score = scorer.score("test", "testing")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        scorer = NormalizedLevenshteinDistanceScorer()
        score = scorer.score("abc", "xyz")
        assert score == 0.0

    def test_whitespace_stripped(self):
        scorer = NormalizedLevenshteinDistanceScorer()
        assert scorer.score("  test  ", "test") == 1.0

    def test_name(self):
        scorer = NormalizedLevenshteinDistanceScorer()
        assert scorer.name == "levenshtein"


class TestExactMatchScorer:
    def test_exact_match(self):
        scorer = ExactMatchScorer()
        assert scorer.score("test", "test") == 1.0

    def test_both_empty(self):
        scorer = ExactMatchScorer()
        assert scorer.score(None, None) == 1.0
        assert scorer.score("", "") == 1.0

    def test_off_by_one(self):
        scorer = ExactMatchScorer()
        assert scorer.score("Boris", "Boris.") == 0.5
        assert scorer.score("test", "tests") == 0.5

    def test_no_match(self):
        scorer = ExactMatchScorer()
        assert scorer.score("abc", "xyz") == 0.0
        assert scorer.score("test", "testing") == 0.0

    def test_one_empty(self):
        scorer = ExactMatchScorer()
        # Empty strings match each other (1.0)
        # But empty vs non-empty is 0.0 (more than edit distance 1)
        assert scorer.score("test", "") == 0.0
        assert scorer.score("", "test") == 0.0

    def test_whitespace_stripped(self):
        scorer = ExactMatchScorer()
        assert scorer.score("  test  ", "test") == 1.0

    def test_name(self):
        scorer = ExactMatchScorer()
        assert scorer.name == "exact_match"
