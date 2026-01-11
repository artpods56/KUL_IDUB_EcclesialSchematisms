import pandas as pd
import pytest
from notarius.application.services.scoring import (
    SimilarityEvaluationService,
    ClassificationEvaluationService,
    NormalizedLevenshteinDistanceScorer,
    ExactMatchScorer,
    SimilarityMetrics,
    ClassificationMetrics,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    """Sample DataFrame with various match scenarios."""
    return pd.DataFrame(
        [
            {
                "parish_a": "Kraków",
                "parish_b": "Kraków",
                "deanery_a": "Urbis",
                "deanery_b": "Urbis.",
                "dedication_a": "St. Mark",
                "dedication_b": "St. Luke",
                "building_material_a": "murata",
                "building_material_b": "murata",
            },
            {
                "parish_a": "Warszawa",
                "parish_b": "Kraków",
                "deanery_a": "Centrum",
                "deanery_b": "Centrum",
                "dedication_a": None,
                "dedication_b": None,
                "building_material_a": "lignea",
                "building_material_b": "murata",
            },
        ]
    )


@pytest.fixture
def confusion_matrix_df():
    """DataFrame for testing all confusion matrix scenarios.

    Column order convention: (gt=field_a, pred=field_b)
    """
    return pd.DataFrame(
        [
            # Perfect match: gt="A", pred="A" -> TP
            {"field_a": "A", "field_b": "A"},
            # Wrong value: gt="A", pred="B" -> FP + FN
            {"field_a": "A", "field_b": "B"},
            # Miss: gt="A", pred="" -> FN
            {"field_a": "A", "field_b": ""},
            # Hallucination: gt="", pred="B" -> FP
            {"field_a": "", "field_b": "B"},
            # Both empty: gt="", pred="" -> TN
            {"field_a": "", "field_b": ""},
        ]
    )


# --------------------------------------------------------------------------
# SimilarityEvaluationService Tests
# --------------------------------------------------------------------------


class TestSimilarityEvaluationService:
    def test_evaluate_field_returns_similarity_metrics(self, sample_df):
        scorer = NormalizedLevenshteinDistanceScorer()
        service = SimilarityEvaluationService(scorer)

        # Column order is (gt, pred)
        metrics = service.evaluate_field(sample_df, ("parish_a", "parish_b"))

        assert isinstance(metrics, SimilarityMetrics)
        assert metrics.field == "parish"
        assert 0.0 <= metrics.average_similarity <= 1.0
        assert 0.0 <= metrics.min_similarity <= 1.0
        assert 0.0 <= metrics.max_similarity <= 1.0
        assert metrics.sample_count == 2

    def test_evaluate_all_fields(self, sample_df):
        scorer = NormalizedLevenshteinDistanceScorer()
        service = SimilarityEvaluationService(scorer)

        # Column order is (gt, pred)
        columns = {
            "parish": ("parish_a", "parish_b"),
            "deanery": ("deanery_a", "deanery_b"),
        }
        metrics_list = service.evaluate(sample_df, columns)

        assert len(metrics_list) == 2
        assert all(isinstance(m, SimilarityMetrics) for m in metrics_list)

    def test_perfect_match_similarity(self):
        df = pd.DataFrame([{"gt": "hello", "pred": "hello"}])
        scorer = NormalizedLevenshteinDistanceScorer()
        service = SimilarityEvaluationService(scorer)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.average_similarity == 1.0
        assert metrics.max_similarity == 1.0
        assert metrics.min_similarity == 1.0

    def test_complete_mismatch_similarity(self):
        df = pd.DataFrame([{"gt": "abc", "pred": "xyz"}])
        scorer = NormalizedLevenshteinDistanceScorer()
        service = SimilarityEvaluationService(scorer)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.average_similarity < 0.5


# --------------------------------------------------------------------------
# ClassificationEvaluationService Tests
# --------------------------------------------------------------------------


class TestClassificationEvaluationService:
    def test_evaluate_field_returns_classification_metrics(self, sample_df):
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        # Column order is (gt, pred)
        metrics = service.evaluate_field(sample_df, ("parish_a", "parish_b"))

        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.field == "parish"
        assert metrics.tp >= 0
        assert metrics.fp >= 0
        assert metrics.fn >= 0
        assert metrics.tn >= 0

    def test_confusion_matrix_semantics(self, confusion_matrix_df):
        """Test that confusion matrix is computed with correct IE semantics.

        Column order is (gt, pred) = (field_a, field_b).
        """
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(confusion_matrix_df, ("field_a", "field_b"))

        # Expected based on the 5 rows (gt=field_a, pred=field_b):
        # Row 0: gt="A", pred="A" -> Perfect match -> TP=1
        # Row 1: gt="A", pred="B" -> Wrong value -> FP=1, FN=1
        # Row 2: gt="A", pred="" -> Miss -> FN=1
        # Row 3: gt="", pred="B" -> Hallucination -> FP=1
        # Row 4: gt="", pred="" -> Both empty -> TN=1
        assert metrics.tp == 1, "Perfect match should count as TP"
        assert metrics.fp == 2, "Wrong value + hallucination should count as FP"
        assert metrics.fn == 2, "Wrong value + miss should count as FN"
        assert metrics.tn == 1, "Both empty should count as TN"

    def test_perfect_match_row(self):
        """Perfect match: pred exists, gt exists, and they're equal."""
        df = pd.DataFrame([{"gt": "hello", "pred": "hello"}])
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.tp == 1
        assert metrics.fp == 0
        assert metrics.fn == 0
        assert metrics.tn == 0

    def test_wrong_value_row(self):
        """Wrong value: pred exists, gt exists, but they're different.

        This should count as BOTH FP (produced wrong value) and FN (missed correct value).
        """
        df = pd.DataFrame([{"gt": "hello", "pred": "world"}])
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.tp == 0
        assert metrics.fp == 1, "Wrong value should count as FP"
        assert metrics.fn == 1, "Wrong value should also count as FN"
        assert metrics.tn == 0

    def test_hallucination_row(self):
        """Hallucination: pred exists but gt is empty."""
        df = pd.DataFrame([{"gt": "", "pred": "hello"}])
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.tp == 0
        assert metrics.fp == 1, "Hallucination should count as FP"
        assert metrics.fn == 0
        assert metrics.tn == 0

    def test_miss_row(self):
        """Miss: gt exists but pred is empty."""
        df = pd.DataFrame([{"gt": "world", "pred": ""}])
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 1, "Miss should count as FN"
        assert metrics.tn == 0

    def test_both_empty_row(self):
        """Both empty: neither pred nor gt exists."""
        df = pd.DataFrame([{"gt": "", "pred": ""}])
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 0
        assert metrics.tn == 1, "Both empty should count as TN"

    def test_null_values_treated_as_empty(self):
        """None values should be treated as empty (absent)."""
        df = pd.DataFrame([{"gt": None, "pred": None}])
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.tn == 1, "None values should be treated as empty (TN)"

    def test_precision_recall_f1_properties(self, confusion_matrix_df):
        """Test that precision, recall, and F1 are computed correctly."""
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(confusion_matrix_df, ("field_a", "field_b"))

        # tp=1, fp=2, fn=2, tn=1
        # precision = tp / (tp + fp) = 1 / 3 ≈ 0.333
        # recall = tp / (tp + fn) = 1 / 3 ≈ 0.333
        # f1 = 2 * p * r / (p + r) ≈ 0.333
        assert abs(metrics.precision - 0.333) < 0.01
        assert abs(metrics.recall - 0.333) < 0.01
        assert abs(metrics.f1 - 0.333) < 0.01

    def test_accuracy_with_tn(self):
        """Test that accuracy includes TN in the calculation."""
        # 3 TN, 1 TP, 0 FP, 0 FN -> accuracy = (1+3)/(1+3+0+0) = 1.0
        df = pd.DataFrame(
            [
                {"gt": "match", "pred": "match"},  # TP
                {"gt": "", "pred": ""},  # TN
                {"gt": "", "pred": ""},  # TN
                {"gt": "", "pred": ""},  # TN
            ]
        )
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        metrics = service.evaluate_field(df, ("gt", "pred"))

        assert metrics.tp == 1
        assert metrics.tn == 3
        assert metrics.accuracy == 1.0

    def test_threshold_affects_correctness(self):
        """Test that threshold parameter affects what counts as correct."""
        # Levenshtein similarity between "hello" and "hallo" is high (~0.8)
        df = pd.DataFrame([{"gt": "hello", "pred": "hallo"}])
        scorer = NormalizedLevenshteinDistanceScorer()

        # With strict threshold (1.0), this is wrong
        strict_service = ClassificationEvaluationService(scorer, threshold=1.0)
        strict_metrics = strict_service.evaluate_field(df, ("gt", "pred"))
        assert strict_metrics.tp == 0
        assert strict_metrics.fp == 1
        assert strict_metrics.fn == 1

        # With relaxed threshold (0.7), this is correct
        relaxed_service = ClassificationEvaluationService(scorer, threshold=0.7)
        relaxed_metrics = relaxed_service.evaluate_field(df, ("gt", "pred"))
        assert relaxed_metrics.tp == 1
        assert relaxed_metrics.fp == 0
        assert relaxed_metrics.fn == 0

    def test_evaluate_multiple_fields(self, sample_df):
        """Test evaluating multiple fields at once."""
        scorer = ExactMatchScorer()
        service = ClassificationEvaluationService(scorer, threshold=1.0)

        # Column order is (gt, pred)
        columns = {
            "parish": ("parish_a", "parish_b"),
            "deanery": ("deanery_a", "deanery_b"),
            "building_material": ("building_material_a", "building_material_b"),
        }
        metrics_list = service.evaluate(sample_df, columns)

        assert len(metrics_list) == 3
        assert all(isinstance(m, ClassificationMetrics) for m in metrics_list)
        fields = {m.field for m in metrics_list}
        assert fields == {"parish", "deanery", "building_material"}
