from notarius.application.services.scoring.protocol import (
    Scorer,
    StylingStrategy,
    EvaluationService,
)
from notarius.application.services.scoring.levenshtein import (
    NormalizedLevenshteinDistanceScorer,
)
from notarius.application.services.scoring.exact_match import ExactMatchScorer
from notarius.application.services.scoring.styling import (
    ExactMatchStyling,
    GradientStyling,
    CellComparisonStyler,
)
from notarius.application.services.scoring.evaluation import (
    FieldEvaluationMetrics,
    SimilarityMetrics,
    ClassificationMetrics,
    SimilarityEvaluationService,
    ClassificationEvaluationService,
)

__all__ = [
    # Protocols
    "Scorer",
    "StylingStrategy",
    "EvaluationService",
    # Scorers
    "NormalizedLevenshteinDistanceScorer",
    "ExactMatchScorer",
    # Styling
    "ExactMatchStyling",
    "GradientStyling",
    "CellComparisonStyler",
    # Metrics dataclasses
    "FieldEvaluationMetrics",
    "SimilarityMetrics",
    "ClassificationMetrics",
    # Evaluation services
    "SimilarityEvaluationService",
    "ClassificationEvaluationService",
]
