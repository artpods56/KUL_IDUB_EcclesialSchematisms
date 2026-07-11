from notarius_schematisms.data.aligning import GreedyAligner
from notarius_schematisms.domain.models import SchematismEntry


def test_greedy_aligner_matches_similar_entries() -> None:
    aligner = GreedyAligner(weights={"parish": 1.0}, threshold=0.8)

    left, right = aligner.align_entries(
        [SchematismEntry(parish="Krakow")],
        [SchematismEntry(parish="Krakow")],
    )

    assert left[0].parish == right[0].parish == "Krakow"

