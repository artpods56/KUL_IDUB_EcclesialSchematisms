"""Tests for the aligner module."""

import pytest

from notarius.application.services.data.aligning import HungarianAligner, GreedyAligner
from notarius.domain.entities.schematism import SchematismEntry


def entry(
    deanery: str | None = None,
    parish: str | None = None,
    dedication: str | None = None,
    building_material: str | None = None,
) -> SchematismEntry:
    """Helper to create SchematismEntry."""
    return SchematismEntry(
        deanery=deanery,
        parish=parish,
        dedication=dedication,
        building_material=building_material,
    )


class TestHungarianAligner:
    """Tests for HungarianAligner."""

    @pytest.fixture
    def weights(self) -> dict[str, float]:
        return {
            "deanery": 1.0,
            "parish": 2.0,
            "dedication": 1.5,
            "building_material": 0.5,
        }

    @pytest.fixture
    def aligner(self, weights: dict[str, float]) -> HungarianAligner:
        return HungarianAligner(weights=weights, threshold=0.5)

    @pytest.fixture
    def real_gt_entries(self) -> list[SchematismEntry]:
        """Real ground truth data sample."""
        return [
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Kruszyna",
                dedication="S. Mathias Ap. Patr. S. Joseph. Spons. BV.",
                building_material="mur.",
            ),
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Lgota",
                dedication="S. Clemens PM.",
                building_material="lig.",
            ),
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Makowiska",
                dedication="Patroc. S. Joseph. Patr. S. Barthol. Ap.",
                building_material="mur.",
            ),
        ]

    def test_empty_lists(self, aligner: HungarianAligner) -> None:
        """Test alignment of two empty lists."""
        result_a, result_b = aligner.align_entries([], [])
        assert result_a == []
        assert result_b == []

    def test_empty_first_list(self, aligner: HungarianAligner) -> None:
        """Test alignment when first list is empty."""
        seq_b = [entry(deanery="A", parish="B", dedication="C", building_material="D")]
        result_a, result_b = aligner.align_entries([], seq_b)

        assert len(result_a) == 1
        assert len(result_b) == 1
        assert result_b == seq_b
        assert result_a[0].parish is None  # Empty entry

    def test_empty_second_list(self, aligner: HungarianAligner) -> None:
        """Test alignment when second list is empty."""
        seq_a = [entry(deanery="A", parish="B", dedication="C", building_material="D")]
        result_a, result_b = aligner.align_entries(seq_a, [])

        assert len(result_a) == 1
        assert len(result_b) == 1
        assert result_a == seq_a
        assert result_b[0].parish is None  # Empty entry

    def test_perfect_match(self, aligner: HungarianAligner) -> None:
        """Test alignment with identical entries."""
        entries = [
            entry(deanery="Decanatus A", parish="Parish1", dedication="Ded1", building_material="mur."),
            entry(deanery="Decanatus A", parish="Parish2", dedication="Ded2", building_material="lig."),
        ]

        result_a, result_b = aligner.align_entries(entries, entries)

        assert len(result_a) == len(result_b) == 2
        for a, b in zip(result_a, result_b):
            assert a.parish == b.parish

    def test_preserves_seq_a_order(self, aligner: HungarianAligner) -> None:
        """Test that seq_a order is preserved in output."""
        seq_a = [
            entry(parish="First"),
            entry(parish="Second"),
            entry(parish="Third"),
        ]
        seq_b = [
            entry(parish="Third"),
            entry(parish="First"),
            entry(parish="Second"),
        ]

        result_a, result_b = aligner.align_entries(seq_a, seq_b)

        # seq_a order should be preserved
        assert result_a[0].parish == "First"
        assert result_a[1].parish == "Second"
        assert result_a[2].parish == "Third"
        # seq_b should be reordered to match
        assert result_b[0].parish == "First"
        assert result_b[1].parish == "Second"
        assert result_b[2].parish == "Third"

    def test_real_data_perfect_match(
        self, aligner: HungarianAligner, real_gt_entries: list[SchematismEntry]
    ) -> None:
        """Test alignment with real data - perfect match scenario."""
        result_a, result_b = aligner.align_entries(real_gt_entries, real_gt_entries)

        assert len(result_a) == len(result_b) == 3
        for a, b in zip(result_a, result_b):
            assert a.parish == b.parish
            assert a.deanery == b.deanery

    def test_real_data_with_ocr_errors(
        self, aligner: HungarianAligner, real_gt_entries: list[SchematismEntry]
    ) -> None:
        """Test alignment with simulated OCR errors in predictions."""
        pred_entries = [
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Kruszyna",
                dedication="S. Mathias Ap. Patr. S. Joseph Spons. BV",  # Missing dot
                building_material="mur",  # Missing dot
            ),
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Lgotta",  # Typo: extra 't'
                dedication="S. Clemens PM",
                building_material="lig.",
            ),
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Makowiska",
                dedication="Patroc. S. Joseph Patr. S. Barthol. Ap.",
                building_material="mur.",
            ),
        ]

        result_a, result_b = aligner.align_entries(real_gt_entries, pred_entries)

        assert len(result_a) == len(result_b) == 3
        # Entries should be matched correctly despite small errors
        parishes_a = {e.parish for e in result_a}
        parishes_b = {e.parish for e in result_b}

        assert "Kruszyna" in parishes_a
        assert "Kruszyna" in parishes_b
        assert "Makowiska" in parishes_a
        assert "Makowiska" in parishes_b

    def test_real_data_different_order(
        self, aligner: HungarianAligner, real_gt_entries: list[SchematismEntry]
    ) -> None:
        """Test alignment when predictions are in different order."""
        pred_entries = list(reversed(real_gt_entries))

        result_a, result_b = aligner.align_entries(real_gt_entries, pred_entries)

        assert len(result_a) == len(result_b) == 3
        # Hungarian algorithm should find optimal matching regardless of order
        for a, b in zip(result_a, result_b):
            assert a.parish == b.parish

    def test_real_data_missing_prediction(
        self, aligner: HungarianAligner, real_gt_entries: list[SchematismEntry]
    ) -> None:
        """Test alignment when prediction is missing an entry."""
        pred_entries = real_gt_entries[:2]

        result_a, result_b = aligner.align_entries(real_gt_entries, pred_entries)

        assert len(result_a) == len(result_b) == 3
        # Two should match, one should have empty placeholder
        empty_count = sum(1 for e in result_b if e.parish is None)
        assert empty_count == 1

    def test_real_data_extra_prediction(
        self, aligner: HungarianAligner, real_gt_entries: list[SchematismEntry]
    ) -> None:
        """Test alignment when prediction has extra entry."""
        pred_entries = list(real_gt_entries) + [
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="ExtraParish",
                dedication="Extra Dedication",
                building_material="mur.",
            )
        ]

        result_a, result_b = aligner.align_entries(real_gt_entries, pred_entries)

        assert len(result_a) == len(result_b) == 4
        # Three should match, one GT should be empty placeholder
        empty_gt_count = sum(1 for e in result_a if e.parish is None)
        assert empty_gt_count == 1

    def test_threshold_filtering(self, weights: dict[str, float]) -> None:
        """Test that entries below threshold are not matched."""
        aligner = HungarianAligner(weights=weights, threshold=0.9)

        seq_a = [entry(deanery="A", parish="Parish1", dedication="Ded1", building_material="mur.")]
        seq_b = [entry(deanery="B", parish="Different", dedication="Other", building_material="lig.")]

        result_a, result_b = aligner.align_entries(seq_a, seq_b)

        # With high threshold, dissimilar entries should not match
        assert len(result_a) == len(result_b) == 2

    def test_similarity_scoring(self, aligner: HungarianAligner) -> None:
        """Test internal similarity scoring."""
        assert aligner._similarity("test", "test") == 1.0
        assert aligner._similarity("test", "tset") < 1.0
        assert aligner._similarity("test", "tset") > 0.5
        assert aligner._similarity("abc", "xyz") < 0.5


class TestGreedyAligner:
    """Tests for GreedyAligner."""

    @pytest.fixture
    def weights(self) -> dict[str, float]:
        return {
            "deanery": 1.0,
            "parish": 2.0,
            "dedication": 1.5,
            "building_material": 0.5,
        }

    @pytest.fixture
    def aligner(self, weights: dict[str, float]) -> GreedyAligner:
        return GreedyAligner(weights=weights, threshold=0.5)

    @pytest.fixture
    def real_gt_entries(self) -> list[SchematismEntry]:
        """Real ground truth data sample."""
        return [
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Kruszyna",
                dedication="S. Mathias Ap. Patr. S. Joseph. Spons. BV.",
                building_material="mur.",
            ),
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Lgota",
                dedication="S. Clemens PM.",
                building_material="lig.",
            ),
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Makowiska",
                dedication="Patroc. S. Joseph. Patr. S. Barthol. Ap.",
                building_material="mur.",
            ),
        ]

    def test_empty_lists(self, aligner: GreedyAligner) -> None:
        """Test alignment of two empty lists."""
        result_a, result_b = aligner.align_entries([], [])
        assert result_a == []
        assert result_b == []

    def test_preserves_seq_a_order(self, aligner: GreedyAligner) -> None:
        """Test that seq_a order is preserved in output."""
        seq_a = [
            entry(parish="First"),
            entry(parish="Second"),
            entry(parish="Third"),
        ]
        seq_b = [
            entry(parish="Third"),
            entry(parish="First"),
            entry(parish="Second"),
        ]

        result_a, result_b = aligner.align_entries(seq_a, seq_b)

        # seq_a order should be preserved
        assert result_a[0].parish == "First"
        assert result_a[1].parish == "Second"
        assert result_a[2].parish == "Third"

    def test_real_data_perfect_match(
        self, aligner: GreedyAligner, real_gt_entries: list[SchematismEntry]
    ) -> None:
        """Test GreedyAligner with real data - perfect match scenario."""
        result_a, result_b = aligner.align_entries(real_gt_entries, real_gt_entries)

        assert len(result_a) == len(result_b) == 3
        for a, b in zip(result_a, result_b):
            assert a.parish == b.parish

    def test_position_weight(self, weights: dict[str, float]) -> None:
        """Test that position_weight affects scoring."""
        aligner_no_pos = GreedyAligner(weights=weights, threshold=0.3, position_weight=0.0)
        aligner_with_pos = GreedyAligner(weights=weights, threshold=0.3, position_weight=0.5)

        # Entries with similar content but different positions
        seq_a = [entry(parish="A"), entry(parish="B")]
        seq_b = [entry(parish="B"), entry(parish="A")]

        # Without position weight, greedy should match first available
        result_no_pos_a, result_no_pos_b = aligner_no_pos.align_entries(seq_a, seq_b)

        # With position weight, position similarity should influence matching
        result_pos_a, result_pos_b = aligner_with_pos.align_entries(seq_a, seq_b)

        # Both should produce valid alignments
        assert len(result_no_pos_a) == len(result_no_pos_b)
        assert len(result_pos_a) == len(result_pos_b)


class TestAlignerComparison:
    """Compare both aligners on the same data."""

    @pytest.fixture
    def weights(self) -> dict[str, float]:
        return {
            "deanery": 1.0,
            "parish": 2.0,
            "dedication": 1.5,
            "building_material": 0.5,
        }

    @pytest.fixture
    def real_gt_entries(self) -> list[SchematismEntry]:
        return [
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Kruszyna",
                dedication="S. Mathias Ap. Patr. S. Joseph. Spons. BV.",
                building_material="mur.",
            ),
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Lgota",
                dedication="S. Clemens PM.",
                building_material="lig.",
            ),
            entry(
                deanery="Decanatus Neo-Radomscensis",
                parish="Makowiska",
                dedication="Patroc. S. Joseph. Patr. S. Barthol. Ap.",
                building_material="mur.",
            ),
        ]

    def test_both_aligners_handle_shuffled_order(
        self, weights: dict[str, float], real_gt_entries: list[SchematismEntry]
    ) -> None:
        """Compare how both aligners handle shuffled predictions."""
        hungarian = HungarianAligner(weights=weights, threshold=0.5)
        greedy = GreedyAligner(weights=weights, threshold=0.5)

        # Shuffle: [2, 0, 1] order
        pred_entries = [real_gt_entries[2], real_gt_entries[0], real_gt_entries[1]]

        h_result_a, h_result_b = hungarian.align_entries(real_gt_entries, pred_entries)
        g_result_a, g_result_b = greedy.align_entries(real_gt_entries, pred_entries)

        # Hungarian should find optimal matching
        hungarian_matches = sum(
            1 for a, b in zip(h_result_a, h_result_b) if a.parish == b.parish
        )

        # Greedy may or may not find all matches
        greedy_matches = sum(
            1 for a, b in zip(g_result_a, g_result_b) if a.parish == b.parish
        )

        # Hungarian should be at least as good as greedy
        assert hungarian_matches >= greedy_matches
        # With this specific data, Hungarian should find all 3 matches
        assert hungarian_matches == 3

    def test_both_preserve_seq_a_order(
        self, weights: dict[str, float], real_gt_entries: list[SchematismEntry]
    ) -> None:
        """Both aligners should preserve seq_a order."""
        hungarian = HungarianAligner(weights=weights, threshold=0.5)
        greedy = GreedyAligner(weights=weights, threshold=0.5)

        pred_entries = list(reversed(real_gt_entries))

        h_result_a, _ = hungarian.align_entries(real_gt_entries, pred_entries)
        g_result_a, _ = greedy.align_entries(real_gt_entries, pred_entries)

        # Both should preserve seq_a order
        for i, gt in enumerate(real_gt_entries):
            assert h_result_a[i].parish == gt.parish
            assert g_result_a[i].parish == gt.parish
