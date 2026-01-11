import numpy as np
import pandas as pd
import pytest
from notarius.application.services.scoring import (
    ExactMatchStyling,
    GradientStyling,
    CellComparisonStyler,
    ExactMatchScorer,
    NormalizedLevenshteinDistanceScorer,
)


class TestExactMatchStyling:
    def test_match_color(self):
        styling = ExactMatchStyling()
        style = styling.get_style(1.0)
        assert "background-color" in style
        assert "#90EE90" in style  # Green

    def test_off_by_one_color(self):
        styling = ExactMatchStyling()
        style = styling.get_style(0.5)
        assert "background-color" in style
        assert "#FFFFE0" in style  # Yellow

    def test_mismatch_color(self):
        styling = ExactMatchStyling()
        style = styling.get_style(0.0)
        assert "background-color" in style
        assert "#FFB6C1" in style  # Red

    def test_custom_colors(self):
        styling = ExactMatchStyling(
            color_match="#00FF00",
            color_off_by_one="#FFFF00",
            color_mismatch="#FF0000",
        )
        assert "#00FF00" in styling.get_style(1.0)
        assert "#FFFF00" in styling.get_style(0.5)
        assert "#FF0000" in styling.get_style(0.0)

    def test_neutral_color(self):
        styling = ExactMatchStyling()
        style = styling.get_neutral_style()
        assert "background-color" in style
        assert "#D3D3D3" in style  # Light gray

    def test_custom_neutral_color(self):
        styling = ExactMatchStyling(color_neutral="#808080")
        assert "#808080" in styling.get_neutral_style()


class TestGradientStyling:
    def test_low_score(self):
        styling = GradientStyling()
        style = styling.get_style(0.0)
        assert "background-color" in style
        # Should be reddish (close to #FFB6C1)

    def test_mid_score(self):
        styling = GradientStyling()
        style = styling.get_style(0.5)
        assert "background-color" in style
        # Should be yellowish

    def test_high_score(self):
        styling = GradientStyling()
        style = styling.get_style(1.0)
        assert "background-color" in style
        # Should be greenish (close to #90EE90)

    def test_clamping_above_one(self):
        styling = GradientStyling()
        style = styling.get_style(1.5)
        assert "background-color" in style
        # Should be same as 1.0

    def test_clamping_below_zero(self):
        styling = GradientStyling()
        style = styling.get_style(-0.5)
        assert "background-color" in style
        # Should be same as 0.0

    def test_gradient_continuity(self):
        """Test that gradient changes smoothly."""
        styling = GradientStyling()
        styles = [styling.get_style(i / 10) for i in range(11)]
        # All should be unique colors (except perhaps edge cases)
        assert len(styles) == 11
        for style in styles:
            assert "background-color" in style

    def test_neutral_color(self):
        styling = GradientStyling()
        style = styling.get_neutral_style()
        assert "background-color" in style
        assert "#D3D3D3" in style  # Light gray

    def test_custom_neutral_color(self):
        styling = GradientStyling(color_neutral="#808080")
        assert "#808080" in styling.get_neutral_style()


class TestCellComparisonStyler:
    """Tests for CellComparisonStyler service."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing."""
        return pd.DataFrame(
            {
                "col_a": ["hello", "world", "test"],
                "col_b": ["hello", "word", "different"],
            }
        )

    @pytest.fixture
    def exact_match_styler(self):
        """Create a styler with ExactMatchScorer and ExactMatchStyling."""
        return CellComparisonStyler(
            scorer=ExactMatchScorer(), styling=ExactMatchStyling()
        )

    def test_applies_green_for_exact_match(self, sample_dataframe, exact_match_styler):
        """Styler should apply green for exact matches to both columns."""
        styled = exact_match_styler.style(
            sample_dataframe, columns_to_compare=[("col_a", "col_b")]
        )
        # Get the styles applied
        styles = styled._compute()
        # Check that the first row (exact match) has green on both columns
        assert "#90EE90" in styles.ctx[(0, 0)][0][1]  # col_a
        assert "#90EE90" in styles.ctx[(0, 1)][0][1]  # col_b

    def test_applies_yellow_for_off_by_one(self, sample_dataframe, exact_match_styler):
        """Styler should apply yellow for off-by-one matches to both columns."""
        styled = exact_match_styler.style(
            sample_dataframe, columns_to_compare=[("col_a", "col_b")]
        )
        styles = styled._compute()
        # "world" vs "word" is off-by-one (edit distance 1)
        assert "#FFFFE0" in styles.ctx[(1, 0)][0][1]  # col_a
        assert "#FFFFE0" in styles.ctx[(1, 1)][0][1]  # col_b

    def test_applies_red_for_mismatch(self, sample_dataframe, exact_match_styler):
        """Styler should apply red for complete mismatches to both columns."""
        styled = exact_match_styler.style(
            sample_dataframe, columns_to_compare=[("col_a", "col_b")]
        )
        styles = styled._compute()
        # "test" vs "different" is a complete mismatch
        assert "#FFB6C1" in styles.ctx[(2, 0)][0][1]  # col_a
        assert "#FFB6C1" in styles.ctx[(2, 1)][0][1]  # col_b

    def test_handles_nan_values(self, exact_match_styler):
        """Styler should handle NaN values gracefully."""
        df = pd.DataFrame(
            {
                "col_a": ["hello", np.nan, "test"],
                "col_b": [np.nan, "world", "test"],
            }
        )
        styled = exact_match_styler.style(df, columns_to_compare=[("col_a", "col_b")])
        styles = styled._compute()
        # Should not raise and should produce valid styles for both columns
        assert (0, 0) in styles.ctx  # col_a
        assert (0, 1) in styles.ctx  # col_b
        assert (1, 0) in styles.ctx
        assert (1, 1) in styles.ctx
        assert (2, 0) in styles.ctx
        assert (2, 1) in styles.ctx

    def test_applies_neutral_for_both_empty(self, exact_match_styler):
        """Styler should apply neutral gray when both values are empty."""
        df = pd.DataFrame(
            {
                "col_a": ["hello", np.nan, "", None],
                "col_b": ["hello", np.nan, "", None],
            }
        )
        styled = exact_match_styler.style(df, columns_to_compare=[("col_a", "col_b")])
        styles = styled._compute()
        # Row 0: exact match → green
        assert "#90EE90" in styles.ctx[(0, 0)][0][1]
        assert "#90EE90" in styles.ctx[(0, 1)][0][1]
        # Row 1: both NaN → neutral gray
        assert "#D3D3D3" in styles.ctx[(1, 0)][0][1]
        assert "#D3D3D3" in styles.ctx[(1, 1)][0][1]
        # Row 2: both empty string → neutral gray
        assert "#D3D3D3" in styles.ctx[(2, 0)][0][1]
        assert "#D3D3D3" in styles.ctx[(2, 1)][0][1]
        # Row 3: both None → neutral gray
        assert "#D3D3D3" in styles.ctx[(3, 0)][0][1]
        assert "#D3D3D3" in styles.ctx[(3, 1)][0][1]

    def test_applies_red_when_only_one_empty(self, exact_match_styler):
        """Styler should apply red (mismatch) when only one value is empty."""
        df = pd.DataFrame(
            {
                "col_a": ["hello", np.nan, "test"],
                "col_b": [np.nan, "world", ""],
            }
        )
        styled = exact_match_styler.style(df, columns_to_compare=[("col_a", "col_b")])
        styles = styled._compute()
        # Row 0: "hello" vs NaN → mismatch (red)
        assert "#FFB6C1" in styles.ctx[(0, 0)][0][1]
        assert "#FFB6C1" in styles.ctx[(0, 1)][0][1]
        # Row 1: NaN vs "world" → mismatch (red)
        assert "#FFB6C1" in styles.ctx[(1, 0)][0][1]
        assert "#FFB6C1" in styles.ctx[(1, 1)][0][1]
        # Row 2: "test" vs "" → mismatch (red)
        assert "#FFB6C1" in styles.ctx[(2, 0)][0][1]
        assert "#FFB6C1" in styles.ctx[(2, 1)][0][1]

    def test_skips_missing_columns(self, sample_dataframe, exact_match_styler):
        """Styler should skip column pairs not present in dataframe."""
        # Should not raise when columns don't exist
        styled = exact_match_styler.style(
            sample_dataframe, columns_to_compare=[("nonexistent_a", "nonexistent_b")]
        )
        styles = styled._compute()
        # No styles should be applied to existing columns
        assert len(styles.ctx) == 0

    def test_with_gradient_styling(self, sample_dataframe):
        """Styler should work with GradientStyling strategy."""
        styler = CellComparisonStyler(
            scorer=NormalizedLevenshteinDistanceScorer(), styling=GradientStyling()
        )
        styled = styler.style(
            sample_dataframe, columns_to_compare=[("col_a", "col_b")]
        )
        styles = styled._compute()
        # Should have styles applied to both col_a and col_b cells
        for row in range(3):
            assert (row, 0) in styles.ctx  # col_a
            assert (row, 1) in styles.ctx  # col_b
            # All should have some style applied (either full CSS or parsed value)
            for col in range(2):
                style_value = styles.ctx[(row, col)][0][1]
                assert "#" in style_value or "background-color" in style_value

    def test_returns_styler_object(self, sample_dataframe, exact_match_styler):
        """Styler should return a pandas Styler object."""
        result = exact_match_styler.style(
            sample_dataframe, columns_to_compare=[("col_a", "col_b")]
        )
        assert isinstance(result, pd.io.formats.style.Styler)

    def test_multiple_column_pairs(self):
        """Styler should handle multiple column pairs and color both columns in each pair."""
        df = pd.DataFrame(
            {
                "a1": ["hello", "foo"],
                "b1": ["hello", "bar"],
                "a2": ["test", "baz"],
                "b2": ["test", "qux"],
            }
        )
        styler = CellComparisonStyler(
            scorer=ExactMatchScorer(), styling=ExactMatchStyling()
        )
        styled = styler.style(
            df, columns_to_compare=[("a1", "b1"), ("a2", "b2")]
        )
        styles = styled._compute()
        # All columns in pairs should have styles
        # Row 0: both matches (green) - a1, b1, a2, b2 all green
        assert "#90EE90" in styles.ctx[(0, 0)][0][1]  # a1 column
        assert "#90EE90" in styles.ctx[(0, 1)][0][1]  # b1 column
        assert "#90EE90" in styles.ctx[(0, 2)][0][1]  # a2 column
        assert "#90EE90" in styles.ctx[(0, 3)][0][1]  # b2 column
        # Row 1: both mismatches (red) - a1, b1, a2, b2 all red
        assert "#FFB6C1" in styles.ctx[(1, 0)][0][1]  # a1 column
        assert "#FFB6C1" in styles.ctx[(1, 1)][0][1]  # b1 column
        assert "#FFB6C1" in styles.ctx[(1, 2)][0][1]  # a2 column
        assert "#FFB6C1" in styles.ctx[(1, 3)][0][1]  # b2 column
