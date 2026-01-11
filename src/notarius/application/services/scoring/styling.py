from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import pandas as pd

from notarius.application.services.scoring.protocol import Scorer, StylingStrategy

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler


@dataclass
class ExactMatchStyling(StylingStrategy):
    """Discrete coloring: Green (1.0), Yellow (0.5), Red (0.0)"""

    color_match: str = "#90EE90"  # Light green
    color_off_by_one: str = "#FFFFE0"  # Light yellow
    color_mismatch: str = "#FFB6C1"  # Light red
    color_neutral: str = "#D3D3D3"  # Light gray

    @override
    def get_style(self, score: float) -> str:
        if score >= 1.0:
            return f"background-color: {self.color_match}"
        elif score >= 0.5:
            return f"background-color: {self.color_off_by_one}"
        else:
            return f"background-color: {self.color_mismatch}"

    @override
    def get_neutral_style(self) -> str:
        return f"background-color: {self.color_neutral}"


@dataclass
class GradientStyling(StylingStrategy):
    """Continuous gradient: Red (0.0) → Yellow (0.5) → Green (1.0)"""

    color_neutral: str = "#D3D3D3"  # Light gray

    @override
    def get_style(self, score: float) -> str:
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        if score < 0.5:
            # Interpolate between Red (255, 182, 193) and Yellow (255, 255, 224)
            t = score / 0.5
            r = 255
            g = int(182 + (255 - 182) * t)
            b = int(193 + (224 - 193) * t)
        else:
            # Interpolate between Yellow (255, 255, 224) and Green (144, 238, 144)
            t = (score - 0.5) / 0.5
            r = int(255 + (144 - 255) * t)
            g = int(255 + (238 - 255) * t)
            b = int(224 + (144 - 224) * t)

        return f"background-color: #{r:02x}{g:02x}{b:02x}"

    @override
    def get_neutral_style(self) -> str:
        return f"background-color: {self.color_neutral}"


def _is_empty(val: str | None) -> bool:
    """Check if a value is empty (None, NaN, or whitespace-only string)."""
    if val is None:
        return True
    return str(val).strip() == ""


@dataclass
class CellComparisonStyler:
    """Styler that applies cell-level coloring based on comparison scores.

    This service separates presentation logic from evaluation services.
    Evaluation services compute metrics; this styler decorates DataFrames for display.
    """

    scorer: Scorer
    styling: StylingStrategy

    def style(
        self,
        df: pd.DataFrame,
        *,
        columns_to_compare: list[tuple[str, str]],
    ) -> Styler:
        """Apply styling to a DataFrame based on comparison scores.

        For each column pair (col_a, col_b), computes the similarity score
        and applies the styling strategy's color to both col_a and col_b cells.
        When both values are empty/missing, applies neutral styling instead.
        """

        def _color_cells(data: pd.DataFrame) -> pd.DataFrame:
            style_df = pd.DataFrame("", index=data.index, columns=data.columns)
            for col_a, col_b in columns_to_compare:
                if col_a in data.columns and col_b in data.columns:
                    for idx in data.index:
                        val_a = data.at[idx, col_a]
                        val_b = data.at[idx, col_b]
                        s_a = str(val_a) if pd.notna(val_a) else None
                        s_b = str(val_b) if pd.notna(val_b) else None

                        # Both empty → neutral styling
                        if _is_empty(s_a) and _is_empty(s_b):
                            style = self.styling.get_neutral_style()
                        else:
                            score = self.scorer.score(s_a, s_b)
                            style = self.styling.get_style(score)

                        style_df.at[idx, col_a] = style
                        style_df.at[idx, col_b] = style
            return style_df

        return df.style.apply(_color_cells, axis=None)
