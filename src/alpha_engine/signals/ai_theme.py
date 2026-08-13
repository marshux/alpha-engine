"""AI theme tilt: a deliberate, discretionary overweight toward companies
with direct AI infrastructure/platform exposure (config/themes.yaml's `ai`
mapping), expressed as just another signal so it composes with the rest
of the deterministic decision engine rather than bypassing it.

Unlike every other signal in this package, this is NOT a backtested
academic factor -- it's a directional bet that AI's economic importance
keeps growing. Untagged symbols are absent from the raw signal entirely,
which decision/scorer.combine_scores treats as neutral (0): being outside
the theme is never penalized, only inside it is rewarded. Portfolio-level
concentration risk from this bet is capped separately by
decision/portfolio.apply_theme_cap (config/strategy.yaml's
portfolio.max_theme_weight) -- that's the "hedge," a bound on how much the
bet can dominate the portfolio, not an offsetting position.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.signals.base import zscore


def compute(
    prices: dict[str, pd.DataFrame], as_of: pd.Timestamp, theme_weights: dict[str, float]
) -> pd.Series:
    """`theme_weights`: symbol -> hand-curated AI-exposure intensity
    (config/themes.yaml's `ai` mapping) -- graded, not binary, so the
    resulting cross-sectional z-score has real spread. A uniform
    membership flag for every tagged symbol would have zero variance and
    z-score to exactly 0 for everyone, silently making the tilt a no-op.
    """
    values = {symbol: weight for symbol, weight in theme_weights.items() if symbol in prices}
    return zscore(pd.Series(values, dtype=float))
