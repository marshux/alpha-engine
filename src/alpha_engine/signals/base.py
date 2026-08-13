"""Shared interface and helpers for signal modules (momentum, mean_reversion,
volatility, ai_theme).

Every signal module exposes a module-level `compute(prices, as_of) -> pd.Series`
function: `prices` is `{symbol: OHLCV DataFrame}` (as returned by
`data.equities.get_universe_prices`), `as_of` is the rebalance date, and the
returned Series (indexed by symbol) is a cross-sectionally z-scored value —
comparable across signals so the decision engine can combine them by simple
weighted sum. `value.py`/`quality.py` (need fundamentals) and `news_ai.py`
(single portfolio-level scalar, not per-symbol -- see its own docstring)
diverge from this shape and aren't called through it.
"""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class SignalFn(Protocol):
    def __call__(self, prices: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series: ...


def zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score. Returns all-zero if the universe has no spread."""
    if series.empty:
        return series
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std
