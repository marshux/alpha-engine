"""Deterministic combination of per-signal scores into one ranked score per
asset. Plain weighted sum -- no ML/LLM in this module. Given identical
signal inputs this always produces identical output, which is the property
that makes backtests trustworthy: the AI news signal's non-determinism is
isolated to signal *generation* (signals/news_ai.py), never to how signals
get turned into trades.
"""
from __future__ import annotations

import pandas as pd


def combine_scores(signal_table: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """`signal_table`: index=symbol, columns=signal name, values already
    z-scored by the producing signal module.

    Missing values for a symbol/signal pair are treated as neutral (0), not
    dropped -- a symbol shouldn't be excluded from ranking just because one
    signal (e.g. news_ai, before it's wired up) has no data for it yet.
    """
    active_weights = {
        name: w for name, w in weights.items() if name in signal_table.columns and w != 0
    }
    if not active_weights:
        return pd.Series(0.0, index=signal_table.index)

    weighted = sum(
        signal_table[name].fillna(0.0) * w for name, w in active_weights.items()
    )
    return weighted.sort_values(ascending=False)
