"""Short-term mean reversion: how far price has strayed from its rolling
moving average, inverted — a large positive deviation implies an expected
pullback (lower score), a large negative deviation implies an expected
bounce (higher score).
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.signals.base import zscore

WINDOW_DAYS = 20


def compute(prices: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, df in prices.items():
        hist = df.loc[:as_of]
        if len(hist) < WINDOW_DAYS:
            continue
        window = hist["close"].iloc[-WINDOW_DAYS:]
        std = window.std()
        if std == 0 or pd.isna(std):
            continue
        deviation = (window.iloc[-1] - window.mean()) / std
        values[symbol] = -deviation
    return zscore(pd.Series(values, dtype=float))
