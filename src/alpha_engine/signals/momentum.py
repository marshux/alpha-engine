"""12-1 month momentum: total return from ~12 months ago to ~1 month ago,
skipping the most recent month. This is the classic academic momentum
construction — including the most recent month tends to pick up short-term
reversal, which works against the momentum effect rather than with it.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.signals.base import zscore

LOOKBACK_DAYS = 252
SKIP_DAYS = 21


def compute(prices: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, df in prices.items():
        hist = df.loc[:as_of]
        if len(hist) < LOOKBACK_DAYS + 1:
            continue
        start_price = hist["close"].iloc[-LOOKBACK_DAYS - 1]
        end_price = hist["close"].iloc[-SKIP_DAYS - 1]
        if start_price <= 0:
            continue
        values[symbol] = (end_price / start_price) - 1.0
    return zscore(pd.Series(values, dtype=float))
