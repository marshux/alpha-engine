"""Earnings press-release language sentiment (Loughran-McDonald scored
8-K exhibit text -- see data/earnings_sentiment.py's module docstring for
the correction on what's actually being scored vs. what
systematic_core_action_plan (4).md describes).

Composite of level and change, per the plan's spec:
  raw_sentiment   = positive_score - negative_score - 0.5 * uncertainty_score
  delta_sentiment = raw_sentiment - raw_sentiment(prior quarter's filing)
  composite       = 0.5 * raw_sentiment + 0.5 * delta_sentiment

If a symbol has a filing history but nothing in the trailing
`lookback_days` (between reporting periods): composite = 0.0, included in
the z-score as a genuine neutral reading, not excluded as missing data --
explicit in the plan's spec. A symbol with no filing history at all
(never resolved a CIK, or no earnings 8-Ks found) is excluded entirely,
same missing-signal handling as everywhere else in this codebase.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.data.earnings_sentiment import latest_earnings_sentiment_as_of
from alpha_engine.signals.base import zscore

LOOKBACK_DAYS = 90


def _raw_sentiment(report: pd.Series) -> float:
    return report["positive_score"] - report["negative_score"] - 0.5 * report["uncertainty_score"]


def compute(
    earnings_sentiment: dict[str, pd.DataFrame], as_of: pd.Timestamp, lookback_days: int = LOOKBACK_DAYS
) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, df in earnings_sentiment.items():
        current = latest_earnings_sentiment_as_of(df, as_of)
        if current is None:
            continue  # no filing history at all -- true missing data, excluded

        current_date = current.name
        if (as_of - current_date).days > lookback_days:
            values[symbol] = 0.0  # between reporting periods -- neutral, not missing
            continue

        raw_sentiment = _raw_sentiment(current)
        prior = latest_earnings_sentiment_as_of(df, current_date - pd.Timedelta(days=lookback_days))
        delta_sentiment = (raw_sentiment - _raw_sentiment(prior)) if prior is not None else 0.0

        values[symbol] = 0.5 * raw_sentiment + 0.5 * delta_sentiment

    return zscore(pd.Series(values, dtype=float))
