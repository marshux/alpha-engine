"""Insider buying signal: net insider open-market purchase activity
(purchases minus sales, as a fraction of total open-market transaction
*share volume*) over a trailing lookback window. Positive net buying is
a positive signal -- an insider committing personal capital via a genuine
open-market purchase is a meaningfully different signal from routine
equity-compensation vesting/grants, which data/insider.py's transaction-
code filter (P/S only) already excludes before this module ever sees it.

Fundamentals-shaped input (like signals/short_interest.py/quality.py),
called explicitly rather than through SIGNAL_MODULES. Coverage caveat:
SEC's full-text search index and per-symbol fetch cost mean coverage may
be thin for some symbols/date ranges -- gaps score neutral, same
missing-signal handling as everywhere else in this codebase.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.signals.base import zscore

DEFAULT_LOOKBACK_DAYS = 90


def compute(
    insider_transactions: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.Series:
    values: dict[str, float] = {}
    window_start = as_of - pd.Timedelta(days=lookback_days)

    for symbol, df in insider_transactions.items():
        if df.empty:
            continue
        window = df.loc[(df.index > window_start) & (df.index <= as_of)]
        if window.empty:
            continue
        buys = window.loc[window["code"] == "P", "shares"].sum()
        sells = window.loc[window["code"] == "S", "shares"].sum()
        total = buys + sells
        if total <= 0:
            continue
        values[symbol] = (buys - sells) / total

    return zscore(pd.Series(values, dtype=float))
