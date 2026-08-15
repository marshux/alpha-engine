"""Short-interest signal: crowded shorts (high days-to-cover -- how many
days of average trading volume it would take to close out the currently
outstanding short position) as a NEGATIVE cross-sectional signal. The
guide itself flags the ambiguity here (high short interest could mean
"squeeze potential" -- a positive contrarian read -- or "smart money is
already positioned against this stock" -- negative); this module follows
the guide's own stated empirical convention and treats it as negative.

Fundamentals-shaped input (like signals/quality.py), not the uniform
prices-only shape in signals/base.py -- called explicitly rather than
through SIGNAL_MODULES. Bi-weekly cadence, not daily:
data/short_interest.py's latest_short_interest_as_of finds the most
recent report available as of `as_of`, same point-in-time pattern
data/fundamentals.py's latest_report_as_of uses elsewhere in this
codebase. Coverage only extends back to ~2018 (see
data/short_interest.py's module docstring for why) -- symbols/dates
without coverage score neutral, same missing-signal handling as
everywhere else.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.data.short_interest import latest_short_interest_as_of
from alpha_engine.signals.base import zscore


def compute(short_interest: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, df in short_interest.items():
        report = latest_short_interest_as_of(df, as_of)
        if report is None:
            continue
        days_to_cover = report.get("days_to_cover")
        if days_to_cover is None or pd.isna(days_to_cover):
            continue
        values[symbol] = float(days_to_cover)

    return zscore(-pd.Series(values, dtype=float))
