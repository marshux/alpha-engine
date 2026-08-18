"""Short-interest VELOCITY: the ~4-week rate of change in days-to-cover,
as a NEGATIVE cross-sectional signal -- a rapidly *rising* short position
reflects active, recent bearish conviction, distinct from the existing
`short_interest` signal's *level* (a historical overhang that may be
stale). Same empirical "high = bearish" convention as the level signal.

Separate signal key from `short_interest`, not a replacement --
systematic_core_action_plan (4).md's "preferred" approach: keeping the
level and velocity reads independent lets each be IC-gated on its own
before either earns composite weight, rather than baking an untested
combination into one signal.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.data.short_interest import latest_short_interest_as_of
from alpha_engine.signals.base import zscore

VELOCITY_WINDOW_DAYS = 28  # ~4 weeks, matching the plan's spec


def compute(short_interest: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, df in short_interest.items():
        current = latest_short_interest_as_of(df, as_of)
        prior = latest_short_interest_as_of(df, as_of - pd.Timedelta(days=VELOCITY_WINDOW_DAYS))
        if current is None or prior is None:
            continue
        current_dtc = current.get("days_to_cover")
        prior_dtc = prior.get("days_to_cover")
        if current_dtc is None or prior_dtc is None or pd.isna(current_dtc) or pd.isna(prior_dtc):
            continue
        if prior_dtc <= 0:
            continue  # avoid a divide-by-zero / meaningless ratio
        values[symbol] = (current_dtc - prior_dtc) / prior_dtc

    return zscore(-pd.Series(values, dtype=float))
