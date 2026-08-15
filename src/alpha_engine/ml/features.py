"""Feature construction for alpha_engine.ml.alpha_model.AlphaModel --
reuses the existing signal modules (already cross-sectionally z-scored)
as base features, plus a handful of interaction and macro/temporal
context columns. Same shape as the systematic trading guide's
MLFeatureBuilder -- now including earnings-revision/short-interest/
insider-buying, once genuinely unavailable on this repo's original
63-symbol universe (no data source), added after
systematic_core_action_plan.md's 200-symbol universe expansion made real
(verified, not assumed -- see signals/earnings_revision.py, data/
short_interest.py, data/insider.py's own docstrings) data sources for
them available.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.signals import (
    earnings_revision,
    insider_buying,
    mean_reversion,
    momentum,
    quality,
    short_interest,
    value,
    volatility,
)


def build_ml_features(
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame] | None,
    macro_data: pd.DataFrame | None,
    as_of: pd.Timestamp,
    symbols: list[str] | None = None,
    benchmark_prices: pd.DataFrame | None = None,
    short_interest_data: dict[str, pd.DataFrame] | None = None,
    insider_transactions: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """One row per symbol (index = symbol, sorted for a stable column/row
    order across calls -- LightGBM cares about column order, not row
    order, but a stable index makes tests and debugging easier). Every
    base feature is already cross-sectionally z-scored by its producing
    signal module (`momentum`/`mean_reversion`/`volatility` always
    computed; `value`/`quality` only when `fundamentals` is given, same
    gating as backtest/engine.py's signal_table). `volatility.compute`
    is already inverted (higher score = lower vol = better), so the
    momentum/low-vol interaction below doesn't need a second negation.

    `earnings_revision` needs both `fundamentals` AND `benchmark_prices`
    (gated on both being present); `short_interest`/`insider_buying` each
    need their own respective dict (`short_interest_data`/
    `insider_transactions`) -- all three default to a neutral 0.0 column
    when their required input isn't given, same missing-signal handling
    as everywhere else, so a caller (e.g. live inference, which doesn't
    wire these three up -- see live/runner.py's module docstring) doesn't
    need special-case handling.

    `macro_data`, if given: the latest (as-of `as_of`) vix/yield_curve/
    credit_spread reading, broadcast to every symbol (these are
    market-level, not per-symbol) -- same macro context columns
    decision/regime.py's HMM classifier uses, so the ML model and the
    regime classifier see a consistent picture of "what kind of market is
    this."
    """
    mom = momentum.compute(prices, as_of)
    mean_rev = mean_reversion.compute(prices, as_of)
    vol = volatility.compute(prices, as_of)
    val = value.compute(prices, fundamentals, as_of) if fundamentals else pd.Series(dtype=float)
    qual = quality.compute(fundamentals, as_of) if fundamentals else pd.Series(dtype=float)

    earnings_rev = (
        earnings_revision.compute(prices, fundamentals, benchmark_prices, as_of)
        if fundamentals and benchmark_prices is not None
        else pd.Series(dtype=float)
    )
    short_int = short_interest.compute(short_interest_data, as_of) if short_interest_data else pd.Series(dtype=float)
    insider_buy = (
        insider_buying.compute(insider_transactions, as_of) if insider_transactions else pd.Series(dtype=float)
    )

    index = sorted(symbols) if symbols is not None else sorted(prices.keys())
    features = pd.DataFrame(index=index)
    features["momentum"] = mom.reindex(index).fillna(0.0)
    features["mean_reversion"] = mean_rev.reindex(index).fillna(0.0)
    features["volatility"] = vol.reindex(index).fillna(0.0)
    features["value"] = val.reindex(index).fillna(0.0)
    features["quality"] = qual.reindex(index).fillna(0.0)
    features["earnings_revision"] = earnings_rev.reindex(index).fillna(0.0)
    features["short_interest"] = short_int.reindex(index).fillna(0.0)
    features["insider_buying"] = insider_buy.reindex(index).fillna(0.0)

    features["momentum_x_quality"] = features["momentum"] * features["quality"]
    features["value_x_momentum"] = features["value"] * features["momentum"]
    features["momentum_x_low_vol"] = features["momentum"] * features["volatility"]

    features["month_of_year"] = as_of.month / 12.0
    features["is_january"] = 1.0 if as_of.month == 1 else 0.0

    vix_level, yield_curve, credit_spread = 0.0, 0.0, 0.0
    if macro_data is not None:
        macro_hist = macro_data.loc[:as_of]
        if not macro_hist.empty:
            latest = macro_hist.iloc[-1]
            if pd.notna(latest.get("vix")):
                vix_level = float(latest["vix"])
            if pd.notna(latest.get("yield_curve")):
                yield_curve = float(latest["yield_curve"])
            if pd.notna(latest.get("credit_spread")):
                credit_spread = float(latest["credit_spread"])
    features["vix_level"] = vix_level
    features["yield_curve"] = yield_curve
    features["credit_spread"] = credit_spread

    return features


FEATURE_COLUMNS = [
    "momentum",
    "mean_reversion",
    "volatility",
    "value",
    "quality",
    "earnings_revision",
    "short_interest",
    "insider_buying",
    "momentum_x_quality",
    "value_x_momentum",
    "momentum_x_low_vol",
    "month_of_year",
    "is_january",
    "vix_level",
    "yield_curve",
    "credit_spread",
]
