"""Information Coefficient (IC) validation for
`alpha_engine.ml.alpha_model.AlphaModel`. The systematic trading
engineering guide's own backtesting checklist (section 5.7) requires
"each signal has positive IC on out-of-sample data (IC > 0.03)" before
trusting it in a composite score -- this was never actually checked for
`ml_alpha` before (see `systematic_core_postmortem.md`'s root-cause
analysis: an unvalidated, feature-poor ML signal was flagged as one
likely contributor to Systematic Core's underperformance). These
functions close that gap.

`compute_signal_ic` (below `compute_mean_ic`) applies the same gate to a
plain cross-sectional signal directly, not just the ml_alpha model's
predictions -- every prior signal added to this codebase (momentum,
value, quality, earnings_revision, short_interest, insider_buying) was
integrated without its own standalone IC ever actually being measured;
only ml_alpha's composite-model IC had real gating infrastructure. Added
for systematic_core_action_plan (4).md's new-signal validation
(short_interest_velocity, and later congress_trading/earnings_sentiment),
which need this same discipline applied per-signal, not just at the
model level.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alpha_engine.backtest.engine import rebalance_dates
from alpha_engine.ml.alpha_model import AlphaModel
from alpha_engine.ml.features import build_ml_features
from alpha_engine.ml.training import forward_excess_return

MIN_IC_THRESHOLD = 0.03


def compute_ic(signal_scores: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman rank correlation between a cross-sectional signal and
    realized forward returns for the same date -- the guide's own IC
    definition (section 4.4's `AdaptiveWeightLearner.compute_ic`). `0.0`
    if fewer than 5 paired (symbol, return) observations exist that
    date -- not enough for a meaningful correlation, same "degrade
    gracefully" posture as the rest of this codebase.
    """
    aligned = pd.concat([signal_scores, forward_returns], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(ic) if pd.notna(ic) else 0.0


def compute_mean_ic(
    model: AlphaModel,
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame] | None,
    macro_data: pd.DataFrame | None,
    benchmark_prices: pd.DataFrame,
    symbols: list[str],
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    cadence: str = "weekly",
    forward_days: int = 21,
    short_interest_data: dict[str, pd.DataFrame] | None = None,
    insider_transactions: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Runs `model.predict()` at every rebalance-cadence date in
    `[validation_start, validation_end]`, computing that date's
    cross-sectional IC (`compute_ic`) against realized forward returns.
    This period should be held out from the model's own training range
    -- point-in-time safety is the caller's responsibility, same as
    `alpha_engine.ml.training.build_training_set`'s train/test split.

    `short_interest_data`/`insider_transactions`, if given: forwarded
    straight to build_ml_features -- see its docstring for the
    per-feature gating.

    Returns `{mean_ic, ic_std, ic_ir, n_periods, passes}` -- `ic_ir` is
    the IC information ratio (`mean_ic / ic_std`, a stability measure:
    a signal with the same weak-but-consistent IC every period is more
    trustworthy than one with a high mean IC driven by a couple of
    lucky periods). `passes` is `mean_ic >= MIN_IC_THRESHOLD`.
    """
    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    dates = rebalance_dates(all_dates, cadence)
    dates = dates[(dates >= validation_start) & (dates <= validation_end)]

    ics: list[float] = []
    for as_of in dates:
        features = build_ml_features(
            prices,
            fundamentals,
            macro_data,
            as_of,
            symbols,
            benchmark_prices=benchmark_prices,
            short_interest_data=short_interest_data,
            insider_transactions=insider_transactions,
        )
        preds = model.predict(features)
        fwd_returns = pd.Series(
            {s: forward_excess_return(prices, benchmark_prices, s, as_of, forward_days) for s in preds.index}
        ).dropna()
        ics.append(compute_ic(preds, fwd_returns))

    ics_arr = np.array(ics, dtype=float)
    mean_ic = float(ics_arr.mean()) if len(ics_arr) else 0.0
    ic_std = float(ics_arr.std()) if len(ics_arr) else 0.0
    ic_ir = float(mean_ic / ic_std) if ic_std > 0 else 0.0

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "n_periods": len(ics_arr),
        "passes": mean_ic >= MIN_IC_THRESHOLD,
    }


def compute_signal_ic(
    signal_fn: Callable[[pd.Timestamp], pd.Series],
    prices: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    cadence: str = "weekly",
    forward_days: int = 21,
) -> dict:
    """Same methodology and return shape as `compute_mean_ic`, but for any
    single cross-sectional signal directly (`signal_fn(as_of) ->
    pd.Series`) instead of an `AlphaModel`'s predictions -- gates a raw
    signal on its own IC before it's trusted with a composite weight, the
    same way `ml_alpha` is gated, without needing to route it through a
    trained model first. `signal_fn` should already be point-in-time safe
    (only look at data `<= as_of`), same requirement as every other
    signal's `compute()` in this codebase.
    """
    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    dates = rebalance_dates(all_dates, cadence)
    dates = dates[(dates >= validation_start) & (dates <= validation_end)]

    ics: list[float] = []
    for as_of in dates:
        scores = signal_fn(as_of)
        if scores.empty:
            continue
        fwd_returns = pd.Series(
            {s: forward_excess_return(prices, benchmark_prices, s, as_of, forward_days) for s in scores.index}
        ).dropna()
        ics.append(compute_ic(scores, fwd_returns))

    ics_arr = np.array(ics, dtype=float)
    mean_ic = float(ics_arr.mean()) if len(ics_arr) else 0.0
    ic_std = float(ics_arr.std()) if len(ics_arr) else 0.0
    ic_ir = float(mean_ic / ic_std) if ic_std > 0 else 0.0

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "n_periods": len(ics_arr),
        "passes": mean_ic >= MIN_IC_THRESHOLD,
    }
