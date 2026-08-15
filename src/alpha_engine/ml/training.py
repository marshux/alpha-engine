"""Historical training-set construction for AlphaModel -- one row per
(date, symbol) pair within a date range, features from
ml/features.py::build_ml_features, labeled with the realized forward
excess return vs a benchmark. This only ever runs on historical data (a
walk-forward train window, or all-available-history for a manual/local
retrain) -- the label needs future prices relative to the feature date,
which is fine since training never runs at prediction time, only ahead of
it.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.backtest.engine import rebalance_dates
from alpha_engine.ml.features import FEATURE_COLUMNS, build_ml_features

DEFAULT_FORWARD_DAYS = 21


def forward_excess_return(
    prices: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    symbol: str,
    as_of: pd.Timestamp,
    forward_days: int,
) -> float | None:
    df = prices.get(symbol)
    if df is None:
        return None
    hist = df.loc[as_of:]
    if len(hist) <= forward_days:
        return None
    start_price = hist["close"].iloc[0]
    end_price = hist["close"].iloc[forward_days]
    if start_price <= 0:
        return None

    bench_hist = benchmark_prices.loc[as_of:]
    if len(bench_hist) <= forward_days:
        return None
    bench_start = bench_hist["close"].iloc[0]
    bench_end = bench_hist["close"].iloc[forward_days]
    if bench_start <= 0:
        return None

    return (end_price / start_price - 1.0) - (bench_end / bench_start - 1.0)


def build_training_set(
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame] | None,
    macro_data: pd.DataFrame | None,
    benchmark_prices: pd.DataFrame,
    symbols: list[str],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    cadence: str = "weekly",
    forward_days: int = DEFAULT_FORWARD_DAYS,
    short_interest_data: dict[str, pd.DataFrame] | None = None,
    insider_transactions: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Iterates rebalance-cadence dates within [train_start, train_end],
    builds features at each date (using only data <= that date, same as
    every live/backtest signal computation), and labels each (date,
    symbol) row with its realized forward `forward_days`-day excess
    return vs `benchmark_prices`. Rows with no computable label
    (insufficient forward price history -- i.e. too close to `train_end`)
    are dropped, not zero-filled -- a fabricated zero label would quietly
    bias the model rather than just shrinking the training set.

    `short_interest_data`/`insider_transactions`, if given: forwarded
    straight to build_ml_features -- see its docstring for the
    per-feature gating.
    """
    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    dates = rebalance_dates(all_dates, cadence)
    dates = dates[(dates >= train_start) & (dates <= train_end)]

    X_rows: list[pd.Series] = []
    y_rows: list[float] = []
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
        for symbol in features.index:
            label = forward_excess_return(prices, benchmark_prices, symbol, as_of, forward_days)
            if label is None:
                continue
            X_rows.append(features.loc[symbol])
            y_rows.append(label)

    if not X_rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float)

    X = pd.DataFrame(X_rows).reset_index(drop=True)
    y = pd.Series(y_rows)
    return X, y
