"""Realized volatility (annualized, trailing window), inverted — lower
volatility scores higher. This is a risk-adjusted tilt away from names
moving erratically, independent of direction, not a directional bet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.signals.base import zscore

WINDOW_DAYS = 20
TRADING_DAYS_PER_YEAR = 252
BETA_WINDOW_DAYS = 252


def raw_annualized_vol(
    prices: dict[str, pd.DataFrame], as_of: pd.Timestamp, window: int = WINDOW_DAYS
) -> pd.Series:
    """Trailing annualized realized volatility per symbol, un-inverted and
    un-z-scored. Also used directly by the backtester's volatility-scaled
    position sizing (decision/portfolio.py) -- kept separate from `compute`
    so that consumer doesn't have to invert a signal back out of its
    z-scored form to recover the actual vol number.
    """
    values: dict[str, float] = {}
    for symbol, df in prices.items():
        hist = df.loc[:as_of]
        if len(hist) < window + 1:
            continue
        returns = hist["close"].iloc[-window - 1 :].pct_change().dropna()
        vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        if pd.notna(vol) and vol > 0:
            values[symbol] = vol
    return pd.Series(values, dtype=float)


def compute(prices: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    vol = raw_annualized_vol(prices, as_of)
    return zscore(-vol)


def trailing_returns_matrix(
    prices: dict[str, pd.DataFrame], symbols: list[str], as_of: pd.Timestamp, window: int = WINDOW_DAYS
) -> pd.DataFrame:
    """Daily returns for each symbol over the trailing `window`, as of
    `as_of` -- symbols as columns, dates as rows. Feeds the min-variance
    portfolio sizing in decision/portfolio.py (its covariance matrix is
    just this DataFrame's `.cov()`). Symbols with insufficient history are
    dropped entirely rather than left with NaNs, since a covariance matrix
    with missing entries isn't invertible.
    """
    series = {}
    for symbol in symbols:
        df = prices.get(symbol)
        if df is None:
            continue
        hist = df.loc[:as_of]
        if len(hist) < window + 1:
            continue
        returns = hist["close"].iloc[-window - 1 :].pct_change().dropna()
        series[symbol] = returns
    if not series:
        return pd.DataFrame()
    # Align by date (not position) -- a symbol missing a trading day that
    # others have would otherwise silently misalign the covariance calc.
    # Drop any symbol with a gap rather than trying to patch it.
    return pd.concat(series, axis=1).dropna(axis=1, how="any")


def compute_beta(
    prices: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = BETA_WINDOW_DAYS,
) -> pd.Series:
    """Trailing beta of each symbol vs `benchmark_prices` (cov/var of daily
    returns over `window` trading days) -- raw, un-inverted, un-z-scored,
    same posture as `raw_annualized_vol`. Diagnostic use (portfolio-beta
    audits, an optional beta floor on the investable universe) rather than
    a z-scored signal feeding the composite score. Symbols/benchmark with
    insufficient trailing history are simply absent from the result.
    """
    bench_hist = benchmark_prices.loc[:as_of]
    if len(bench_hist) < window + 1:
        return pd.Series(dtype=float)
    bench_returns = bench_hist["close"].iloc[-window - 1 :].pct_change().dropna()
    bench_var = bench_returns.var()
    if not pd.notna(bench_var) or bench_var == 0:
        return pd.Series(dtype=float)

    values: dict[str, float] = {}
    for symbol, df in prices.items():
        hist = df.loc[:as_of]
        if len(hist) < window + 1:
            continue
        returns = hist["close"].iloc[-window - 1 :].pct_change().dropna()
        aligned = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
        if len(aligned) < window // 2:
            continue
        cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        values[symbol] = cov / bench_var
    return pd.Series(values, dtype=float)


def portfolio_beta(weights: pd.Series, betas: pd.Series) -> float:
    """Weighted-average beta of a portfolio. Symbols with no beta available
    (e.g. insufficient trailing history) are excluded from both the
    numerator and the weight-sum denominator, rather than treated as
    beta=0 -- so a portfolio's beta isn't silently diluted by symbols this
    function simply has no opinion on. NaN if no held symbol has a beta.
    """
    aligned = betas.reindex(weights.index).dropna()
    if aligned.empty:
        return float("nan")
    w = weights.reindex(aligned.index)
    denom = w.abs().sum()
    if denom == 0:
        return float("nan")
    return float((w * aligned).sum() / denom)
