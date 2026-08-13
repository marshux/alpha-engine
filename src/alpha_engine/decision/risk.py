"""Deterministic risk rules evaluated at each rebalance: per-position
stop-loss, a portfolio-level max-drawdown circuit breaker, and a
benchmark trend/regime filter. All operate purely on realized price/equity
history -- no forward-looking or LLM input.
"""
from __future__ import annotations

import pandas as pd


def stopped_out_positions(
    entry_prices: dict[str, float],
    current_prices: dict[str, float],
    stop_loss_pct: float,
) -> set[str]:
    """Symbols whose price has fallen more than `stop_loss_pct` below their
    entry price -- these should be force-closed regardless of current
    signal score.
    """
    stopped = set()
    for symbol, entry in entry_prices.items():
        current = current_prices.get(symbol)
        if current is None or entry <= 0:
            continue
        if (current / entry - 1.0) <= -stop_loss_pct:
            stopped.add(symbol)
    return stopped


def circuit_breaker_triggered(equity_curve: pd.Series, max_drawdown_pct: float) -> bool:
    """True if current drawdown from the running peak equity exceeds the
    configured max. When true, the caller should flatten to cash rather
    than open new positions at this rebalance.
    """
    if equity_curve.empty:
        return False
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    return bool(drawdown.iloc[-1] <= -max_drawdown_pct)


def anomalous_drop_positions(
    prices: dict[str, pd.DataFrame],
    symbols: set[str],
    as_of: pd.Timestamp,
    z_threshold: float = 3.0,
    lookback_days: int = 60,
) -> set[str]:
    """Symbols whose most recent daily return is an outlier relative to
    their OWN trailing daily-return distribution -- catches a sudden
    abnormal single-day move for that specific stock, independent of
    cumulative loss from entry (stopped_out_positions) or portfolio-level
    drawdown (circuit_breaker_triggered). Volatility-adjusted rather than a
    flat percentage so a naturally choppy stock doesn't flag on an
    ordinary day for it, while a naturally stable one flags on a much
    smaller absolute move. Only flags downside moves ("going red"),
    intentionally not symmetric. Meant to be checked daily
    (live/anomaly_guard.py), independent of rebalance cadence entirely.
    """
    flagged = set()
    for symbol in symbols:
        df = prices.get(symbol)
        if df is None:
            continue
        hist = df.loc[:as_of]
        returns = hist["close"].pct_change().dropna()
        if len(returns) < lookback_days + 1:
            continue
        today_return = returns.iloc[-1]
        # Excludes today's return from its own baseline so one big move
        # doesn't inflate the std it's being measured against.
        trailing = returns.iloc[-lookback_days - 1 : -1]
        std = trailing.std()
        if not std or std <= 0:
            continue
        if today_return / std <= -z_threshold:
            flagged.add(symbol)
    return flagged


def regime_is_risk_on(benchmark_prices: pd.DataFrame, as_of: pd.Timestamp, ma_window: int) -> bool:
    """Faber-style trend filter: True if the benchmark's close as of
    `as_of` is above its own trailing `ma_window`-day simple moving
    average -- i.e. the broad market is in an uptrend and the portfolio
    should be invested. False means "sit in cash regardless of what
    individual stock signals say" -- a strategy long individual stocks in
    a declining broad market has historically fared worse than one that
    also times broad market exposure.

    Defaults to True (invested) when there's not yet `ma_window` days of
    benchmark history, rather than blocking the entire early backtest
    period before a trend judgment is even possible.
    """
    hist = benchmark_prices.loc[:as_of]
    if len(hist) < ma_window:
        return True
    sma = hist["close"].iloc[-ma_window:].mean()
    current = hist["close"].iloc[-1]
    return bool(current > sma)
