"""Deterministic risk rules evaluated at each rebalance: per-position
stop-loss and a portfolio-level max-drawdown circuit breaker. Both operate
purely on realized price/equity history -- no forward-looking or LLM input.
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
