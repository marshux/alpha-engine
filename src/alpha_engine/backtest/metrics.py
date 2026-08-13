"""Backtest performance metrics: CAGR, Sharpe, max drawdown, and comparison
against a buy-and-hold benchmark (SPY/VOO/VTI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0 or total_return <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def sharpe_ratio(
    equity_curve: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0
) -> float:
    returns = equity_curve.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float((excess.mean() / returns.std()) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def benchmark_equity_curve(
    benchmark_prices: pd.DataFrame, dates: pd.DatetimeIndex, initial_capital: float
) -> pd.Series:
    """Buy-and-hold equity curve for a benchmark, evaluated at the same
    dates as the strategy's equity curve for an apples-to-apples comparison.
    """
    aligned = benchmark_prices["close"].reindex(dates, method="ffill")
    return initial_capital * (aligned / aligned.iloc[0])


def summarize(
    equity_curve: pd.Series, benchmark_curve: pd.Series | None, periods_per_year: float
) -> dict:
    summary = {
        "cagr": cagr(equity_curve),
        "sharpe": sharpe_ratio(equity_curve, periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "final_equity": float(equity_curve.iloc[-1]) if len(equity_curve) else 0.0,
    }
    if benchmark_curve is not None and len(benchmark_curve) > 1:
        summary["benchmark_cagr"] = cagr(benchmark_curve)
        summary["benchmark_max_drawdown"] = max_drawdown(benchmark_curve)
        summary["alpha_cagr"] = summary["cagr"] - summary["benchmark_cagr"]
    return summary
