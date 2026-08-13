"""Tear-sheet style reporting: one figure summarizing a backtest run, in
the spirit of pyfolio's tear sheet -- equity curve vs benchmark(s),
underwater/drawdown plot, rolling Sharpe, and turnover per rebalance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- this is a CLI tool, never an interactive session
import matplotlib.pyplot as plt
import pandas as pd

ROLLING_SHARPE_WINDOW = 26


def _rolling_sharpe(equity_curve: pd.Series, periods_per_year: float, window: int) -> pd.Series:
    returns = equity_curve.pct_change().dropna()
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std) * (periods_per_year**0.5)


def generate_tearsheet(
    equity_curve: pd.Series,
    benchmark_curves: dict[str, pd.Series],
    turnover_history: dict[pd.Timestamp, float],
    periods_per_year: float,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=False)

    ax = axes[0]
    ax.plot(equity_curve.index, equity_curve.values, label="Strategy", linewidth=1.5)
    for symbol, curve in benchmark_curves.items():
        ax.plot(curve.index, curve.values, label=symbol, linewidth=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Equity Curve (log scale)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    ax.fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.5)
    ax.set_title("Drawdown")

    ax = axes[2]
    rolling_sharpe = _rolling_sharpe(equity_curve, periods_per_year, ROLLING_SHARPE_WINDOW)
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="darkgreen", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Rolling {ROLLING_SHARPE_WINDOW}-Period Sharpe")

    ax = axes[3]
    turnover = pd.Series(turnover_history).sort_index()
    ax.bar(turnover.index, turnover.values, width=5, color="steelblue")
    ax.set_title("Turnover per Rebalance")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
