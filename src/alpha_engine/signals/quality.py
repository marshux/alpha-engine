"""Quality factor: profitability (ROE) and low leverage (inverse
debt-to-equity), combined into one composite and z-scored. Fundamentals-
only (no price data needed) -- unlike signals/value.py, which needs both.

Point-in-time via data.fundamentals.latest_report_as_of; see that module's
docstring and CLAUDE.md for the caveat about yfinance's shallow historical
depth (coverage will be thin for older backtest dates -- those silently
fall back to neutral, same as any other missing signal).
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.data.fundamentals import latest_report_as_of
from alpha_engine.signals.base import zscore


def compute(fundamentals: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.Series:
    roe_values: dict[str, float] = {}
    debt_equity_values: dict[str, float] = {}

    for symbol, df in fundamentals.items():
        report = latest_report_as_of(df, as_of)
        if report is None:
            continue
        equity = report.get("Stockholders Equity")
        if equity is None or pd.isna(equity) or equity <= 0:
            continue

        net_income = report.get("Net Income")
        if net_income is not None and pd.notna(net_income):
            roe_values[symbol] = net_income / equity

        total_debt = report.get("Total Debt")
        if total_debt is not None and pd.notna(total_debt):
            debt_equity_values[symbol] = total_debt / equity

    roe_z = zscore(pd.Series(roe_values, dtype=float))
    debt_equity_z = zscore(pd.Series(debt_equity_values, dtype=float))

    combined: dict[str, float] = {}
    for symbol in sorted(set(roe_z.index) | set(debt_equity_z.index)):
        parts = []
        if symbol in roe_z.index:
            parts.append(roe_z[symbol])
        if symbol in debt_equity_z.index:
            parts.append(-debt_equity_z[symbol])  # lower leverage is better
        if parts:
            combined[symbol] = sum(parts) / len(parts)

    return zscore(pd.Series(combined, dtype=float))
