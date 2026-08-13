"""Value factor: cheapness via trailing price-to-book (lower P/B scores
higher). Needs both price data (for the `as_of` price) and fundamentals
data (for book value per share), so `compute()`'s signature differs from
the uniform prices-only shape in signals/base.py -- backtest/engine.py
calls it explicitly rather than through SIGNAL_MODULES, the same way it
already special-cases signals/quality.py (fundamentals-only) and
signals/news_ai.py (live-only).

See data/fundamentals.py's docstring for the point-in-time caveat.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.data.fundamentals import latest_report_as_of
from alpha_engine.signals.base import zscore


def compute(
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
) -> pd.Series:
    price_to_book: dict[str, float] = {}

    for symbol, df in fundamentals.items():
        report = latest_report_as_of(df, as_of)
        if report is None:
            continue
        equity = report.get("Stockholders Equity")
        shares = report.get("Ordinary Shares Number")
        if equity is None or shares is None or pd.isna(equity) or pd.isna(shares) or shares <= 0:
            continue
        book_value_per_share = equity / shares
        if book_value_per_share <= 0:
            continue

        price_df = prices.get(symbol)
        if price_df is None:
            continue
        hist = price_df.loc[:as_of]
        if hist.empty:
            continue

        price_to_book[symbol] = hist["close"].iloc[-1] / book_value_per_share

    return zscore(-pd.Series(price_to_book, dtype=float))
