"""Earnings-revision proxy: this project has no access to paid analyst
consensus estimate feeds (see systematic_trading_engineering_guide.md's
corrected data-sources table -- "No reliable free source" for earnings
estimates). Instead, approximates earnings-revision momentum with
Post-Earnings-Announcement Drift (PEAD): the well-documented anomaly that
a stock's *initial* price reaction to a genuine earnings surprise predicts
continued drift in the same direction for weeks afterward. This module
measures only that initial reaction (a short window ending at the report
date) -- the signal is a bet that this reaction predicts future drift, not
a measurement of drift that has already happened, so it stays a legitimate
forward-looking alpha signal without needing to see anything past `as_of`.

Needs both price data (for the cumulative abnormal return calculation)
and fundamentals data (to locate the report date), plus a benchmark
series to compute the *abnormal* (market-adjusted) part of the return --
so, like signals/value.py, `compute()`'s signature differs from the
uniform prices-only shape in signals/base.py and is called explicitly
rather than through SIGNAL_MODULES.

Point-in-time caveat: there is no free, exact public earnings-announcement-
date field on this data source. The most recent fundamentals report's
period-end date stands in for the announcement date -- the same
"optimistic but never selects a future report" approximation
data/fundamentals.py's latest_report_as_of already makes everywhere else
in this codebase (real filings lag the period end by 4-8 weeks, so this
window is measured slightly earlier than the real announcement in
practice, not later -- never a look-ahead risk, only a fuzziness in
exactly which few days the reaction is measured over).
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.data.fundamentals import latest_report_as_of
from alpha_engine.signals.base import zscore

WINDOW_DAYS = 3


def compute(
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int = WINDOW_DAYS,
) -> pd.Series:
    values: dict[str, float] = {}
    for symbol, fdf in fundamentals.items():
        report = latest_report_as_of(fdf, as_of)
        if report is None:
            continue
        report_date = report.name  # the report's period-end date (its DataFrame index value)

        price_df = prices.get(symbol)
        if price_df is None:
            continue

        car = _cumulative_abnormal_return(price_df, benchmark_prices, report_date, window_days)
        if car is not None:
            values[symbol] = car

    return zscore(pd.Series(values, dtype=float))


def _cumulative_abnormal_return(
    price_df: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    event_date: pd.Timestamp,
    window_days: int,
) -> float | None:
    """Stock return minus benchmark return over the `window_days` trading
    days ending at (and including) `event_date` -- looking backward from
    the report date only, never forward, so this is safe to compute at
    any later `as_of` without leaking information the report itself
    wouldn't have already reflected.
    """
    stock_hist = price_df.loc[:event_date]["close"]
    bench_hist = benchmark_prices.loc[:event_date]["close"]
    if len(stock_hist) <= window_days or len(bench_hist) <= window_days:
        return None
    stock_ret = stock_hist.iloc[-1] / stock_hist.iloc[-window_days - 1] - 1.0
    bench_ret = bench_hist.iloc[-1] / bench_hist.iloc[-window_days - 1] - 1.0
    return stock_ret - bench_ret
