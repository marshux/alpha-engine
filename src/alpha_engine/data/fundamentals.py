"""Quarterly fundamentals fetch (balance sheet + income statement) via
yfinance, cached like data/equities.py.

LOOKAHEAD CAVEAT (same shape as signals/news_ai.py's -- see CLAUDE.md):
yfinance's quarterly history is shallow in practice (a handful of the most
recent quarters, not deep multi-year restated history -- typically under
two years back as of when this was written), not a full point-in-time
archive. We approximate "available as of a date" by using each report's
period-end date as its availability date, which is optimistic (real
filings lag the period end by 4-8 weeks) but never selects a report dated
after `as_of`. Coverage will be thin or empty for backtest dates before
yfinance's available quarterly history starts -- those dates silently get
a neutral value/quality signal, via the same missing-signal handling
`decision/scorer.combine_scores` already applies everywhere else.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "fundamentals"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"


def get_fundamentals(symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    """Quarterly balance-sheet + income-statement line items for `symbol`,
    indexed by report (period-end) date ascending, one column per line
    item. Empty DataFrame if yfinance has no fundamentals for this symbol
    (common for ETFs, which don't file financial statements).
    """
    path = _cache_path(symbol)
    if not force_refresh and path.exists():
        return pd.read_parquet(path)

    ticker = yf.Ticker(symbol)
    balance_sheet = ticker.quarterly_balance_sheet
    financials = ticker.quarterly_financials

    if balance_sheet.empty and financials.empty:
        df = pd.DataFrame()
    else:
        # yfinance returns line-items-as-rows, report-dates-as-columns --
        # transpose so report date is the row index, matching every other
        # time series in this codebase.
        combined = pd.concat([balance_sheet.T, financials.T], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        df = combined.sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def get_universe_fundamentals(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch fundamentals for each symbol; symbols with no data (ETFs,
    fetch failures) are skipped rather than failing the whole batch --
    same pattern as data.equities.get_universe_prices.
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = get_fundamentals(symbol)
        except Exception:
            continue
        if not df.empty:
            out[symbol] = df
    return out


def latest_report_as_of(fundamentals: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    """The most recent report row dated on or before `as_of` -- never a
    future report relative to `as_of`. None if no such report exists yet.
    """
    if fundamentals.empty:
        return None
    available = fundamentals.loc[fundamentals.index <= as_of]
    if available.empty:
        return None
    return available.iloc[-1]
