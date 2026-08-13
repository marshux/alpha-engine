"""Equity/ETF OHLCV fetch via yfinance, backed by the parquet cache in cache.py."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from alpha_engine.data.cache import covers_range, read_cached, write_cache


def get_prices(symbol: str, start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV for `symbol` between `start` and `end` (inclusive), adjusted for splits/dividends."""
    if not force_refresh:
        cached = read_cached(symbol)
        if cached is not None and covers_range(cached, start, end):
            return cached.loc[start:end]

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {symbol} between {start} and {end}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"

    write_cache(symbol, df)
    return df.loc[start:end]


def get_universe_prices(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch prices for each symbol; symbols with no data in range are skipped, not fatal."""
    prices: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            prices[symbol] = get_prices(symbol, start, end)
        except ValueError:
            continue
    return prices
