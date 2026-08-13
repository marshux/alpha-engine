"""Read-through parquet cache for per-symbol price history.

Stored at <repo_root>/data/cache/<symbol>.parquet, gitignored — this is a
local performance cache, not tracked data. Cache entries are overwritten
wholesale on a miss rather than merged incrementally: yfinance full-history
downloads are cheap, and range-merge logic is a common source of subtle
off-by-one/gap bugs that aren't worth the complexity here.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"


def read_cached(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cache(symbol: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(symbol))


def covers_range(df: pd.DataFrame, start: str, end: str) -> bool:
    if df.empty:
        return False
    return df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end)
