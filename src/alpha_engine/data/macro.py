"""Macro/market-context data: Treasury yield curve, VIX, and a corporate
credit spread, used by decision/regime.py's HMM regime classifier.
Fetched from FRED's public CSV endpoint (no API key required) and cached
like data/fundamentals.py -- existence-based cache check, whole-file
overwrite, no incremental merge.

Series: DGS10/DGS2 (10y/2y Treasury yield, combined into a yield-curve
slope), VIXCLS (VIX close), BAA/AAA (Moody's Seasoned Baa/Aaa Corporate
Bond Yield, combined into a credit spread).

BAA/AAA, not the more standard-sounding BAMLH0A0HYM2 (ICE BofA US High
Yield Index Option-Adjusted Spread): confirmed directly against FRED's
endpoint that BAMLH0A0HYM2's free/anonymous CSV access only covers a
rolling ~3-year window, not full history (a licensing restriction on that
specific ICE-provided series -- DGS10/DGS2/VIXCLS are all unrestricted).
BAA/AAA are freely available back to 1919, monthly cadence (vs the other
three series' daily cadence) -- get_macro_data's existing `.ffill()`
already handles that difference correctly, propagating each month's
reading forward until the next one. This was a real bug: build_regime_features
used to `.dropna()` across all 8 feature columns, so credit_spread's
missing history before ~2023 silently discarded the ENTIRE feature panel
for every older date, meaning the regime classifier never actually fit on
real data for most of a full 2008-2024 walk-forward run (see
systematic_core_postmortem.md). build_regime_features also independently
no longer drops rows over a macro column gap (fillna(0.0) instead) as a
second line of defense, but fixing the root cause here means credit_spread
is now genuinely populated with real values across the full history
instead of relying on that fallback.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "macro"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _cache_path(series_id: str) -> Path:
    return CACHE_DIR / f"{series_id}.parquet"


def get_fred_series(series_id: str, force_refresh: bool = False) -> pd.Series:
    """A single FRED daily series, indexed by date. FRED represents
    missing/holiday values as '.' in the CSV -- coerced to NaN here;
    callers are expected to forward-fill (see get_macro_data).
    """
    path = _cache_path(series_id)
    if not force_refresh and path.exists():
        return pd.read_parquet(path)[series_id]

    df = pd.read_csv(FRED_CSV_URL.format(series_id=series_id))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.set_index("date").sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df[series_id]


def get_macro_data(force_refresh: bool = False) -> pd.DataFrame:
    """Combined macro panel -- columns: yield_curve (10y-2y spread, in
    percentage points), vix, credit_spread (Baa-Aaa spread, in percentage
    points). Forward-filled to a daily index so callers can safely
    `.loc[:as_of]` slice and take the last available value, the same
    point-in-time access pattern used everywhere else in this codebase
    (see data/fundamentals.py).
    """
    dgs10 = get_fred_series("DGS10", force_refresh)
    dgs2 = get_fred_series("DGS2", force_refresh)
    vix = get_fred_series("VIXCLS", force_refresh)
    baa = get_fred_series("BAA", force_refresh)
    aaa = get_fred_series("AAA", force_refresh)

    combined = pd.DataFrame(
        {
            "yield_curve": dgs10 - dgs2,
            "vix": vix,
            "credit_spread": baa - aaa,
        }
    )
    return combined.sort_index().ffill()
