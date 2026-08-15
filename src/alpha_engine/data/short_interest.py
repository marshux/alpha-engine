"""Bi-weekly consolidated short interest, via FINRA's real (verified)
public data API -- not the URL pattern in systematic_core_action_plan.md,
which pointed at Reg SHO's *daily short volume* file (a same-day trading-
flow measure) under a mistaken "short interest" label. That file returned
403 at the exact path given; the genuine bi-weekly short-*position*
dataset (total shares currently sold short, days-to-cover) lives at a
different FINRA API endpoint entirely, found and verified directly
against FINRA's API (POST, JSON body, no key required) before writing
this module -- same "verify before trusting a data source" discipline
that `data/macro.py`'s BAMLH0A0HYM2 incident (see
systematic_core_postmortem.md) established the hard way.

Coverage caveat: this endpoint's history only extends back to ~2018, not
the full 2008-2024 range this project's walk-forward validation covers.
Older dates get no short-interest data at all -- signals/short_interest.py
degrades that to a neutral score, same "coverage is thin for older
backtest dates" posture data/fundamentals.py already documents.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "short_interest"

FINRA_API_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
MAX_ROWS_PER_REQUEST = 5000  # server-enforced cap, confirmed empirically


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"


def get_short_interest(symbol: str, force_refresh: bool = False) -> pd.DataFrame:
    """All available bi-weekly short-interest reports for `symbol`,
    indexed by settlement date ascending. Columns: `short_position`
    (shares currently sold short), `days_to_cover`. Empty DataFrame if
    FINRA has no data for this symbol (e.g. it wasn't listed yet, or
    coverage doesn't reach this far back -- see module docstring).
    """
    path = _cache_path(symbol)
    if not force_refresh and path.exists():
        return pd.read_parquet(path)

    csv_text = _post_csv(
        FINRA_API_URL,
        {
            "limit": MAX_ROWS_PER_REQUEST,
            "compareFilters": [{"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": symbol}],
        },
    )
    response = pd.read_csv(io.StringIO(csv_text)) if csv_text.strip() else pd.DataFrame()

    if response.empty:
        df = pd.DataFrame(columns=["short_position", "days_to_cover"])
    else:
        df = pd.DataFrame(
            {
                "short_position": pd.to_numeric(response["currentShortPositionQuantity"], errors="coerce"),
                "days_to_cover": pd.to_numeric(response["daysToCoverQuantity"], errors="coerce"),
            }
        )
        df.index = pd.to_datetime(response["settlementDate"])
        df.index.name = "date"
        df = df.sort_index()
        # FINRA uses 999.99 as a sentinel for "not meaningful" (e.g. zero
        # average daily volume) rather than a real days-to-cover value --
        # treat it as missing, not as an extreme outlier that would
        # otherwise dominate the cross-sectional z-score.
        df.loc[df["days_to_cover"] >= 999, "days_to_cover"] = float("nan")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def get_universe_short_interest(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch short interest for each symbol; symbols with no data (fetch
    failures, no coverage) are skipped rather than failing the whole
    batch -- same pattern as data.equities.get_universe_prices.
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = get_short_interest(symbol)
        except Exception:
            continue
        if not df.empty:
            out[symbol] = df
    return out


def latest_short_interest_as_of(short_interest: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    """The most recent report dated on or before `as_of` -- never a
    future report relative to `as_of`. None if no such report exists yet
    (same shape as data.fundamentals.latest_report_as_of).
    """
    if short_interest.empty:
        return None
    available = short_interest.loc[short_interest.index <= as_of]
    if available.empty:
        return None
    return available.iloc[-1]


def _post_csv(url: str, body: dict) -> str:
    """Thin wrapper around a POST request returning the raw CSV response
    body as text -- kept as a separate function (not inlined into
    get_short_interest) purely so tests can monkeypatch just the network
    call without needing to mock pandas' CSV parsing too. A plain POST
    with a JSON body, not the GET-only `pd.read_csv(url)` pattern
    data/macro.py uses -- FINRA's API requires filters in the request
    body, not query-string params.
    """
    import requests

    response = requests.post(url, json=body, timeout=30)
    response.raise_for_status()
    return response.text
