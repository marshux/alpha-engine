"""Rebuild data/cache/insider/<SYMBOL>.parquet from SEC's official quarterly
bulk Form 3/4/5 datasets:
https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

Supersedes the incremental live-scrape approach (scripts/backfill_insider_full.py
+ data/insider.py's per-symbol full-text-search fetch): SEC publishes every
company's Form 3/4/5 data for a quarter as one ZIP, so the whole 2006-2026
universe history is ~82 requests instead of one (possibly many-request,
many-minute) live search per symbol.

Each quarterly ZIP contains several TSVs; only two matter here:
- SUBMISSION.tsv: one row per filing (ACCESSION_NUMBER, ISSUERCIK,
  ISSUERTRADINGSYMBOL, ...).
- NONDERIV_TRANS.tsv: one row per non-derivative transaction line item
  within a filing (ACCESSION_NUMBER, TRANS_DATE, TRANS_CODE, TRANS_SHARES,
  ...) -- one filing can have several transaction rows, joined via
  ACCESSION_NUMBER.

Matching a filing to a universe symbol uses BOTH normalized
ISSUERTRADINGSYMBOL text AND ISSUERCIK, not text alone. Reason: Berkshire
Hathaway's Form 4s in the bulk data all carry ISSUERTRADINGSYMBOL "BRK.A"
-- confirmed by direct inspection -- even though our universe's "BRK-B"
(Class B) resolves, via the same SEC company_tickers.json map
data/insider.py already caches, to the *same* CIK (0001067983) that owns
those "BRK.A"-labeled filings. Relying on literal ticker text alone would
silently have dropped any symbol whose filings happen to use a
differently-formatted or share-class-variant ticker string -- the same
class of silent-undercount risk the MANUAL_CIK_OVERRIDES fix for XOM
already addressed on the live-scrape side (see data/insider.py). Here,
CIK resolution reuses that same map + override table, so a filing counts
toward a universe symbol if *either* signal matches.

Replaces the existing cache entirely (does not merge with whatever the
live scraper had already fetched) -- this bulk source is authoritative
and complete for the whole 2006-present range, so partial live-scrape
results are strictly superseded, not merged.
"""
from __future__ import annotations

import io
import re
import time
import zipfile

import pandas as pd
import requests

from alpha_engine.config import load_universe, universe_symbols
from alpha_engine.data.insider import (
    CACHE_DIR,
    MANUAL_CIK_OVERRIDES,
    OPEN_MARKET_CODES,
    USER_AGENT,
    _load_ticker_to_cik_map,
)

BULK_INDEX_URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
REQUEST_PAUSE_SECONDS = 1.0  # ~82 requests total -- no real rate-limit concern, still a reasonable citizen


def _discover_quarterly_zip_urls() -> list[str]:
    """Scrapes the actual href list from the bulk-data index page rather
    than assuming one uniform URL pattern -- the most recent quarter is
    hosted under a different path prefix
    (/files/datastandardsinnovation/... instead of /files/structureddata/...),
    confirmed by inspecting the live page.
    """
    resp = requests.get(BULK_INDEX_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    hrefs = re.findall(r'href="([^"]*_form345\.zip)"', resp.text)
    urls = sorted(
        {("https://www.sec.gov" + h) if h.startswith("/") else h for h in hrefs}
    )
    return urls


def _normalize_ticker(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def main() -> None:
    universe = load_universe()
    symbols = universe_symbols(universe)

    ticker_cik_map = _load_ticker_to_cik_map()
    symbol_by_cik: dict[str, str] = {}
    symbol_by_ticker: dict[str, str] = {}
    for symbol in symbols:
        cik = MANUAL_CIK_OVERRIDES.get(symbol.upper()) or ticker_cik_map.get(symbol.upper())
        if cik:
            symbol_by_cik[cik] = symbol
        symbol_by_ticker[_normalize_ticker(symbol)] = symbol

    urls = _discover_quarterly_zip_urls()
    print(f"Discovered {len(urls)} quarterly ZIPs", flush=True)

    accumulated: dict[str, list[dict]] = {symbol: [] for symbol in symbols}

    t0 = time.time()
    for i, url in enumerate(urls):
        q_t0 = time.time()
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))

        sub = pd.read_csv(
            zf.open("SUBMISSION.tsv"),
            sep="\t",
            usecols=["ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL"],
            dtype=str,
        )
        sub["ISSUERCIK"] = sub["ISSUERCIK"].str.zfill(10)
        by_cik = sub["ISSUERCIK"].map(symbol_by_cik)
        by_ticker = sub["ISSUERTRADINGSYMBOL"].map(_normalize_ticker).map(symbol_by_ticker)
        sub["_symbol"] = by_cik.fillna(by_ticker)
        matched_sub = sub.loc[sub["_symbol"].notna(), ["ACCESSION_NUMBER", "_symbol"]]

        if matched_sub.empty:
            print(
                f"[{i+1}/{len(urls)}] {url.rsplit('/', 1)[-1]}: 0 matched filings "
                f"({time.time()-q_t0:.1f}s, {time.time()-t0:.0f}s elapsed total)",
                flush=True,
            )
            time.sleep(REQUEST_PAUSE_SECONDS)
            continue

        trans = pd.read_csv(
            zf.open("NONDERIV_TRANS.tsv"),
            sep="\t",
            usecols=["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES"],
            dtype=str,
        )
        trans = trans[trans["TRANS_CODE"].isin(OPEN_MARKET_CODES)]

        merged = matched_sub.merge(trans, on="ACCESSION_NUMBER", how="inner")
        merged = merged.dropna(subset=["TRANS_DATE", "TRANS_SHARES"])

        for symbol, group in merged.groupby("_symbol"):
            for date, code, shares_text in zip(group["TRANS_DATE"], group["TRANS_CODE"], group["TRANS_SHARES"]):
                try:
                    shares = float(shares_text)
                except (TypeError, ValueError):
                    continue
                accumulated[symbol].append({"date": date, "code": code, "shares": shares})

        print(
            f"[{i+1}/{len(urls)}] {url.rsplit('/', 1)[-1]}: {len(merged)} matched txns across "
            f"{merged['_symbol'].nunique()} symbols ({time.time()-q_t0:.1f}s, {time.time()-t0:.0f}s elapsed total)",
            flush=True,
        )
        time.sleep(REQUEST_PAUSE_SECONDS)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for existing in CACHE_DIR.glob("*.parquet"):
        existing.unlink()

    total_txns = 0
    covered = 0
    bad_date_count = 0
    for symbol in symbols:
        rows = accumulated[symbol]
        if rows:
            df = pd.DataFrame(rows)
            # errors="coerce": SEC's raw bulk data has occasional filer
            # typos in TRANS_DATE (observed in practice: "25-JUL-0025",
            # a 4-digit-year truncation that overflows pandas' bounded
            # nanosecond timestamp range) -- these are data-entry errors
            # in the original filing, not something a stricter parse
            # would recover; drop them rather than let one bad row abort
            # the whole rebuild.
            parsed = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
            bad = parsed.isna()
            if bad.any():
                bad_date_count += int(bad.sum())
                print(
                    f"  {symbol}: dropping {int(bad.sum())} row(s) with unparseable TRANS_DATE "
                    f"(e.g. {df.loc[bad, 'date'].iloc[0]!r})",
                    flush=True,
                )
            df["date"] = parsed
            df = df.dropna(subset=["date"]).set_index("date").sort_index()[["code", "shares"]]
        else:
            df = pd.DataFrame(columns=["code", "shares"])
        if not df.empty:
            covered += 1
        total_txns += len(df)
        path = CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"
        df.to_parquet(path)

    if bad_date_count:
        print(f"\n{bad_date_count} total row(s) dropped for unparseable TRANS_DATE", flush=True)

    elapsed = time.time() - t0
    print(
        f"\nDONE in {elapsed:.0f}s -- {len(symbols)} symbols processed, "
        f"{covered} with data, {total_txns} total transactions",
        flush=True,
    )


if __name__ == "__main__":
    main()
