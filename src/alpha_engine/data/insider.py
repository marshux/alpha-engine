"""Insider Form 4 open-market transactions, via SEC EDGAR's real full-text
search API + individual filing XML parsing -- verified directly against
SEC's endpoints before writing this module (same discipline as
data/short_interest.py's incident-driven verification habit; see
systematic_core_postmortem.md).

Two-stage pipeline, both stages real SEC EDGAR endpoints:
1. `https://efts.sec.gov/LATEST/search-index` (full-text search) locates
   every Form 4 filing mentioning a symbol within a date range -- this
   endpoint returns filing *metadata* only (who filed, when, accession
   number), not the actual transaction detail.
2. Each filing's own XML document (fetched from
   `https://www.sec.gov/Archives/edgar/data/...`) is parsed for
   `nonDerivativeTransaction` entries with transaction code `P` (open-
   market purchase) or `S` (open-market sale) -- explicitly excluding
   `A` (grants/awards), `M` (option exercises), `F` (tax withholding),
   `G` (gifts), and other non-open-market codes, which aren't a genuine
   read on insider conviction the way a real purchase/sale is.

SEC requires a descriptive User-Agent identifying the requester on every
request (an unidentified/generic User-Agent gets throttled or blocked) --
see USER_AGENT below; update the contact email if this is ever run
against SEC's production endpoints outside this research context.

Rate limits: SEC's fair-access policy asks for a modest, paced request
rate (their own guidance: keep it well under ~10 req/sec) -- fetching a
symbol's *entire* available history is therefore a real, possibly
multi-minute-per-symbol cost the first time (many individual filing
documents per company per year), which is exactly why this follows the
same cache-once, existence-based-check pattern as every other data
module in this codebase: the cost is paid once, not on every backtest
run.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "insider"
TICKER_CIK_CACHE_PATH = REPO_ROOT / "data" / "cache" / "sec_ticker_cik_map.json"

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"
TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "alpha-engine-research contact@alpha-engine.local"

OPEN_MARKET_CODES = {"P", "S"}  # purchase, sale -- excludes grants/exercises/gifts/etc.
# ~8.5 req/sec aggregate (1/0.1176) -- comfortably under SEC's ~10 req/sec
# fair-access guidance, and the only knob that determines request rate (see
# _throttle() below): with the backfill driver's modest thread pool, this
# is what makes concurrency worth it, since a single thread's own per-
# request overhead (parsing, etc.) wouldn't otherwise saturate a cap this
# tight. Set to 0 to disable pacing entirely (tests do this).
REQUEST_PAUSE_SECONDS = 0.1176

# Manual overrides for symbols where SEC's live ticker->CIK mapping
# (`_load_ticker_to_cik_map`) resolves to the wrong entity -- specifically,
# a company that reorganized into a holding-company structure, where SEC's
# *current* ticker ownership snapshot points at the new holding entity
# (which has no Form-4 history of its own) instead of the legacy operating
# entity that actually holds the multi-decade insider-trading record.
#
# Discovered via XOM during the 204-symbol backfill: SEC's live mapping put
# "XOM" -> CIK 0002115436 ("ExxonMobil Holdings Corp"), which has zero Form
# 4 filings (`insiderTransactionForIssuerExists: 0` at
# https://data.sec.gov/submissions/CIK0002115436.json) -- silently producing
# a false "0 open-market transactions" result indistinguishable from a
# symbol with genuinely no insider activity (e.g. BRK-B, where 0 is real).
# The actual history sits under CIK 0000034088 ("EXXON MOBIL CORP"), which
# still has the filings but no longer carries the ticker in SEC's current
# mapping (confirmed via https://data.sec.gov/submissions/CIK0000034088.json).
#
# SCOPE NOTE: only XOM has been verified and fixed so far. Other old,
# large-cap symbols in config/universe.yaml that went through a similar
# holding-company reorganization could have the same silent false-zero and
# have *not* been audited for this -- that's a separate investigation, not
# done as part of this backfill. To check a suspect symbol: fetch
# https://data.sec.gov/submissions/CIK{cik}.json for whatever CIK
# `_load_ticker_to_cik_map()` currently resolves it to, and look at
# `insiderTransactionForIssuerExists` -- if it's 0 for an old, large
# company, the ticker has likely been reassigned the same way XOM's was.
MANUAL_CIK_OVERRIDES = {
    "XOM": "0000034088",
}


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"


def _load_ticker_to_cik_map(force_refresh: bool = False) -> dict[str, str]:
    """SEC's own ticker -> 10-digit zero-padded CIK mapping
    (`TICKER_CIK_URL`), cached once to `TICKER_CIK_CACHE_PATH` (~1MB,
    ~10k entries, doesn't change often -- an existence-based cache check
    is enough, same pattern as everywhere else in this codebase).

    Exists so `_search_form4_filings` can query EDGAR by CIK instead of
    by ticker text: a bare `q="{symbol}"` phrase search is a full-text
    match against filing content, not a symbol-code filter, so a short
    ticker (e.g. "V", "F") that also happens to be a common word/letter
    matches thousands of irrelevant filings that mention that character
    anywhere -- 10,000+ hits (Elasticsearch's reporting cap, the true
    count is likely far higher) for "V" alone, observed in practice
    during a real backfill run, versus 891 genuine hits when the same
    query is scoped to Visa's actual CIK. CIK-based search is both more
    accurate (no incidental-text false positives for *any* ticker, not
    just short ones) and dramatically faster.
    """
    if not force_refresh and TICKER_CIK_CACHE_PATH.exists():
        return json.loads(TICKER_CIK_CACHE_PATH.read_text())

    payload = _get_json(TICKER_CIK_URL, params={})
    mapping = {entry["ticker"].upper(): f"{entry['cik_str']:010d}" for entry in payload.values()}

    TICKER_CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TICKER_CIK_CACHE_PATH.write_text(json.dumps(mapping))
    return mapping


def _search_form4_filings(symbol: str, start_date: str, end_date: str) -> list[dict]:
    """Every Form 4 filing SEC's full-text search associates with
    `symbol` in `[start_date, end_date]` -- paginated (10 hits/page),
    stops once a page returns fewer hits than requested.

    Queries by CIK (`_load_ticker_to_cik_map`) whenever `symbol` resolves
    to one -- see that function's docstring for why this matters, not
    just a minor optimization. Falls back to the old ticker-text phrase
    query only for symbols absent from SEC's mapping (e.g. some ETFs),
    where a text match is the only option available at all.
    """
    cik = MANUAL_CIK_OVERRIDES.get(symbol.upper()) or _load_ticker_to_cik_map().get(symbol.upper())
    base_params = {
        "forms": "4",
        "dateRange": "custom",
        "startdt": start_date,
        "enddt": end_date,
    }
    if cik is not None:
        base_params["ciks"] = cik
    else:
        base_params["q"] = f'"{symbol}"'

    hits: list[dict] = []
    page_from = 0
    while True:
        payload = _get_json(SEARCH_URL, params={**base_params, "from": page_from})
        page_hits = payload.get("hits", {}).get("hits", [])
        hits.extend(page_hits)
        if len(page_hits) < 10:
            break
        page_from += 10
    return hits


def _parse_form4_xml(xml_text: str, expected_symbol: str) -> list[dict]:
    """Open-market (P/S) non-derivative transactions from one Form 4's
    XML body. Cross-checks `issuerTradingSymbol` against
    `expected_symbol` -- the full-text search query can occasionally
    surface a filing where the symbol appears incidentally rather than
    as the actual issuer (e.g. mentioned in a footnote), so this is a
    defensive filter, not just a formality.
    """
    root = ET.fromstring(xml_text)
    issuer_symbol = root.findtext("issuer/issuerTradingSymbol", default="")
    if issuer_symbol.strip().upper() != expected_symbol.strip().upper():
        return []

    transaction_date = root.findtext(
        "nonDerivativeTable/nonDerivativeTransaction/transactionDate/value", default=None
    )

    out = []
    for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = txn.findtext("transactionCoding/transactionCode", default="")
        if code not in OPEN_MARKET_CODES:
            continue
        date = txn.findtext("transactionDate/value", default=transaction_date)
        shares_text = txn.findtext("transactionAmounts/transactionShares/value", default=None)
        if date is None or shares_text is None:
            continue
        try:
            shares = float(shares_text)
        except ValueError:
            continue
        out.append({"date": date, "code": code, "shares": shares})
    return out


def get_insider_transactions(
    symbol: str, start_date: str = "2005-01-01", end_date: str | None = None, force_refresh: bool = False
) -> pd.DataFrame:
    """All open-market (P/S) insider transactions for `symbol` in
    `[start_date, end_date]` (default: today), one row per transaction,
    indexed by transaction date ascending. Columns: `code` ('P'/'S'),
    `shares`. Empty DataFrame if SEC has no matching filings (e.g. the
    symbol wasn't public yet, or genuinely had no open-market insider
    activity in the window).
    """
    path = _cache_path(symbol)
    if not force_refresh and path.exists():
        return pd.read_parquet(path)

    end_date = end_date or pd.Timestamp.now().date().isoformat()
    filings = _search_form4_filings(symbol, start_date, end_date)

    rows: list[dict] = []
    for hit in filings:
        source = hit.get("_source", {})
        ciks = source.get("ciks", [])
        issuer_cik = ciks[-1] if ciks else None  # issuer is conventionally listed last
        accession = source.get("adsh", "").replace("-", "")
        filename = hit.get("_id", "").split(":")[-1]
        if not issuer_cik or not accession or not filename:
            continue

        url = ARCHIVES_URL.format(cik=int(issuer_cik), accession_nodash=accession, filename=filename)
        try:
            xml_text = _get_text(url)
            rows.extend(_parse_form4_xml(xml_text, symbol))
        except Exception:
            continue

    if not rows:
        df = pd.DataFrame(columns=["code", "shares"])
    else:
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.pop("date"))
        df.index.name = "date"
        df = df.sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def get_cached_insider_transactions(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Like `get_universe_insider_transactions`, but never fetches over
    the network -- only returns symbols that already have a cached
    parquet file. Use this when a fresh live fetch isn't practical (SEC's
    fair-access pacing means a full-universe fetch can take hours -- see
    `get_universe_insider_transactions`'s docstring) and partial, already-
    warmed coverage is preferable to either an unbounded wait or an
    exception. Symbols with no cache entry are simply absent from the
    result, same "missing coverage degrades gracefully" posture as
    everywhere else in this codebase.
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = _cache_path(symbol)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if not df.empty:
            out[symbol] = df
    return out


def get_universe_insider_transactions(
    symbols: list[str], start_date: str = "2005-01-01", end_date: str | None = None
) -> dict[str, pd.DataFrame]:
    """Fetch insider transactions for each symbol; symbols with no data
    (fetch failures, no filings) are skipped rather than failing the
    whole batch -- same pattern as data.equities.get_universe_prices.

    No explicit pause between symbols here -- `_throttle()` (see its
    docstring) already enforces the minimum request interval globally,
    at the level of every individual `_get_json`/`_get_text` call, so a
    new symbol's first request is paced against the previous symbol's
    last request exactly the same as any two requests within one symbol's
    own search. This is what originally motivated a per-symbol pause here
    (found in practice that a large-cap company's hundreds of back-to-back
    paginated requests could make the next symbol's very first request
    look like a burst to SEC's search-index endpoint, triggering transient
    500s -- confirmed: retrying moments later succeeded) -- centralizing
    the throttle covers that case without a separate sleep.
    """
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = get_insider_transactions(symbol, start_date, end_date)
        except Exception:
            continue
        if not df.empty:
            out[symbol] = df
    return out


MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

_rate_limit_lock = threading.Lock()
_next_request_not_before = 0.0


def _throttle() -> None:
    """Block the calling thread until it's safe to issue the next SEC
    request, enforcing a minimum REQUEST_PAUSE_SECONDS interval between the
    *start* of any two requests -- globally, across every thread, not just
    within one. This is the thread-safe generalization of the old "sleep
    after each call" pattern that used to live at each individual call
    site (`_search_form4_filings`'s page loop, `get_insider_transactions`'s
    per-filing loop, `get_universe_insider_transactions`'s per-symbol
    loop): centralizing it here, called once at the top of every actual
    HTTP attempt in `_get_json`/`_get_text` (retries included), means
    callers -- including a `ThreadPoolExecutor` mapping over multiple
    symbols concurrently -- never need to know pacing exists at all; the
    cap is enforced no matter how many threads are issuing requests.

    With exactly one caller thread this reduces to the same 1/REQUEST_PAUSE_SECONDS
    ceiling the old per-call-site sleeps produced. REQUEST_PAUSE_SECONDS <= 0
    disables it entirely (skips the lock too) -- same escape hatch
    tests/test_insider.py already relies on.
    """
    if REQUEST_PAUSE_SECONDS <= 0:
        return
    global _next_request_not_before
    with _rate_limit_lock:
        now = time.monotonic()
        wait = _next_request_not_before - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _next_request_not_before = now + REQUEST_PAUSE_SECONDS


def _get_json(url: str, params: dict) -> dict:
    """Kept as a separate function (not inlined) so tests can monkeypatch
    just the network call. Retries on transient 5xx (observed in practice:
    SEC's search-index endpoint occasionally 500s on an isolated request
    with no surrounding burst -- distinct from the sustained rate-limit
    incident this module's docstring describes, which needed pacing, not
    retries. A large-cap symbol's search alone can span hundreds of
    paginated requests, so one unretried transient error would otherwise
    abort that entire symbol's fetch partway through.)."""
    import requests

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            _throttle()
            response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code < 500:
                raise  # a 4xx won't be fixed by retrying
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error


def _get_text(url: str) -> str:
    import requests

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            _throttle()
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response is not None and e.response.status_code < 500:
                raise
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error
