"""Earnings 8-K press-release text + Loughran-McDonald sentiment scoring.

Correction from `systematic_core_action_plan (4).md`: the plan describes
this as scoring "earnings call transcripts" filed on SEC EDGAR. Verified
directly against a real filing (Apple, accession 0000320193-26-000005):
the 8-K's Item 2.02 exhibit EX-99.1 is the **earnings press release**,
not a call transcript -- the press release text itself states the actual
conference call happens hours *later* the same day. SEC does not require
call transcripts to be filed at all. This module scores the press
release's language (financial results + management quotes), a real and
legitimate text source, just not what the plan describes. The "leading
indicator vs. `earnings_revision`'s analyst reactions" framing still
roughly holds -- the press release is the first public disclosure,
released essentially simultaneously with (or just before) the call --
just not for the reason the plan states.

Pipeline, both stages real SEC endpoints:
1. Resolve symbol -> CIK via `data.insider._load_ticker_to_cik_map`/
   `MANUAL_CIK_OVERRIDES` (same mapping already built for
   `insider_buying`, reused rather than duplicated).
2. Fetch the company's SEC submissions JSON (`data.sec.gov/submissions/`)
   -- paginated: the "recent" bucket covers roughly the company's last
   ~1000 filings; older filings live in separate, linked JSON files with
   the identical flat column structure, fetched only if `recent` doesn't
   reach back far enough for the requested date range. Filtered to 8-K
   filings whose `items` include "2.02" (Results of Operations and
   Financial Condition -- the standard earnings-8-K classification,
   confirmed present on every real earnings 8-K checked during
   development).
3. For each qualifying 8-K, parse its `-index.html` (a real HTML table
   with a "Type" column, e.g. "EX-99.1") to reliably locate the exhibit
   document -- not a filename-pattern guess, since naming conventions
   vary by filer.
4. Score the exhibit's visible text against the Loughran-McDonald Master
   Dictionary (see `_load_loughran_mcdonald_dictionary`).

Cached to parquet, same pattern as every other data module here -- the
per-filing fetch/parse cost is paid once, not on every backtest run.
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd

from alpha_engine.data.insider import MANUAL_CIK_OVERRIDES, USER_AGENT, _get_json, _get_text, _load_ticker_to_cik_map

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "earnings_sentiment"
LM_DICT_CACHE_PATH = REPO_ROOT / "data" / "cache" / "loughran_mcdonald_dictionary.parquet"

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
OLDER_SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}-index.html"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"
# Direct-download form of the LM Master Dictionary's public Google Drive
# share link (from https://sraf.nd.edu/loughranmcdonald-master-dictionary/,
# verified directly before use -- real CSV, ~9MB, Word/Negative/Positive/
# Uncertainty/... columns).
LM_DICT_URL = "https://drive.google.com/uc?export=download&id=1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"

EARNINGS_ITEM_CODE = "2.02"
REQUEST_PAUSE_SECONDS = 0.15


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol.replace('/', '_')}.parquet"


def _load_loughran_mcdonald_dictionary(force_refresh: bool = False) -> dict[str, dict[str, bool]]:
    """`{WORD: {"negative": bool, "positive": bool, "uncertainty": bool}}`
    -- the dictionary's Negative/Positive/Uncertainty columns encode the
    *year* a word was tagged into that category (e.g. "2009"), not a
    plain 0/1 flag; any non-zero value means "tagged," 0 means "not
    tagged" (verified directly against the real downloaded file before
    writing this parser). Cached once (~9MB, doesn't change often --
    existence-based cache check, same pattern as everywhere else).
    """
    if not force_refresh and LM_DICT_CACHE_PATH.exists():
        return pd.read_parquet(LM_DICT_CACHE_PATH).set_index("word").to_dict("index")

    # Google Drive intermittently serves an anti-bot "virus scan warning"
    # HTML interstitial with a 200 status instead of the real file --
    # observed directly during development, roughly every other request,
    # unrelated to request rate (a single isolated request can trigger
    # it). This is a wrong-content-with-a-200 failure, not an HTTP error,
    # so it isn't caught by _get_text's own retry logic -- validate the
    # content shape here and retry a few times ourselves.
    csv_text = None
    last_error: Exception | None = None
    for attempt in range(5):
        candidate = _get_text(LM_DICT_URL)
        if candidate.lstrip().startswith("Word,"):
            csv_text = candidate
            break
        last_error = ValueError(f"LM dictionary fetch returned non-CSV content (attempt {attempt + 1}/5)")
        time.sleep(2.0 * (attempt + 1))
    if csv_text is None:
        raise last_error

    df = pd.read_csv(io.StringIO(csv_text), usecols=["Word", "Negative", "Positive", "Uncertainty"])
    df["word"] = df["Word"].str.upper()
    df["negative"] = df["Negative"].astype(int) != 0
    df["positive"] = df["Positive"].astype(int) != 0
    df["uncertainty"] = df["Uncertainty"].astype(int) != 0
    out = df[["word", "negative", "positive", "uncertainty"]]

    LM_DICT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LM_DICT_CACHE_PATH)
    return out.set_index("word").to_dict("index")


def _fetch_earnings_8k_filings(cik: str, start_date: str, end_date: str) -> list[dict]:
    """Every Item 2.02 8-K filing for `cik` in `[start_date, end_date]`,
    as `{accession, filing_date, primary_doc}`, oldest first. Paginates
    into SEC's older, linked submissions files only if `recent` doesn't
    reach back far enough.
    """
    payload = _get_json(SUBMISSIONS_URL.format(cik=cik), params={})
    recent = payload["filings"]["recent"]
    filings = _extract_earnings_8ks(recent, start_date, end_date)

    recent_dates = recent.get("filingDate", [])
    recent_earliest = min(recent_dates) if recent_dates else None
    if recent_earliest is not None and recent_earliest > start_date:
        for older_file in payload["filings"].get("files", []):
            if older_file["filingTo"] < start_date:
                continue
            older_payload = _get_json(OLDER_SUBMISSIONS_URL.format(name=older_file["name"]), params={})
            filings.extend(_extract_earnings_8ks(older_payload, start_date, end_date))
            time.sleep(REQUEST_PAUSE_SECONDS)

    filings.sort(key=lambda f: f["filing_date"])
    return filings


def _extract_earnings_8ks(filings_block: dict, start_date: str, end_date: str) -> list[dict]:
    out = []
    forms = filings_block.get("form", [])
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = (filings_block["items"][i] or "").split(",")
        if EARNINGS_ITEM_CODE not in items:
            continue
        filing_date = filings_block["filingDate"][i]
        if not (start_date <= filing_date <= end_date):
            continue
        out.append({
            "accession": filings_block["accessionNumber"][i],
            "filing_date": filing_date,
            "primary_doc": filings_block["primaryDocument"][i],
        })
    return out


def _find_exhibit_99_1_filename(cik: str, accession: str) -> str | None:
    """Parses the filing's real `-index.html` table (has a "Type" column,
    e.g. "EX-99.1") to reliably locate the exhibit document -- filename
    conventions vary by filer, so this doesn't guess from the name.
    Falls back from EX-99.1 to any EX-99.x if no exact ".1" exists (some
    filers use EX-99 or EX-99.2 for the earnings release instead).
    """
    cik_int = int(cik)
    accession_nodash = accession.replace("-", "")
    url = FILING_INDEX_URL.format(cik_int=cik_int, accession_nodash=accession_nodash, accession=accession)
    html = _get_text(url)

    candidates: dict[str, str] = {}
    for row_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_match.group(1), re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        for j, cell in enumerate(cells):
            if re.fullmatch(r"EX-99(\.\d+)?", cell, re.IGNORECASE):
                filename_cell = next((c for c in cells if re.search(r"\.(htm|txt)$", c, re.IGNORECASE)), None)
                if filename_cell:
                    candidates[cell.upper()] = filename_cell.split()[-1]

    if "EX-99.1" in candidates:
        return candidates["EX-99.1"]
    return next(iter(candidates.values()), None)


def _html_to_text(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&#8217;|&#8216;", "'", text)
    text = re.sub(r"&#8220;|&#8221;", '"', text)
    text = re.sub(r"&#8212;|&#8211;", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _score_text(text: str, lm_dict: dict[str, dict[str, bool]]) -> dict:
    words = re.findall(r"[A-Za-z']+", text.upper())
    if not words:
        return {"positive_score": 0.0, "negative_score": 0.0, "uncertainty_score": 0.0}
    pos = sum(1 for w in words if lm_dict.get(w, {}).get("positive"))
    neg = sum(1 for w in words if lm_dict.get(w, {}).get("negative"))
    unc = sum(1 for w in words if lm_dict.get(w, {}).get("uncertainty"))
    n = len(words)
    return {"positive_score": pos / n, "negative_score": neg / n, "uncertainty_score": unc / n}


def get_earnings_sentiment_history(
    symbol: str, start_date: str = "2015-01-01", end_date: str | None = None, force_refresh: bool = False
) -> pd.DataFrame:
    """Per-filing sentiment scores for `symbol`'s earnings 8-Ks, indexed
    by filing date ascending. Columns: `positive_score`, `negative_score`,
    `uncertainty_score` (all word-count fractions). Empty DataFrame if no
    CIK resolves or no qualifying filings exist in the window.
    """
    path = _cache_path(symbol)
    if not force_refresh and path.exists():
        return pd.read_parquet(path)

    end_date = end_date or pd.Timestamp.now().date().isoformat()
    cik = MANUAL_CIK_OVERRIDES.get(symbol.upper()) or _load_ticker_to_cik_map().get(symbol.upper())
    if cik is None:
        df = pd.DataFrame(columns=["positive_score", "negative_score", "uncertainty_score"])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    lm_dict = _load_loughran_mcdonald_dictionary()
    filings = _fetch_earnings_8k_filings(cik, start_date, end_date)

    rows: list[dict] = []
    for filing in filings:
        try:
            exhibit_filename = _find_exhibit_99_1_filename(cik, filing["accession"])
            if exhibit_filename is None:
                continue
            accession_nodash = filing["accession"].replace("-", "")
            url = ARCHIVES_URL.format(cik_int=int(cik), accession_nodash=accession_nodash, filename=exhibit_filename)
            html = _get_text(url)
            text = _html_to_text(html)
            scores = _score_text(text, lm_dict)
            rows.append({"date": filing["filing_date"], **scores})
        except Exception:
            continue
        time.sleep(REQUEST_PAUSE_SECONDS)

    if not rows:
        df = pd.DataFrame(columns=["positive_score", "negative_score", "uncertainty_score"])
    else:
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.pop("date"))
        df.index.name = "date"
        df = df.sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def get_cached_earnings_sentiment(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Cache-only read -- never fetches over the network. Same rationale
    as `data.insider.get_cached_insider_transactions`: a live fetch
    across the full universe is a genuinely multi-hour cost, so backtest/
    validation runs should read whatever's already been warmed rather
    than trigger more live fetches."""
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = _cache_path(symbol)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if not df.empty:
            out[symbol] = df
    return out


def get_universe_earnings_sentiment(
    symbols: list[str], start_date: str = "2015-01-01", end_date: str | None = None
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = get_earnings_sentiment_history(symbol, start_date, end_date)
        except Exception:
            continue
        if not df.empty:
            out[symbol] = df
        time.sleep(REQUEST_PAUSE_SECONDS)
    return out


def latest_earnings_sentiment_as_of(sentiment: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    """The most recent scored filing on or before `as_of`. None if none
    exists yet -- same point-in-time pattern used across this codebase."""
    if sentiment.empty:
        return None
    available = sentiment.loc[sentiment.index <= as_of]
    if available.empty:
        return None
    return available.iloc[-1]
