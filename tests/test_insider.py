"""Tests for data/insider.py's SEC EDGAR fetch/parse pipeline and
signals/insider_buying.py's net-buying signal -- mocked network calls
(the underlying pipeline was verified separately against real SEC EDGAR
data during development; see data/insider.py's module docstring), no
real HTTP requests in the test suite itself.
"""
from __future__ import annotations

import pandas as pd
import pytest

from alpha_engine.data import insider as insider_data
from alpha_engine.signals import insider_buying


FORM4_XML_TEMPLATE = """<?xml version="1.0"?>
<ownershipDocument>
    <issuer>
        <issuerCik>0000320193</issuerCik>
        <issuerName>Apple Inc.</issuerName>
        <issuerTradingSymbol>{symbol}</issuerTradingSymbol>
    </issuer>
    <nonDerivativeTable>
        {transactions}
    </nonDerivativeTable>
</ownershipDocument>
"""

TRANSACTION_TEMPLATE = """
<nonDerivativeTransaction>
    <transactionDate><value>{date}</value></transactionDate>
    <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>{shares}</value></transactionShares></transactionAmounts>
</nonDerivativeTransaction>
"""


def _xml(symbol: str, transactions: list[tuple[str, str, float]]) -> str:
    txn_xml = "".join(TRANSACTION_TEMPLATE.format(date=d, code=c, shares=s) for d, c, s in transactions)
    return FORM4_XML_TEMPLATE.format(symbol=symbol, transactions=txn_xml)


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(insider_data, "CACHE_DIR", tmp_path / "insider")
    monkeypatch.setattr(insider_data, "REQUEST_PAUSE_SECONDS", 0.0)
    # Pre-seed an empty ticker->CIK cache so the real _load_ticker_to_cik_map
    # hits its cache path (no network call) and every existing test's fake
    # _get_json only ever sees the ticker-text-query shape (no "ciks"
    # param), matching pre-CIK-lookup behavior. TestLoadTickerToCikMap and
    # TestCikBasedSearch below override this per-test to exercise the real
    # fetch/CIK-resolution paths.
    cik_cache_path = tmp_path / "ticker_cik.json"
    cik_cache_path.write_text("{}")
    monkeypatch.setattr(insider_data, "TICKER_CIK_CACHE_PATH", cik_cache_path)


class TestParseForm4Xml:
    def test_extracts_open_market_purchase_and_sale(self):
        xml = _xml("AAPL", [("2024-01-15", "P", 100.0), ("2024-01-16", "S", 50.0)])
        rows = insider_data._parse_form4_xml(xml, "AAPL")
        assert rows == [
            {"date": "2024-01-15", "code": "P", "shares": 100.0},
            {"date": "2024-01-16", "code": "S", "shares": 50.0},
        ]

    def test_excludes_non_open_market_codes(self):
        # A: award/grant, M: option exercise -- neither is a genuine
        # open-market conviction signal.
        xml = _xml("AAPL", [("2024-01-15", "A", 1000.0), ("2024-01-16", "M", 500.0)])
        rows = insider_data._parse_form4_xml(xml, "AAPL")
        assert rows == []

    def test_rejects_mismatched_issuer_symbol(self):
        xml = _xml("MSFT", [("2024-01-15", "P", 100.0)])
        rows = insider_data._parse_form4_xml(xml, "AAPL")
        assert rows == []


class TestGetInsiderTransactions:
    def test_full_pipeline_with_mocked_network(self, monkeypatch):
        search_result = {
            "hits": {
                "hits": [
                    {
                        "_id": "0000320193-24-000035:wk-form4.xml",
                        "_source": {"ciks": ["0001051401", "0000320193"], "adsh": "0000320193-24-000035"},
                    }
                ]
            }
        }
        monkeypatch.setattr(insider_data, "_get_json", lambda url, params: search_result)
        monkeypatch.setattr(
            insider_data, "_get_text", lambda url: _xml("AAPL", [("2024-01-15", "P", 200.0)])
        )

        df = insider_data.get_insider_transactions("AAPL", start_date="2024-01-01", end_date="2024-02-01")

        assert len(df) == 1
        assert df.iloc[0]["code"] == "P"
        assert df.iloc[0]["shares"] == 200.0

    def test_pagination_stops_when_page_has_fewer_than_10_hits(self, monkeypatch):
        call_count = []

        def fake_search(url, params):
            call_count.append(params["from"])
            if params["from"] == 0:
                hits = [
                    {"_id": f"acc-{i}:f.xml", "_source": {"ciks": ["1", "2"], "adsh": f"000-24-{i:06d}"}}
                    for i in range(10)
                ]
            else:
                hits = []
            return {"hits": {"hits": hits}}

        monkeypatch.setattr(insider_data, "_get_json", fake_search)
        monkeypatch.setattr(insider_data, "_get_text", lambda url: _xml("AAPL", []))

        insider_data.get_insider_transactions("AAPL", start_date="2024-01-01", end_date="2024-02-01")

        assert call_count == [0, 10]

    def test_uses_cache_on_second_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr(insider_data, "_get_json", lambda url, params: calls.append(1) or {"hits": {"hits": []}})

        insider_data.get_insider_transactions("AAPL", start_date="2024-01-01", end_date="2024-02-01")
        insider_data.get_insider_transactions("AAPL", start_date="2024-01-01", end_date="2024-02-01")

        assert len(calls) == 1

    def test_individual_filing_fetch_failure_does_not_abort_the_whole_symbol(self, monkeypatch):
        search_result = {
            "hits": {
                "hits": [
                    {"_id": "a:f.xml", "_source": {"ciks": ["1", "2"], "adsh": "000-24-000001"}},
                    {"_id": "b:f.xml", "_source": {"ciks": ["1", "2"], "adsh": "000-24-000002"}},
                ]
            }
        }
        monkeypatch.setattr(insider_data, "_get_json", lambda url, params: search_result)

        def fake_get_text(url):
            if "000001" in url:
                raise RuntimeError("network blip")
            return _xml("AAPL", [("2024-01-15", "P", 100.0)])

        monkeypatch.setattr(insider_data, "_get_text", fake_get_text)

        df = insider_data.get_insider_transactions("AAPL", start_date="2024-01-01", end_date="2024-02-01")

        assert len(df) == 1  # the second filing's transaction, first one skipped


class TestLoadTickerToCikMap:
    def test_fetches_and_caches(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "custom_ticker_cik.json"
        monkeypatch.setattr(insider_data, "TICKER_CIK_CACHE_PATH", cache_path)
        payload = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 1403161, "ticker": "V", "title": "VISA INC."},
        }
        monkeypatch.setattr(insider_data, "_get_json", lambda url, params: payload)

        mapping = insider_data._load_ticker_to_cik_map()

        assert mapping == {"AAPL": "0000320193", "V": "0001403161"}
        assert cache_path.exists()

    def test_uses_cache_without_refetching(self, tmp_path, monkeypatch):
        cache_path = tmp_path / "custom_ticker_cik.json"
        cache_path.write_text('{"AAPL": "0000320193"}')
        monkeypatch.setattr(insider_data, "TICKER_CIK_CACHE_PATH", cache_path)
        monkeypatch.setattr(
            insider_data, "_get_json", lambda url, params: (_ for _ in ()).throw(AssertionError("should not fetch"))
        )

        mapping = insider_data._load_ticker_to_cik_map()

        assert mapping == {"AAPL": "0000320193"}


class TestCikBasedSearch:
    def test_search_uses_ciks_param_when_symbol_resolves(self, monkeypatch):
        # A bare ticker-text search ("V", "F", etc.) can match the letter
        # appearing anywhere in filing text, not just as the actual issuer
        # -- CIK-based search avoids that false-positive-match problem
        # entirely (see _load_ticker_to_cik_map's docstring for the real
        # 10,000+-false-hit case this fixes). Confirm the query is built
        # with "ciks", not "q", once a symbol resolves to one.
        monkeypatch.setattr(insider_data, "_load_ticker_to_cik_map", lambda force_refresh=False: {"V": "0001403161"})
        seen_params = []

        def fake_search(url, params):
            seen_params.append(params)
            return {"hits": {"hits": []}}

        monkeypatch.setattr(insider_data, "_get_json", fake_search)

        insider_data._search_form4_filings("V", "2024-01-01", "2024-02-01")

        assert seen_params[0]["ciks"] == "0001403161"
        assert "q" not in seen_params[0]

    def test_search_falls_back_to_text_query_when_symbol_unresolved(self, monkeypatch):
        monkeypatch.setattr(insider_data, "_load_ticker_to_cik_map", lambda force_refresh=False: {})
        seen_params = []

        def fake_search(url, params):
            seen_params.append(params)
            return {"hits": {"hits": []}}

        monkeypatch.setattr(insider_data, "_get_json", fake_search)

        insider_data._search_form4_filings("SOME_ETF", "2024-01-01", "2024-02-01")

        assert seen_params[0]["q"] == '"SOME_ETF"'
        assert "ciks" not in seen_params[0]


class TestGetUniverseInsiderTransactions:
    def test_skips_symbols_that_fail(self, monkeypatch):
        def fake_get(symbol, start_date, end_date):
            if symbol == "BAD":
                raise RuntimeError("boom")
            return pd.DataFrame({"code": ["P"], "shares": [1.0]}, index=pd.to_datetime(["2024-01-01"]))

        monkeypatch.setattr(insider_data, "get_insider_transactions", fake_get)

        result = insider_data.get_universe_insider_transactions(["A", "BAD"])

        assert set(result.keys()) == {"A"}


class TestGetCachedInsiderTransactions:
    def test_never_hits_the_network(self, monkeypatch):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("should never fetch over the network")

        monkeypatch.setattr(insider_data, "_get_json", fail_if_called)
        monkeypatch.setattr(insider_data, "_get_text", fail_if_called)

        result = insider_data.get_cached_insider_transactions(["NOT_CACHED_AT_ALL"])

        assert result == {}

    def test_returns_only_symbols_with_a_cache_entry(self, monkeypatch):
        monkeypatch.setattr(insider_data, "_get_json", lambda url, params: {"hits": {"hits": []}})
        insider_data.get_insider_transactions("WARM", start_date="2024-01-01", end_date="2024-02-01")

        result = insider_data.get_cached_insider_transactions(["WARM", "COLD"])

        assert set(result.keys()) <= {"WARM"}  # WARM cached empty (no filings) so may be excluded too

    def test_returns_cached_data_matching_a_prior_successful_fetch(self, monkeypatch):
        search_result = {
            "hits": {"hits": [{"_id": "a:f.xml", "_source": {"ciks": ["1", "2"], "adsh": "000-24-000001"}}]}
        }
        monkeypatch.setattr(insider_data, "_get_json", lambda url, params: search_result)
        monkeypatch.setattr(
            insider_data,
            "_get_text",
            lambda url: _xml("WARM", [("2024-01-15", "P", 100.0)]),
        )
        insider_data.get_insider_transactions("WARM", start_date="2024-01-01", end_date="2024-02-01")

        result = insider_data.get_cached_insider_transactions(["WARM", "COLD"])

        assert set(result.keys()) == {"WARM"}
        assert len(result["WARM"]) == 1


class TestSignalCompute:
    def test_net_buying_scores_higher_than_net_selling(self):
        transactions = {
            "BUYER": pd.DataFrame(
                {"code": ["P", "P"], "shares": [1000.0, 1000.0]},
                index=pd.to_datetime(["2024-01-10", "2024-01-20"]),
            ),
            "SELLER": pd.DataFrame(
                {"code": ["S", "S"], "shares": [1000.0, 1000.0]},
                index=pd.to_datetime(["2024-01-10", "2024-01-20"]),
            ),
        }
        scores = insider_buying.compute(transactions, as_of=pd.Timestamp("2024-02-01"), lookback_days=90)
        assert scores["BUYER"] > scores["SELLER"]

    def test_transactions_outside_lookback_window_are_ignored(self):
        transactions = {
            "A": pd.DataFrame({"code": ["P"], "shares": [1000.0]}, index=pd.to_datetime(["2023-01-01"]))
        }
        scores = insider_buying.compute(transactions, as_of=pd.Timestamp("2024-02-01"), lookback_days=90)
        assert "A" not in scores.index

    def test_empty_transactions_excluded(self):
        transactions = {"A": pd.DataFrame(columns=["code", "shares"])}
        scores = insider_buying.compute(transactions, as_of=pd.Timestamp("2024-02-01"))
        assert "A" not in scores.index
