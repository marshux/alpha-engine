"""Tests for data/earnings_sentiment.py's SEC 8-K fetch/parse pipeline
and signals/earnings_sentiment.py's Loughran-McDonald sentiment signal --
mocked network calls throughout (the underlying pipeline was verified
directly against real SEC EDGAR data and the real LM dictionary during
development; see data/earnings_sentiment.py's module docstring), no real
HTTP requests in the test suite itself.
"""
from __future__ import annotations

import pandas as pd
import pytest

from alpha_engine.data import earnings_sentiment as es_data
from alpha_engine.signals import earnings_sentiment as es_signal

LM_CSV = (
    "Word,Negative,Positive,Uncertainty\n"
    "GREAT,0,2009,0\n"
    "STRONG,0,2009,0\n"
    "WEAK,2009,0,0\n"
    "DECLINE,2009,0,0\n"
    "UNCERTAIN,0,0,2009\n"
    "MAYBE,0,0,2009\n"
)

SUBMISSIONS_RECENT = {
    "accessionNumber": ["0000320193-24-000005", "0001140361-24-006577", "0000320193-24-000071"],
    "filingDate": ["2024-02-01", "2024-02-24", "2024-07-31"],
    "form": ["8-K", "8-K", "8-K"],
    "items": ["2.02,9.01", "5.07,9.01", "2.02,9.01"],
    "primaryDocument": ["aapl-20240201.htm", "ef2006_8k.htm", "aapl-20240731.htm"],
}

INDEX_HTML = """
<table>
<tr><td>1</td><td>aapl-20240201.htm</td><td>8-K</td><td></td></tr>
<tr><td>2</td><td>a8-kex991.htm</td><td>EX-99.1</td><td>140168</td></tr>
</table>
"""


def _submissions_payload(recent=None, files=None):
    return {"filings": {"recent": recent or SUBMISSIONS_RECENT, "files": files or []}}


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(es_data, "CACHE_DIR", tmp_path / "earnings_sentiment")
    monkeypatch.setattr(es_data, "LM_DICT_CACHE_PATH", tmp_path / "lm_dict.parquet")
    monkeypatch.setattr(es_data, "REQUEST_PAUSE_SECONDS", 0.0)
    monkeypatch.setattr(es_data, "MANUAL_CIK_OVERRIDES", {})
    monkeypatch.setattr(es_data, "_load_ticker_to_cik_map", lambda force_refresh=False: {"AAPL": "0000320193"})


class TestLoadLoughranMcDonaldDictionary:
    def test_parses_year_encoded_columns_as_booleans(self, monkeypatch):
        monkeypatch.setattr(es_data, "_get_text", lambda url: LM_CSV)

        lm = es_data._load_loughran_mcdonald_dictionary()

        assert lm["GREAT"] == {"negative": False, "positive": True, "uncertainty": False}
        assert lm["WEAK"] == {"negative": True, "positive": False, "uncertainty": False}
        assert lm["UNCERTAIN"] == {"negative": False, "positive": False, "uncertainty": True}

    def test_uses_cache_on_second_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr(es_data, "_get_text", lambda url: calls.append(1) or LM_CSV)

        es_data._load_loughran_mcdonald_dictionary()
        es_data._load_loughran_mcdonald_dictionary()

        assert len(calls) == 1

    def test_retries_when_google_drive_serves_an_interstitial_instead_of_the_csv(self, monkeypatch):
        # observed directly in development: Google Drive intermittently
        # returns an anti-bot HTML page with a 200 status instead of the
        # real file -- a wrong-content-not-an-HTTP-error failure mode.
        responses = iter(["<html>Google Drive - Virus scan warning</html>", LM_CSV])
        monkeypatch.setattr(es_data, "_get_text", lambda url: next(responses))
        monkeypatch.setattr(es_data.time, "sleep", lambda s: None)

        lm = es_data._load_loughran_mcdonald_dictionary()

        assert "GREAT" in lm

    def test_gives_up_after_max_retries_of_bad_content(self, monkeypatch):
        monkeypatch.setattr(es_data, "_get_text", lambda url: "<html>Google Drive - Virus scan warning</html>")
        monkeypatch.setattr(es_data.time, "sleep", lambda s: None)

        with pytest.raises(ValueError):
            es_data._load_loughran_mcdonald_dictionary()


class TestExtractEarnings8ks:
    def test_filters_to_item_2_02_only(self):
        out = es_data._extract_earnings_8ks(SUBMISSIONS_RECENT, "2024-01-01", "2024-12-31")

        assert [f["accession"] for f in out] == ["0000320193-24-000005", "0000320193-24-000071"]

    def test_respects_date_range(self):
        out = es_data._extract_earnings_8ks(SUBMISSIONS_RECENT, "2024-03-01", "2024-12-31")

        assert [f["accession"] for f in out] == ["0000320193-24-000071"]


class TestFindExhibit991Filename:
    def test_finds_the_exhibit_991_document_from_the_index_table(self, monkeypatch):
        monkeypatch.setattr(es_data, "_get_text", lambda url: INDEX_HTML)

        filename = es_data._find_exhibit_99_1_filename("320193", "0000320193-24-000005")

        assert filename == "a8-kex991.htm"

    def test_no_exhibit_99_returns_none(self, monkeypatch):
        monkeypatch.setattr(es_data, "_get_text", lambda url: "<table><tr><td>8-K</td></tr></table>")

        filename = es_data._find_exhibit_99_1_filename("320193", "0000320193-24-000005")

        assert filename is None


class TestHtmlToText:
    def test_strips_tags_and_decodes_common_entities(self):
        html = "<p>Apple&#8217;s revenue &amp; growth &#8212; strong quarter</p>"
        text = es_data._html_to_text(html)
        assert text == "Apple's revenue & growth - strong quarter"


class TestScoreText:
    def test_counts_word_fractions_correctly(self, monkeypatch):
        monkeypatch.setattr(es_data, "_get_text", lambda url: LM_CSV)
        lm = es_data._load_loughran_mcdonald_dictionary()

        scores = es_data._score_text("great strong quarter weak decline", lm)

        assert scores["positive_score"] == pytest.approx(2 / 5)
        assert scores["negative_score"] == pytest.approx(2 / 5)
        assert scores["uncertainty_score"] == pytest.approx(0.0)

    def test_empty_text_returns_zeros(self):
        scores = es_data._score_text("", {})
        assert scores == {"positive_score": 0.0, "negative_score": 0.0, "uncertainty_score": 0.0}


class TestGetEarningsSentimentHistory:
    def test_full_pipeline_with_mocked_network(self, monkeypatch):
        monkeypatch.setattr(es_data, "_get_text", _multiplex_get_text)
        monkeypatch.setattr(
            es_data, "_get_json", lambda url, params: _submissions_payload()
        )

        df = es_data.get_earnings_sentiment_history("AAPL", start_date="2024-01-01", end_date="2024-12-31")

        assert list(df.index) == [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-07-31")]
        assert "positive_score" in df.columns

    def test_unresolved_cik_returns_empty_without_fetching_filings(self, monkeypatch):
        monkeypatch.setattr(es_data, "_load_ticker_to_cik_map", lambda force_refresh=False: {})

        def fail_if_called(url, params):
            raise AssertionError("should not fetch filings for an unresolved symbol")

        monkeypatch.setattr(es_data, "_get_json", fail_if_called)

        df = es_data.get_earnings_sentiment_history("NOPE", start_date="2024-01-01", end_date="2024-12-31")

        assert df.empty

    def test_uses_cache_on_second_call(self, monkeypatch):
        calls = []
        monkeypatch.setattr(es_data, "_get_text", lambda url: calls.append(1) or _multiplex_get_text(url))
        monkeypatch.setattr(es_data, "_get_json", lambda url, params: _submissions_payload())

        es_data.get_earnings_sentiment_history("AAPL", start_date="2024-01-01", end_date="2024-12-31")
        n_calls_first = len(calls)
        es_data.get_earnings_sentiment_history("AAPL", start_date="2024-01-01", end_date="2024-12-31")

        assert len(calls) == n_calls_first  # second call hit the cache, no new fetches

    def test_individual_filing_failure_does_not_abort_the_whole_symbol(self, monkeypatch):
        def flaky_get_text(url):
            if "0000320193-24-000005" in url:
                raise RuntimeError("network blip")
            return _multiplex_get_text(url)

        monkeypatch.setattr(es_data, "_get_text", flaky_get_text)
        monkeypatch.setattr(es_data, "_get_json", lambda url, params: _submissions_payload())

        df = es_data.get_earnings_sentiment_history("AAPL", start_date="2024-01-01", end_date="2024-12-31")

        assert list(df.index) == [pd.Timestamp("2024-07-31")]


class TestGetCachedEarningsSentiment:
    def test_never_hits_the_network(self, monkeypatch):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("should never fetch over the network")

        monkeypatch.setattr(es_data, "_get_json", fail_if_called)
        monkeypatch.setattr(es_data, "_get_text", fail_if_called)

        result = es_data.get_cached_earnings_sentiment(["NOT_CACHED"])

        assert result == {}


def _multiplex_get_text(url):
    if url == es_data.LM_DICT_URL:
        return LM_CSV
    if "-index.html" in url:
        return INDEX_HTML
    if "a8-kex991.htm" in url:
        return "<p>Great strong results this quarter.</p>"
    raise AssertionError(f"unexpected URL in test: {url}")


class TestLatestEarningsSentimentAsOf:
    def test_never_uses_a_future_report(self):
        df = pd.DataFrame(
            {"positive_score": [0.01, 0.02]}, index=pd.to_datetime(["2024-02-01", "2024-07-31"])
        )
        report = es_data.latest_earnings_sentiment_as_of(df, pd.Timestamp("2024-03-01"))
        assert report.name == pd.Timestamp("2024-02-01")

    def test_empty_returns_none(self):
        assert es_data.latest_earnings_sentiment_as_of(pd.DataFrame(), pd.Timestamp("2024-01-01")) is None

    def test_before_any_report_returns_none(self):
        df = pd.DataFrame({"positive_score": [0.01]}, index=pd.to_datetime(["2024-07-31"]))
        assert es_data.latest_earnings_sentiment_as_of(df, pd.Timestamp("2024-01-01")) is None


class TestSignalCompute:
    def _report(self, pos, neg, unc):
        return pd.Series({"positive_score": pos, "negative_score": neg, "uncertainty_score": unc})

    def test_improving_tone_scores_higher_than_worsening(self):
        earnings_sentiment = {
            "IMPROVING": pd.DataFrame(
                [
                    {"positive_score": 0.01, "negative_score": 0.02, "uncertainty_score": 0.01},
                    {"positive_score": 0.03, "negative_score": 0.01, "uncertainty_score": 0.01},
                ],
                index=pd.to_datetime(["2023-11-01", "2024-02-01"]),
            ),
            "WORSENING": pd.DataFrame(
                [
                    {"positive_score": 0.03, "negative_score": 0.01, "uncertainty_score": 0.01},
                    {"positive_score": 0.01, "negative_score": 0.03, "uncertainty_score": 0.01},
                ],
                index=pd.to_datetime(["2023-11-01", "2024-02-01"]),
            ),
        }
        scores = es_signal.compute(earnings_sentiment, as_of=pd.Timestamp("2024-02-15"))
        assert scores["IMPROVING"] > scores["WORSENING"]

    def test_no_recent_filing_scores_neutral_zero_before_zscore(self):
        earnings_sentiment = {
            "STALE": pd.DataFrame(
                {"positive_score": [0.02], "negative_score": [0.01], "uncertainty_score": [0.01]},
                index=pd.to_datetime(["2023-01-01"]),  # far more than 90 days before as_of
            ),
            "RECENT_POSITIVE": pd.DataFrame(
                {"positive_score": [0.05], "negative_score": [0.0], "uncertainty_score": [0.0]},
                index=pd.to_datetime(["2024-01-15"]),
            ),
        }
        scores = es_signal.compute(earnings_sentiment, as_of=pd.Timestamp("2024-02-01"))
        assert scores["RECENT_POSITIVE"] > scores["STALE"]

    def test_no_filing_history_at_all_is_excluded(self):
        earnings_sentiment = {"A": pd.DataFrame(columns=["positive_score", "negative_score", "uncertainty_score"])}
        scores = es_signal.compute(earnings_sentiment, as_of=pd.Timestamp("2024-01-01"))
        assert "A" not in scores.index

    def test_no_prior_quarter_uses_zero_delta(self):
        earnings_sentiment = {
            "FIRST_EVER": pd.DataFrame(
                {"positive_score": [0.02], "negative_score": [0.01], "uncertainty_score": [0.01]},
                index=pd.to_datetime(["2024-01-15"]),
            ),
        }
        # should not raise despite no prior report to compute delta_sentiment against
        scores = es_signal.compute(earnings_sentiment, as_of=pd.Timestamp("2024-02-01"))
        assert "FIRST_EVER" in scores.index
