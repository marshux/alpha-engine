"""Tests for data/short_interest.py's FINRA fetch/cache and
signals/short_interest.py's days-to-cover signal -- mocked network call,
no real HTTP requests.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from alpha_engine.data import short_interest as si_data
from alpha_engine.signals import short_interest as si_signal


def _fake_csv(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(si_data, "CACHE_DIR", tmp_path / "short_interest")


class TestGetShortInterest:
    def test_parses_rows_into_indexed_dataframe(self, monkeypatch):
        csv_text = _fake_csv(
            [
                {"symbolCode": "AAPL", "currentShortPositionQuantity": "1000", "daysToCoverQuantity": "2.5", "settlementDate": "2024-01-12"},
                {"symbolCode": "AAPL", "currentShortPositionQuantity": "1200", "daysToCoverQuantity": "3.1", "settlementDate": "2024-01-31"},
            ]
        )
        monkeypatch.setattr(si_data, "_post_csv", lambda url, body: csv_text)

        df = si_data.get_short_interest("AAPL")

        assert list(df.columns) == ["short_position", "days_to_cover"]
        assert len(df) == 2
        assert df.index.is_monotonic_increasing
        assert df.loc["2024-01-31", "days_to_cover"] == pytest.approx(3.1)

    def test_sentinel_999_days_to_cover_becomes_nan(self, monkeypatch):
        csv_text = _fake_csv(
            [
                {"symbolCode": "X", "currentShortPositionQuantity": "500", "daysToCoverQuantity": "999.99", "settlementDate": "2024-01-12"},
            ]
        )
        monkeypatch.setattr(si_data, "_post_csv", lambda url, body: csv_text)

        df = si_data.get_short_interest("X")

        assert pd.isna(df["days_to_cover"].iloc[0])

    def test_empty_response_returns_empty_dataframe(self, monkeypatch):
        monkeypatch.setattr(si_data, "_post_csv", lambda url, body: "")

        df = si_data.get_short_interest("NOPE")

        assert df.empty

    def test_uses_cache_on_second_call(self, monkeypatch):
        csv_text = _fake_csv(
            [{"symbolCode": "A", "currentShortPositionQuantity": "1", "daysToCoverQuantity": "1.0", "settlementDate": "2024-01-12"}]
        )
        calls = []
        monkeypatch.setattr(si_data, "_post_csv", lambda url, body: calls.append(1) or csv_text)

        si_data.get_short_interest("A")
        si_data.get_short_interest("A")

        assert len(calls) == 1

    def test_get_universe_short_interest_skips_failures(self, monkeypatch):
        def fake_post(url, body):
            symbol = body["compareFilters"][0]["fieldValue"]
            if symbol == "BAD":
                raise RuntimeError("network error")
            return _fake_csv(
                [{"symbolCode": symbol, "currentShortPositionQuantity": "1", "daysToCoverQuantity": "1.0", "settlementDate": "2024-01-12"}]
            )

        monkeypatch.setattr(si_data, "_post_csv", fake_post)

        result = si_data.get_universe_short_interest(["A", "BAD"])

        assert set(result.keys()) == {"A"}


class TestLatestShortInterestAsOf:
    def test_never_uses_a_future_report(self):
        df = pd.DataFrame(
            {"short_position": [100, 200], "days_to_cover": [1.0, 2.0]},
            index=pd.to_datetime(["2024-01-12", "2024-01-31"]),
        )
        report = si_data.latest_short_interest_as_of(df, as_of=pd.Timestamp("2024-01-20"))
        assert report["days_to_cover"] == 1.0

    def test_empty_returns_none(self):
        assert si_data.latest_short_interest_as_of(pd.DataFrame(), as_of=pd.Timestamp("2024-01-01")) is None

    def test_before_any_report_returns_none(self):
        df = pd.DataFrame({"days_to_cover": [1.0]}, index=pd.to_datetime(["2024-01-12"]))
        assert si_data.latest_short_interest_as_of(df, as_of=pd.Timestamp("2020-01-01")) is None


class TestSignalCompute:
    def test_higher_days_to_cover_scores_lower(self):
        short_interest = {
            "CROWDED": pd.DataFrame({"days_to_cover": [8.0]}, index=pd.to_datetime(["2024-01-12"])),
            "CLEAN": pd.DataFrame({"days_to_cover": [0.5]}, index=pd.to_datetime(["2024-01-12"])),
        }
        scores = si_signal.compute(short_interest, as_of=pd.Timestamp("2024-02-01"))
        assert scores["CLEAN"] > scores["CROWDED"]

    def test_symbol_with_no_coverage_is_excluded(self):
        short_interest = {"A": pd.DataFrame(columns=["days_to_cover"])}
        scores = si_signal.compute(short_interest, as_of=pd.Timestamp("2024-01-01"))
        assert "A" not in scores.index

    def test_nan_days_to_cover_is_excluded(self):
        short_interest = {
            "A": pd.DataFrame({"days_to_cover": [float("nan")]}, index=pd.to_datetime(["2024-01-12"]))
        }
        scores = si_signal.compute(short_interest, as_of=pd.Timestamp("2024-02-01"))
        assert "A" not in scores.index
