"""Tests for data/fundamentals.py's point-in-time selection and
signals/value.py + signals/quality.py's ratio math, against synthetic
multi-quarter fundamentals data -- no network calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.data.fundamentals import latest_report_as_of
from alpha_engine.signals import quality, value


def _fundamentals_df(reports: dict[str, dict]) -> pd.DataFrame:
    """`reports`: {"YYYY-MM-DD": {"Stockholders Equity": ..., ...}, ...}"""
    return pd.DataFrame(reports).T.set_index(pd.to_datetime(list(reports.keys())))


def _price_df(closes: list[float], dates: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000}, index=idx
    )


def test_latest_report_as_of_never_uses_a_future_report():
    df = _fundamentals_df(
        {
            "2023-03-31": {"Stockholders Equity": 100.0},
            "2023-06-30": {"Stockholders Equity": 110.0},
            "2023-09-30": {"Stockholders Equity": 120.0},
        }
    )

    report = latest_report_as_of(df, as_of=pd.Timestamp("2023-07-15"))
    assert report["Stockholders Equity"] == 110.0  # the Q2 report, not Q3

    report = latest_report_as_of(df, as_of=pd.Timestamp("2023-04-01"))
    assert report["Stockholders Equity"] == 100.0

    report = latest_report_as_of(df, as_of=pd.Timestamp("2023-01-01"))
    assert report is None  # before any report exists


def test_latest_report_as_of_empty_fundamentals():
    assert latest_report_as_of(pd.DataFrame(), as_of=pd.Timestamp("2023-01-01")) is None


def test_quality_favors_higher_roe_and_lower_leverage():
    fundamentals = {
        "GOOD": _fundamentals_df(
            {"2023-03-31": {"Stockholders Equity": 100.0, "Net Income": 20.0, "Total Debt": 10.0}}
        ),
        "BAD": _fundamentals_df(
            {"2023-03-31": {"Stockholders Equity": 100.0, "Net Income": 2.0, "Total Debt": 200.0}}
        ),
    }

    scores = quality.compute(fundamentals, as_of=pd.Timestamp("2023-06-01"))

    assert scores["GOOD"] > scores["BAD"]


def test_quality_skips_reports_with_missing_or_nonpositive_equity():
    fundamentals = {
        "OK": _fundamentals_df(
            {"2023-03-31": {"Stockholders Equity": 100.0, "Net Income": 10.0, "Total Debt": 20.0}}
        ),
        "NEGATIVE_EQUITY": _fundamentals_df(
            {"2023-03-31": {"Stockholders Equity": -50.0, "Net Income": 10.0, "Total Debt": 20.0}}
        ),
    }

    scores = quality.compute(fundamentals, as_of=pd.Timestamp("2023-06-01"))

    assert "OK" in scores.index
    assert "NEGATIVE_EQUITY" not in scores.index


def test_value_favors_cheaper_price_to_book():
    dates = ["2023-05-01", "2023-05-02", "2023-05-03"]
    prices = {
        "CHEAP": _price_df([10.0, 10.0, 10.0], dates),  # book value/share = 10 -> P/B = 1.0
        "EXPENSIVE": _price_df([50.0, 50.0, 50.0], dates),  # book value/share = 10 -> P/B = 5.0
    }
    fundamentals = {
        "CHEAP": _fundamentals_df(
            {"2023-03-31": {"Stockholders Equity": 1000.0, "Ordinary Shares Number": 100.0}}
        ),
        "EXPENSIVE": _fundamentals_df(
            {"2023-03-31": {"Stockholders Equity": 1000.0, "Ordinary Shares Number": 100.0}}
        ),
    }

    scores = value.compute(prices, fundamentals, as_of=pd.Timestamp("2023-05-03"))

    assert scores["CHEAP"] > scores["EXPENSIVE"]


def test_value_skips_symbols_missing_price_or_fundamentals():
    dates = ["2023-05-01"]
    prices = {"HAS_PRICE": _price_df([10.0], dates)}
    fundamentals = {
        "HAS_PRICE": _fundamentals_df(
            {"2023-03-31": {"Stockholders Equity": 1000.0, "Ordinary Shares Number": 100.0}}
        ),
        "NO_PRICE": _fundamentals_df(
            {"2023-03-31": {"Stockholders Equity": 1000.0, "Ordinary Shares Number": 100.0}}
        ),
    }

    scores = value.compute(prices, fundamentals, as_of=pd.Timestamp("2023-05-01"))

    assert "HAS_PRICE" in scores.index
    assert "NO_PRICE" not in scores.index
