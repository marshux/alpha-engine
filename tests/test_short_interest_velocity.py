"""Tests for signals/short_interest_velocity.py -- pure computation on
already-shaped short-interest DataFrames, no network mocking needed
(data/short_interest.py's fetch/cache layer is already covered by
tests/test_short_interest.py).
"""
from __future__ import annotations

import pandas as pd
import pytest

from alpha_engine.signals import short_interest_velocity as siv


class TestSignalCompute:
    def test_rapidly_rising_days_to_cover_scores_lower_than_falling(self):
        short_interest = {
            "RISING": pd.DataFrame(
                {"days_to_cover": [2.0, 4.0]}, index=pd.to_datetime(["2023-12-01", "2024-01-25"])
            ),
            "FALLING": pd.DataFrame(
                {"days_to_cover": [4.0, 2.0]}, index=pd.to_datetime(["2023-12-01", "2024-01-25"])
            ),
        }
        scores = siv.compute(short_interest, as_of=pd.Timestamp("2024-01-26"))
        assert scores["FALLING"] > scores["RISING"]

    def test_no_change_scores_neutral_relative_to_movers(self):
        short_interest = {
            "STEADY": pd.DataFrame(
                {"days_to_cover": [3.0, 3.0]}, index=pd.to_datetime(["2023-12-01", "2024-01-25"])
            ),
            "RISING": pd.DataFrame(
                {"days_to_cover": [3.0, 6.0]}, index=pd.to_datetime(["2023-12-01", "2024-01-25"])
            ),
        }
        scores = siv.compute(short_interest, as_of=pd.Timestamp("2024-01-26"))
        assert scores["STEADY"] > scores["RISING"]

    def test_missing_prior_report_excludes_symbol(self):
        # only one report ever, no ~4-week-prior reading to compare against
        short_interest = {
            "A": pd.DataFrame({"days_to_cover": [2.0]}, index=pd.to_datetime(["2024-01-20"]))
        }
        scores = siv.compute(short_interest, as_of=pd.Timestamp("2024-01-26"))
        assert "A" not in scores.index

    def test_no_coverage_excludes_symbol(self):
        short_interest = {"A": pd.DataFrame(columns=["days_to_cover"])}
        scores = siv.compute(short_interest, as_of=pd.Timestamp("2024-01-01"))
        assert "A" not in scores.index

    def test_nan_days_to_cover_excludes_symbol(self):
        short_interest = {
            "A": pd.DataFrame(
                {"days_to_cover": [float("nan"), 3.0]}, index=pd.to_datetime(["2023-12-01", "2024-01-25"])
            )
        }
        scores = siv.compute(short_interest, as_of=pd.Timestamp("2024-01-26"))
        assert "A" not in scores.index

    def test_zero_prior_days_to_cover_excludes_symbol(self):
        # avoid divide-by-zero on the % change
        short_interest = {
            "A": pd.DataFrame(
                {"days_to_cover": [0.0, 2.0]}, index=pd.to_datetime(["2023-12-01", "2024-01-25"])
            )
        }
        scores = siv.compute(short_interest, as_of=pd.Timestamp("2024-01-26"))
        assert "A" not in scores.index

    def test_report_older_than_window_still_used_as_prior(self):
        # latest_short_interest_as_of finds the most recent report <=
        # (as_of - 28 days), even if that report is itself much older --
        # same "most recent available, not exact date match" pattern the
        # level signal already relies on.
        short_interest = {
            "A": pd.DataFrame(
                {"days_to_cover": [2.0, 4.0]}, index=pd.to_datetime(["2023-06-01", "2024-01-25"])
            )
        }
        scores = siv.compute(short_interest, as_of=pd.Timestamp("2024-01-26"))
        assert "A" in scores.index
