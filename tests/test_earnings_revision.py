"""Tests for signals/earnings_revision.py's PEAD (post-earnings drift)
proxy -- synthetic prices/fundamentals, no network calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.signals import earnings_revision


def _fundamentals_df(reports: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(reports).T.set_index(pd.to_datetime(list(reports.keys())))


def _price_df(closes: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000}, index=dates
    )


def _flat_benchmark(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return _price_df(np.full(len(dates), 100.0), dates)


class TestCumulativeAbnormalReturn:
    def test_positive_stock_move_vs_flat_benchmark_is_positive(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        closes = np.array([100.0] * 6 + [100.0, 105.0, 110.0, 115.0])
        price_df = _price_df(closes, dates)
        benchmark = _flat_benchmark(dates)
        event_date = dates[9]

        car = earnings_revision._cumulative_abnormal_return(price_df, benchmark, event_date, window_days=3)

        assert car == pytest.approx(115.0 / 100.0 - 1.0)

    def test_insufficient_history_returns_none(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        price_df = _price_df(np.array([100.0, 101.0]), dates)
        benchmark = _flat_benchmark(dates)

        car = earnings_revision._cumulative_abnormal_return(price_df, benchmark, dates[-1], window_days=3)

        assert car is None

    def test_subtracts_benchmark_move_not_just_raw_return(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        stock_closes = np.array([100.0] * 6 + [100.0, 103.0, 106.0, 110.0])  # +10%
        bench_closes = np.array([100.0] * 6 + [100.0, 102.0, 104.0, 106.0])  # +6%
        price_df = _price_df(stock_closes, dates)
        benchmark = _price_df(bench_closes, dates)

        car = earnings_revision._cumulative_abnormal_return(price_df, benchmark, dates[9], window_days=3)

        stock_ret = 110.0 / 100.0 - 1.0
        bench_ret = 106.0 / 100.0 - 1.0
        assert car == pytest.approx(stock_ret - bench_ret)


class TestCompute:
    def test_favors_symbol_with_stronger_initial_reaction(self):
        # The signal measures the *initial* reaction in the window ending
        # at the report date -- per PEAD, that initial reaction is what
        # predicts continued future drift, not a drift that already
        # happened. So the divergence here has to land in the few days
        # up to and including report_date, not after it.
        dates = pd.bdate_range("2023-01-01", periods=90)
        report_date = dates[60]

        strong_closes = np.full(90, 100.0)
        strong_closes[57:61] = [100.0, 105.0, 110.0, 115.0]  # sharp positive reaction into the report
        weak_closes = np.full(90, 100.0)  # no reaction at all

        prices = {"STRONG": _price_df(strong_closes, dates), "WEAK": _price_df(weak_closes, dates)}
        benchmark = _flat_benchmark(dates)
        fundamentals = {
            "STRONG": _fundamentals_df({str(report_date.date()): {"placeholder": 1.0}}),
            "WEAK": _fundamentals_df({str(report_date.date()): {"placeholder": 1.0}}),
        }
        as_of = dates[-1]

        scores = earnings_revision.compute(prices, fundamentals, benchmark, as_of)

        assert scores["STRONG"] > scores["WEAK"]

    def test_symbol_with_no_fundamentals_report_yet_is_excluded(self):
        dates = pd.bdate_range("2023-01-01", periods=30)
        prices = {"A": _price_df(np.full(30, 100.0), dates)}
        benchmark = _flat_benchmark(dates)
        # Report dated AFTER as_of -- latest_report_as_of should find nothing.
        fundamentals = {"A": _fundamentals_df({"2030-01-01": {"placeholder": 1.0}})}

        scores = earnings_revision.compute(prices, fundamentals, benchmark, as_of=dates[-1])

        assert "A" not in scores.index

    def test_symbol_missing_from_prices_is_excluded(self):
        dates = pd.bdate_range("2023-01-01", periods=90)
        report_date = dates[10]
        benchmark = _flat_benchmark(dates)
        fundamentals = {"GHOST": _fundamentals_df({str(report_date.date()): {"placeholder": 1.0}})}

        scores = earnings_revision.compute(prices={}, fundamentals=fundamentals, benchmark_prices=benchmark, as_of=dates[-1])

        assert scores.empty
