"""Tests for alpha_engine.ml.validation's IC (Information Coefficient)
computation -- the guide's own backtesting checklist requires positive
out-of-sample IC (>0.03) before trusting a signal; this validates the
function that actually checks it. All synthetic data, no network calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.ml import validation
from alpha_engine.ml.alpha_model import AlphaModel
from alpha_engine.ml.features import FEATURE_COLUMNS


def _synthetic_prices(n_days: int = 400, n_symbols: int = 8, seed: int = 1) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prices = {}
    for i in range(n_symbols):
        drift = 0.0002 * (1 if i % 2 == 0 else -1)
        rets = rng.normal(drift, 0.012, n_days)
        close = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
        prices[f"SYM{i}"] = pd.DataFrame({"close": close}, index=dates)
    return prices


def _synthetic_benchmark(n_days: int = 400, seed: int = 99) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.009, n_days)
    close = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
    return pd.DataFrame({"close": close}, index=dates)


class TestComputeIc:
    def test_perfectly_correlated_signal_has_ic_near_one(self):
        scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0})
        returns = pd.Series({"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05})
        assert validation.compute_ic(scores, returns) == pytest.approx(1.0)

    def test_perfectly_inverted_signal_has_ic_near_negative_one(self):
        scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0})
        returns = pd.Series({"A": 0.05, "B": 0.04, "C": 0.03, "D": 0.02, "E": 0.01})
        assert validation.compute_ic(scores, returns) == pytest.approx(-1.0)

    def test_too_few_observations_returns_zero(self):
        scores = pd.Series({"A": 1.0, "B": 2.0})
        returns = pd.Series({"A": 0.01, "B": 0.02})
        assert validation.compute_ic(scores, returns) == 0.0

    def test_mismatched_index_only_correlates_overlap(self):
        scores = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0})
        returns = pd.Series({"A": 0.05, "B": 0.04, "C": 0.03, "D": 0.02, "E": 0.01})  # missing F
        ic = validation.compute_ic(scores, returns)
        assert ic == pytest.approx(-1.0)  # A-E still perfectly inverted


class TestComputeMeanIc:
    def test_runs_across_validation_period_and_returns_expected_shape(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        symbols = list(prices.keys())
        all_dates = benchmark.index

        rng = np.random.RandomState(1)
        X = pd.DataFrame(rng.normal(0, 1, size=(200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        y = X["momentum"] * 0.05 + rng.normal(0, 0.01, 200)
        model = AlphaModel().train(X, y)

        result = validation.compute_mean_ic(
            model,
            prices,
            fundamentals=None,
            macro_data=None,
            benchmark_prices=benchmark,
            symbols=symbols,
            validation_start=all_dates[100],
            validation_end=all_dates[300],
            cadence="weekly",
            forward_days=21,
        )

        assert set(result.keys()) == {"mean_ic", "ic_std", "ic_ir", "n_periods", "passes"}
        assert result["n_periods"] > 0
        assert -1.0 <= result["mean_ic"] <= 1.0
        assert isinstance(result["passes"], bool) or isinstance(result["passes"], np.bool_)

    def test_passes_flag_matches_threshold(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        symbols = list(prices.keys())
        all_dates = benchmark.index

        rng = np.random.RandomState(1)
        X = pd.DataFrame(rng.normal(0, 1, size=(200, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        y = X["momentum"] * 0.05 + rng.normal(0, 0.01, 200)
        model = AlphaModel().train(X, y)

        result = validation.compute_mean_ic(
            model,
            prices,
            fundamentals=None,
            macro_data=None,
            benchmark_prices=benchmark,
            symbols=symbols,
            validation_start=all_dates[100],
            validation_end=all_dates[300],
        )

        assert result["passes"] == (result["mean_ic"] >= validation.MIN_IC_THRESHOLD)

    def test_no_dates_in_range_returns_zeroed_result(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        symbols = list(prices.keys())
        all_dates = benchmark.index

        model = AlphaModel().train(
            pd.DataFrame(np.zeros((10, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS),
            pd.Series(np.zeros(10)),
        )

        # validation_end before validation_start -- no rebalance dates fall in range
        result = validation.compute_mean_ic(
            model,
            prices,
            fundamentals=None,
            macro_data=None,
            benchmark_prices=benchmark,
            symbols=symbols,
            validation_start=all_dates[300],
            validation_end=all_dates[100],
        )

        assert result == {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "n_periods": 0, "passes": False}


class TestComputeSignalIc:
    def test_runs_and_returns_expected_shape(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        all_dates = benchmark.index

        def fake_signal(as_of):
            # deterministic per-date scores, unrelated to actual forward
            # returns -- just needs a real, non-trivial cross-section
            rng = np.random.RandomState(hash(as_of) % (2**31))
            return pd.Series(rng.normal(0, 1, len(prices)), index=list(prices.keys()))

        result = validation.compute_signal_ic(
            fake_signal, prices, benchmark_prices=benchmark,
            validation_start=all_dates[100], validation_end=all_dates[300],
            cadence="weekly", forward_days=21,
        )

        assert set(result.keys()) == {"mean_ic", "ic_std", "ic_ir", "n_periods", "passes"}
        assert result["n_periods"] > 0
        assert -1.0 <= result["mean_ic"] <= 1.0
        assert result["passes"] == (result["mean_ic"] >= validation.MIN_IC_THRESHOLD)

    def test_signal_derived_from_actual_forward_returns_scores_strongly_positive(self):
        # a signal that "cheats" by scoring symbols using their own actual
        # forward return should show strong positive IC -- confirms the
        # measurement direction/plumbing is correct, not just that it runs.
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        symbols = list(prices.keys())
        all_dates = benchmark.index

        def oracle_signal(as_of):
            from alpha_engine.ml.training import forward_excess_return
            values = {}
            for s in symbols:
                fwd = forward_excess_return(prices, benchmark, s, as_of, 21)
                if fwd is not None:
                    values[s] = fwd
            return pd.Series(values, dtype=float)

        result = validation.compute_signal_ic(
            oracle_signal, prices, benchmark_prices=benchmark,
            validation_start=all_dates[100], validation_end=all_dates[300],
            cadence="weekly", forward_days=21,
        )

        assert result["mean_ic"] > 0.9  # near-perfect, same value scored against itself

    def test_dates_with_empty_signal_are_skipped_not_counted(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        all_dates = benchmark.index

        result = validation.compute_signal_ic(
            lambda as_of: pd.Series(dtype=float), prices, benchmark_prices=benchmark,
            validation_start=all_dates[100], validation_end=all_dates[300], cadence="weekly",
        )

        assert result == {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "n_periods": 0, "passes": False}
