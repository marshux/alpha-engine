"""Tests for the alpha_engine.ml package: feature construction, the
AlphaModel LightGBM wrapper (train/predict/save/load), and walk-forward-
safe training-set construction. All synthetic data, no network calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.ml.alpha_model import AlphaModel
from alpha_engine.ml.features import FEATURE_COLUMNS, build_ml_features
from alpha_engine.ml.training import build_training_set


def _synthetic_prices(n_days: int = 400, n_symbols: int = 6, seed: int = 1) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prices = {}
    for i in range(n_symbols):
        symbol = f"SYM{i}"
        drift = 0.0002 * (1 if i % 2 == 0 else -1)
        rets = rng.normal(drift, 0.012, n_days)
        close = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
        prices[symbol] = pd.DataFrame({"close": close}, index=dates)
    return prices


def _synthetic_benchmark(n_days: int = 400, seed: int = 99) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.009, n_days)
    close = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
    return pd.DataFrame({"close": close}, index=dates)


def _synthetic_macro(n_days: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {"yield_curve": np.full(n_days, 1.2), "vix": np.full(n_days, 16.0), "credit_spread": np.full(n_days, 3.2)},
        index=dates,
    )


class TestBuildMlFeatures:
    def test_returns_expected_columns_for_every_symbol(self):
        prices = _synthetic_prices()
        macro = _synthetic_macro()
        as_of = list(prices.values())[0].index[-1]

        features = build_ml_features(prices, fundamentals=None, macro_data=macro, as_of=as_of, symbols=list(prices.keys()))

        assert list(features.columns) == FEATURE_COLUMNS
        assert set(features.index) == set(prices.keys())
        assert not features.isna().any().any()

    def test_macro_columns_broadcast_the_latest_reading(self):
        prices = _synthetic_prices()
        macro = _synthetic_macro()
        as_of = list(prices.values())[0].index[-1]

        features = build_ml_features(prices, None, macro, as_of, list(prices.keys()))

        assert (features["vix_level"] == 16.0).all()
        assert (features["yield_curve"] == 1.2).all()

    def test_missing_macro_data_defaults_to_zero_not_a_crash(self):
        prices = _synthetic_prices()
        as_of = list(prices.values())[0].index[-1]

        features = build_ml_features(prices, None, None, as_of, list(prices.keys()))

        assert (features["vix_level"] == 0.0).all()


class TestAlphaModel:
    def _training_data(self, n_rows=200, seed=1):
        rng = np.random.RandomState(seed)
        X = pd.DataFrame(rng.normal(0, 1, size=(n_rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
        # y correlated with the first feature so the model has real signal to learn
        y = X["momentum"] * 0.05 + rng.normal(0, 0.01, n_rows)
        return X, y

    def test_train_predict_round_trip(self):
        X, y = self._training_data()
        model = AlphaModel().train(X, y)

        preds = model.predict(X)

        assert list(preds.index) == list(X.index)
        assert preds.std() > 0  # not a degenerate constant prediction
        assert abs(preds.mean()) < 1e-6  # z-scored

    def test_predict_before_train_raises(self):
        with pytest.raises(ValueError):
            AlphaModel().predict(pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS}))

    def test_predict_missing_feature_column_raises(self):
        X, y = self._training_data()
        model = AlphaModel().train(X, y)
        broken = X.drop(columns=["momentum"])

        with pytest.raises(ValueError, match="missing feature columns"):
            model.predict(broken)

    def test_save_load_round_trip_produces_identical_predictions(self, tmp_path):
        X, y = self._training_data()
        model = AlphaModel().train(X, y, trained_through="2024-01-01")
        before = model.predict(X)

        model_path = tmp_path / "model.txt"
        model.save(model_path)
        loaded = AlphaModel.load(model_path)
        after = loaded.predict(X)

        pd.testing.assert_series_equal(before, after)
        assert loaded.trained_through == "2024-01-01"
        assert loaded.feature_columns == model.feature_columns

    def test_save_untrained_model_raises(self, tmp_path):
        with pytest.raises(ValueError):
            AlphaModel().save(tmp_path / "model.txt")


class TestBuildTrainingSet:
    def test_produces_aligned_features_and_labels(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        macro = _synthetic_macro()
        all_dates = benchmark.index

        X, y = build_training_set(
            prices,
            fundamentals=None,
            macro_data=macro,
            benchmark_prices=benchmark,
            symbols=list(prices.keys()),
            train_start=all_dates[100],
            train_end=all_dates[300],
            cadence="weekly",
            forward_days=21,
        )

        assert len(X) == len(y)
        assert len(X) > 0
        assert list(X.columns) == FEATURE_COLUMNS

    def test_drops_rows_too_close_to_train_end_for_a_forward_label(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        all_dates = benchmark.index

        # A training window narrower than forward_days, right at the end
        # of the available price history -- no rebalance date in range
        # has 21 trading days of forward data left to label with.
        X, y = build_training_set(
            prices,
            fundamentals=None,
            macro_data=None,
            benchmark_prices=benchmark,
            symbols=list(prices.keys()),
            train_start=all_dates[-5],
            train_end=all_dates[-1],
            cadence="weekly",
            forward_days=21,
        )

        assert len(X) == 0
        assert len(y) == 0
