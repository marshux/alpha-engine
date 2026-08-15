"""Tests for decision/regime.py's HMM-based regime classifier. Uses small
synthetic feature panels with a fixed RandomState/random_state so results
are fully reproducible -- no real market or macro data involved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.decision import regime


def _synthetic_prices(n_days: int = 400, seed: int = 1) -> dict[str, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    prices = {}
    for i, symbol in enumerate(["AAA", "BBB", "CCC"]):
        drift = 0.0003 * (1 if i % 2 == 0 else -1)
        rets = rng.normal(drift, 0.01, n_days)
        close = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
        prices[symbol] = pd.DataFrame({"close": close}, index=dates)
    return prices


def _synthetic_benchmark(n_days: int = 400, seed: int = 2) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0.0004, 0.008, n_days)
    close = 100 * (1 + pd.Series(rets, index=dates)).cumprod()
    return pd.DataFrame({"close": close}, index=dates)


def _synthetic_macro(n_days: int = 400) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "yield_curve": np.full(n_days, 1.0),
            "vix": np.full(n_days, 15.0),
            "credit_spread": np.full(n_days, 3.0),
        },
        index=dates,
    )


def _block_features(n_rows: int, seed: int, **overrides: float) -> pd.DataFrame:
    """A feature panel where every row shares the given values plus small
    noise (zero-noise columns break hmmlearn's covariance estimation).
    """
    rng = np.random.RandomState(seed)
    base = {
        "spy_return_1m": 0.0,
        "spy_return_3m": 0.0,
        "realized_vol_1m": 0.12,
        "vix_level": 15.0,
        "vix_change_1m": 0.0,
        "yield_curve": 1.0,
        "credit_spread": 3.0,
        "breadth_200ma": 0.5,
    }
    base.update(overrides)
    data = {col: rng.normal(val, abs(val) * 0.02 + 1e-4, n_rows) for col, val in base.items()}
    return pd.DataFrame(data)


class TestBuildRegimeFeatures:
    def test_returns_expected_columns_with_no_nans(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        macro = _synthetic_macro()
        as_of = benchmark.index[-1]

        features = regime.build_regime_features(prices, benchmark, macro, as_of)

        expected_cols = {
            "spy_return_1m",
            "spy_return_3m",
            "realized_vol_1m",
            "vix_level",
            "vix_change_1m",
            "yield_curve",
            "credit_spread",
            "breadth_200ma",
        }
        assert set(features.columns) == expected_cols
        assert not features.isna().any().any()
        assert features["breadth_200ma"].between(0, 1).all()

    def test_one_macro_column_with_a_multi_year_gap_does_not_drop_the_whole_panel(self):
        """Regression test for a real bug: BAMLH0A0HYM2 (credit_spread)
        only has ~3 years of history available via FRED's free CSV
        access, NaN for everything older. A blanket `.dropna()` across
        all 8 columns meant this single column's long-lasting gap
        silently zeroed out the entire feature panel for every older
        date -- discovered because it made decision/regime.py's HMM
        never actually fit at all during a historical walk-forward test,
        always falling through to the "insufficient history" fallback.
        credit_spread should degrade to 0.0 for the gap instead.
        """
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        macro = _synthetic_macro()
        # Simulate a real multi-year gap: credit_spread only available
        # for the most recent 30 days of an otherwise-full macro panel.
        macro = macro.copy()
        macro.loc[macro.index[:-30], "credit_spread"] = float("nan")
        as_of = benchmark.index[-1]

        features = regime.build_regime_features(prices, benchmark, macro, as_of)

        assert len(features) > 30  # not nuked down to just the last 30 rows
        assert not features.isna().any().any()
        assert (features["credit_spread"].iloc[:-30] == 0.0).all()

    def test_respects_lookback_days(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        macro = _synthetic_macro()
        as_of = benchmark.index[-1]

        features = regime.build_regime_features(prices, benchmark, macro, as_of, lookback_days=10)

        assert len(features) <= 10

    def test_never_leaks_data_after_as_of(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        macro = _synthetic_macro()
        as_of = benchmark.index[200]

        features = regime.build_regime_features(prices, benchmark, macro, as_of)

        assert features.index.max() <= as_of


class TestRegimeClassifier:
    def test_fit_requires_minimum_history(self):
        tiny = _block_features(5, seed=1)
        clf = regime.RegimeClassifier()
        with pytest.raises(ValueError):
            clf.fit(tiny)

    def test_predict_current_labels_the_last_regime(self):
        bull = _block_features(80, seed=1, spy_return_1m=0.03, realized_vol_1m=0.10, vix_level=12.0)
        crisis = _block_features(80, seed=2, spy_return_1m=-0.10, realized_vol_1m=0.45, vix_level=40.0)
        panel = pd.concat([bull, crisis], ignore_index=True)

        # n_components=2, not the default 4: with only two genuinely
        # distinct clusters in this synthetic panel, a 4-component fit
        # would leave 2 components with no data ever assigned to them --
        # exactly the degenerate-fit condition _is_degenerate correctly
        # rejects (see test_regime_classifier_rejects_degenerate_fit).
        clf = regime.RegimeClassifier(n_components=2)
        clf.fit(panel)
        label = clf.predict_current(panel)

        assert label in regime.REGIME_LABELS
        # The last block is unambiguously a crisis regime by construction.
        assert label == "CRISIS"

    def test_gross_exposure_and_weight_lookups_match_tables(self):
        clf = regime.RegimeClassifier()
        assert clf.gross_exposure_for("CRISIS") == regime.GROSS_EXPOSURE_BY_REGIME["CRISIS"]
        assert clf.gross_exposure_for("UNKNOWN_LABEL") == regime.DEFAULT_GROSS_EXPOSURE
        assert clf.weight_overrides_for("BULL_TREND") == regime.REGIME_WEIGHTS["BULL_TREND"]
        assert clf.weight_overrides_for("UNKNOWN_LABEL") is None

    def test_gross_exposure_uses_custom_exposure_table_when_given(self):
        clf = regime.RegimeClassifier(exposure_table={"CRISIS": 0.10})
        assert clf.gross_exposure_for("CRISIS") == 0.10
        # Falls through to the module default for a label the custom
        # table doesn't mention.
        assert clf.gross_exposure_for("BULL_TREND") == regime.DEFAULT_GROSS_EXPOSURE

    def test_fit_raises_when_is_degenerate_flags_the_model(self, monkeypatch):
        # This is the exact real-world failure mode found via walk-forward
        # testing on 2011/2022 data (see systematic_core_postmortem.md) --
        # a GaussianHMM component the Viterbi path never visits leaves a
        # collapsed transmat_/covars_. Reproducing that condition from real
        # hmmlearn fit output isn't reliable (hmmlearn's own regularization
        # keeps unused components at a wide, non-degenerate covariance more
        # often than not), so this forces _is_degenerate's return value
        # directly -- see TestIsDegenerate below for the function's own
        # unit tests against constructed transmat_/means_/covars_ values.
        panel = _block_features(60, seed=1)
        monkeypatch.setattr("alpha_engine.decision.regime._is_degenerate", lambda model: True)

        clf = regime.RegimeClassifier()
        with pytest.raises(ValueError, match="degenerate"):
            clf.fit(panel)


class _FakeHmmModel:
    def __init__(self, transmat_, means_, covars_):
        self.transmat_ = transmat_
        self.means_ = means_
        self.covars_ = covars_


class TestIsDegenerate:
    def _healthy_model(self, n_components=3, n_features=4):
        transmat = np.full((n_components, n_components), 1.0 / n_components)
        means = np.zeros((n_components, n_features))
        # Full (n_components, n_features, n_features) shape, matching
        # hmmlearn's covars_ property (always a full matrix regardless of
        # covariance_type -- see _is_degenerate's docstring).
        covars = np.stack([np.eye(n_features) * 0.1 for _ in range(n_components)])
        return _FakeHmmModel(transmat, means, covars)

    def test_healthy_model_is_not_degenerate(self):
        assert regime._is_degenerate(self._healthy_model()) is False

    def test_transmat_row_summing_to_zero_is_degenerate(self):
        model = self._healthy_model(n_components=2)
        model.transmat_ = np.array([[1.0, 0.0], [0.0, 0.0]])  # second row: no mass at all
        assert regime._is_degenerate(model) is True

    def test_nan_means_is_degenerate(self):
        model = self._healthy_model()
        model.means_[0, 0] = np.nan
        assert regime._is_degenerate(model) is True

    def test_near_zero_diagonal_variance_is_degenerate(self):
        model = self._healthy_model()
        model.covars_[1, 2, 2] = 1e-10  # one state, one feature, collapsed variance
        assert regime._is_degenerate(model) is True

    def test_zero_off_diagonal_is_not_degenerate(self):
        # Off-diagonal zeros are structural for covariance_type="diag" --
        # must NOT be flagged (this was a real bug: an earlier version of
        # _is_degenerate checked the full matrix instead of just the
        # diagonal, so every legitimately diagonal-covariance fit was
        # incorrectly rejected).
        model = self._healthy_model()
        assert (model.covars_ == np.stack([np.eye(4) * 0.1 for _ in range(3)])).all()  # sanity: has off-diag zeros
        assert regime._is_degenerate(model) is False


class TestSimpleRegimeFallback:
    def test_high_vix_is_crisis(self):
        benchmark = _synthetic_benchmark(n_days=250)
        macro = _synthetic_macro(n_days=250)
        macro = macro.copy()
        macro["vix"] = 35.0
        as_of = benchmark.index[-1]

        assert regime._simple_regime_fallback(benchmark, macro, as_of) == "CRISIS"

    def test_deep_drawdown_below_200ma_is_crisis_even_with_calm_vix(self):
        dates = pd.bdate_range("2020-01-01", periods=250)
        # Flat for 200 days (establishes the 200-day MA around 100), then
        # a sharp final drop -- price ends far below its own trailing MA.
        closes = np.concatenate([np.full(220, 100.0), np.linspace(100.0, 70.0, 30)])
        benchmark = pd.DataFrame({"close": closes}, index=dates)
        macro = _synthetic_macro(n_days=250)
        macro = macro.copy()
        macro["vix"] = 14.0  # deliberately calm VIX -- price alone should still trigger CRISIS
        as_of = dates[-1]

        assert regime._simple_regime_fallback(benchmark, macro, as_of) == "CRISIS"

    def test_calm_uptrend_is_bull_trend(self):
        dates = pd.bdate_range("2020-01-01", periods=250)
        closes = 100 * (1.0015 ** np.arange(250))
        benchmark = pd.DataFrame({"close": closes}, index=dates)
        macro = _synthetic_macro(n_days=250)
        macro = macro.copy()
        macro["vix"] = 12.0
        as_of = dates[-1]

        assert regime._simple_regime_fallback(benchmark, macro, as_of) == "BULL_TREND"

    def test_insufficient_price_history_is_transition(self):
        benchmark = _synthetic_benchmark(n_days=50)  # under the 200-day MA requirement
        macro = _synthetic_macro(n_days=50)
        as_of = benchmark.index[-1]

        assert regime._simple_regime_fallback(benchmark, macro, as_of) == "TRANSITION"


class TestResolveExposureTable:
    def test_no_overrides_returns_module_defaults(self):
        table = regime.resolve_exposure_table({})
        assert table == regime.GROSS_EXPOSURE_BY_REGIME

    def test_none_regime_cfg_returns_module_defaults(self):
        assert regime.resolve_exposure_table(None) == regime.GROSS_EXPOSURE_BY_REGIME

    def test_partial_override_only_replaces_given_keys(self):
        table = regime.resolve_exposure_table({"exposure_bear": 0.20, "exposure_crisis": 0.05})
        assert table["BEAR_TREND"] == 0.20
        assert table["CRISIS"] == 0.05
        assert table["BULL_TREND"] == regime.GROSS_EXPOSURE_BY_REGIME["BULL_TREND"]
        assert table["LOW_VOL_GRIND"] == regime.GROSS_EXPOSURE_BY_REGIME["LOW_VOL_GRIND"]

    def test_transition_floor_is_overridable(self):
        # TRANSITION previously only ever fell through to the hardcoded
        # DEFAULT_GROSS_EXPOSURE fallback -- confirm it's now a real,
        # tunable entry like the other four regimes.
        assert regime.GROSS_EXPOSURE_BY_REGIME["TRANSITION"] == regime.DEFAULT_GROSS_EXPOSURE
        table = regime.resolve_exposure_table({"exposure_transition": 0.85})
        assert table["TRANSITION"] == 0.85


class TestEvaluateRegime:
    def test_falls_back_to_conservative_default_when_insufficient_history(self):
        prices = _synthetic_prices(n_days=5)
        benchmark = _synthetic_benchmark(n_days=5)
        macro = _synthetic_macro(n_days=5)
        as_of = benchmark.index[-1]

        label, exposure, overrides = regime.evaluate_regime(prices, benchmark, macro, as_of)

        # Conservative-by-default, not "assume it's fine and invest fully"
        # -- see _interpret_regimes's docstring for why the old
        # full-exposure fallback was a real bug.
        assert label == "TRANSITION"
        assert exposure == regime.DEFAULT_GROSS_EXPOSURE
        assert overrides is None

    def test_end_to_end_returns_valid_regime_state(self):
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        macro = _synthetic_macro()
        as_of = benchmark.index[-1]

        label, exposure, overrides = regime.evaluate_regime(prices, benchmark, macro, as_of)

        assert label in regime.REGIME_LABELS
        assert 0.0 < exposure <= 1.0
        assert overrides is None or set(overrides.keys()) == {
            "momentum",
            "mean_reversion",
            "volatility",
            "value",
            "quality",
            "ml_alpha",
        }

    def test_degenerate_fit_uses_rule_based_fallback_not_a_constant(self, monkeypatch):
        # A real (if simple) regime read from _simple_regime_fallback,
        # not the old constant "TRANSITION" -- forces RegimeClassifier.fit
        # to always raise (simulating a degenerate fit) and checks the
        # returned label matches what the rule-based fallback would say
        # about this exact benchmark/macro data.
        prices = _synthetic_prices()
        benchmark = _synthetic_benchmark()
        macro = _synthetic_macro()
        macro = macro.copy()
        macro["vix"] = 35.0  # unambiguously CRISIS under the fallback rule
        as_of = benchmark.index[-1]

        def _always_degenerate(self, features):
            raise ValueError("degenerate HMM fit (forced for test)")

        monkeypatch.setattr(regime.RegimeClassifier, "fit", _always_degenerate)

        label, exposure, overrides = regime.evaluate_regime(prices, benchmark, macro, as_of)

        assert label == "CRISIS"
        assert exposure == regime.GROSS_EXPOSURE_BY_REGIME["CRISIS"]

    def test_exposure_table_override_is_honored_on_the_fallback_path(self):
        prices = _synthetic_prices(n_days=5)
        benchmark = _synthetic_benchmark(n_days=5)
        macro = _synthetic_macro(n_days=5)
        as_of = benchmark.index[-1]

        label, exposure, _ = regime.evaluate_regime(
            prices, benchmark, macro, as_of, exposure_table={"TRANSITION": 0.42}
        )

        assert label == "TRANSITION"
        assert exposure == 0.42
