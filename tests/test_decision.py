from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.decision import portfolio, risk
from alpha_engine.decision.scorer import combine_scores


def test_combine_scores_weighted_sum():
    table = pd.DataFrame(
        {
            "momentum": [1.0, -1.0, 0.0],
            "mean_reversion": [0.0, 1.0, -1.0],
        },
        index=["A", "B", "C"],
    )
    weights = {"momentum": 0.5, "mean_reversion": 0.5}

    scores = combine_scores(table, weights)

    assert scores["A"] == pytest.approx(0.5)
    assert scores["B"] == pytest.approx(0.0)
    assert scores["C"] == pytest.approx(-0.5)
    # sorted descending
    assert list(scores.index) == ["A", "B", "C"]


def test_combine_scores_treats_missing_signal_as_neutral():
    table = pd.DataFrame({"momentum": [1.0, 2.0]}, index=["A", "B"])
    # news_ai weighted but absent from the table entirely
    weights = {"momentum": 1.0, "news_ai": 0.5}

    scores = combine_scores(table, weights)

    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(2.0)


def test_combine_scores_zero_weight_ignored_even_if_column_has_nan():
    table = pd.DataFrame({"momentum": [1.0, 2.0], "news_ai": [None, None]}, index=["A", "B"])
    weights = {"momentum": 1.0, "news_ai": 0.0}

    scores = combine_scores(table, weights)

    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(2.0)


def test_select_top_n():
    scores = pd.Series({"A": 3.0, "B": 1.0, "C": 2.0, "D": -1.0})
    top2 = portfolio.select_top_n(scores, 2)
    assert list(top2.index) == ["A", "C"]


def test_resolve_top_n_uses_fixed_count_when_top_n_pct_absent():
    scores = pd.Series({s: 0.0 for s in "ABCDEFGHIJ"})  # 10 symbols
    assert portfolio.resolve_top_n(scores, {"top_n": 3}) == 3


def test_resolve_top_n_uses_ratio_of_live_universe_when_configured():
    scores = pd.Series({s: 0.0 for s in range(63)})  # 63 symbols, this repo's actual universe size
    assert portfolio.resolve_top_n(scores, {"top_n_pct": 0.10}) == 6  # int(63 * 0.10) == 6


def test_resolve_top_n_ratio_respects_floor():
    scores = pd.Series({s: 0.0 for s in range(10)})
    assert portfolio.resolve_top_n(scores, {"top_n_pct": 0.10, "top_n_min": 5}) == 5  # int(10*0.10)=1, floored to 5


def test_resolve_top_n_ratio_respects_ceiling():
    scores = pd.Series({s: 0.0 for s in range(500)})
    config = {"top_n_pct": 0.10, "top_n_min": 10, "top_n_max": 30}
    assert portfolio.resolve_top_n(scores, config) == 30  # int(500*0.10)=50, capped to 30


def test_resolve_top_n_no_ceiling_when_top_n_max_absent():
    scores = pd.Series({s: 0.0 for s in range(500)})
    assert portfolio.resolve_top_n(scores, {"top_n_pct": 0.10, "top_n_min": 10}) == 50


def test_resolve_top_n_200_universe_lands_in_the_healthy_range():
    # This repo's actual expanded universe size (config/universe.yaml) --
    # the exact scenario systematic_core_action_plan.md targeted.
    scores = pd.Series({s: 0.0 for s in range(204)})
    config = {"top_n_pct": 0.10, "top_n_min": 10, "top_n_max": 30}
    assert portfolio.resolve_top_n(scores, config) == 20


def test_resolve_top_n_ratio_scales_with_universe_size():
    small = pd.Series({s: 0.0 for s in range(63)})
    large = pd.Series({s: 0.0 for s in range(500)})
    config = {"top_n_pct": 0.10, "top_n_min": 5}
    assert portfolio.resolve_top_n(small, config) == 6
    assert portfolio.resolve_top_n(large, config) == 50  # matches the guide's own n_long=50-of-500 ratio


def test_build_portfolio_uses_ratio_based_top_n_end_to_end():
    scores = pd.Series({s: float(i) for i, s in enumerate("ABCDEFGHIJ")})  # 10 symbols, ranked 0..9
    config = {
        "sizing": "equal_weight",
        "top_n_pct": 0.30,
        "max_position_weight": 1.0,
        "max_sector_weight": 1.0,
    }

    weights = portfolio.build_portfolio(scores, None, sector_map={}, config=config)

    assert len(weights) == 3  # int(10 * 0.30) == 3
    assert set(weights.index) == {"H", "I", "J"}  # the 3 highest-scored symbols


def test_size_positions_equal_weight():
    selected = pd.Series({"A": 3.0, "B": 2.0})
    weights = portfolio.size_positions(selected, None, "equal_weight", max_position_weight=1.0)
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)


def test_size_positions_volatility_scaled_favors_low_vol():
    selected = pd.Series({"A": 1.0, "B": 1.0})
    vol = pd.Series({"A": 0.1, "B": 0.4})
    weights = portfolio.size_positions(
        selected, vol, "volatility_scaled", max_position_weight=1.0
    )
    assert weights["A"] > weights["B"]
    assert weights.sum() == pytest.approx(1.0)


def test_size_positions_caps_leave_excess_as_cash():
    selected = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    weights = portfolio.size_positions(
        selected, None, "equal_weight", max_position_weight=0.2
    )
    # 1/3 each exceeds the 0.2 cap -> each capped to 0.2, total invested is
    # 0.6 (60%), the remaining 40% is deliberately left as cash, not
    # redistributed
    assert (weights <= 0.2 + 1e-9).all()
    assert weights.sum() == pytest.approx(0.6)


def test_capping_diagnostics_matches_the_known_excess():
    selected = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    diag = portfolio.capping_diagnostics(selected, None, "equal_weight", max_position_weight=0.2)
    # same setup as test_size_positions_caps_leave_excess_as_cash: raw 1/3
    # each vs 0.2 cap -> all 3 capped, 0.4 total excess trapped as cash
    assert diag["n_capped"] == 3
    assert diag["excess_weight"] == pytest.approx(0.4)


def test_capping_diagnostics_reports_zero_when_nothing_is_capped():
    selected = pd.Series({"A": 1.0, "B": 1.0})
    diag = portfolio.capping_diagnostics(selected, None, "equal_weight", max_position_weight=0.9)
    assert diag["n_capped"] == 0
    assert diag["excess_weight"] == pytest.approx(0.0)


def test_capping_diagnostics_empty_selection():
    diag = portfolio.capping_diagnostics(pd.Series(dtype=float), None, "equal_weight", max_position_weight=0.2)
    assert diag == {"n_capped": 0, "excess_weight": 0.0}


class TestRedistributeCappedExcess:
    def test_default_is_off_and_matches_uncapped_behavior(self):
        selected = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
        weights = portfolio.size_positions(selected, None, "equal_weight", max_position_weight=0.2)
        assert weights.sum() == pytest.approx(0.6)  # unchanged from the non-redistributing default

    def test_single_capped_position_redistributes_fully_when_others_have_room(self):
        # vol-scaled raw: A=0.6667, B=0.1667, C=0.1667 (inv_vol 20/5/5).
        # cap=0.4 -> A capped, its 0.2667 excess splits evenly onto B/C
        # (equal current weight), landing both at 0.3 -- under the cap,
        # so it converges in one round with the full 1.0 invested, none
        # left as cash.
        selected = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
        vol = pd.Series({"A": 0.05, "B": 0.2, "C": 0.2})
        weights = portfolio.size_positions(
            selected, vol, "volatility_scaled", max_position_weight=0.4, redistribute_capped_excess=True
        )
        assert weights["A"] == pytest.approx(0.4)
        assert weights["B"] == pytest.approx(0.3)
        assert weights["C"] == pytest.approx(0.3)
        assert weights.sum() == pytest.approx(1.0)

    def test_cascading_caps_converge_to_the_structural_ceiling(self):
        # Heavily skewed raw weights (inv_vol 50/10/2) -> redistributing
        # A's excess pushes B over the cap too, then C -- three rounds of
        # cascading caps. cap=0.3 with n=3 means 0.9 is the hard ceiling
        # regardless of the original skew, so all three should converge
        # to exactly the cap with 0.1 left unallocated.
        selected = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
        vol = pd.Series({"A": 0.02, "B": 0.1, "C": 0.5})
        weights = portfolio.size_positions(
            selected, vol, "volatility_scaled", max_position_weight=0.3, redistribute_capped_excess=True
        )
        assert weights["A"] == pytest.approx(0.3)
        assert weights["B"] == pytest.approx(0.3)
        assert weights["C"] == pytest.approx(0.3)
        assert weights.sum() == pytest.approx(0.9)  # 3 * 0.3, the real structural ceiling

    def test_build_portfolio_honors_the_config_flag(self):
        scores = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0})
        vol = pd.Series({"A": 0.05, "B": 0.2, "C": 0.2})
        config = {
            "top_n": 3,
            "sizing": "volatility_scaled",
            "max_position_weight": 0.4,
            "max_sector_weight": 1.0,
            "redistribute_capped_excess": True,
        }
        weights = portfolio.build_portfolio(scores, vol, sector_map={}, config=config)
        assert weights.sum() == pytest.approx(1.0)


def test_size_positions_below_cap_is_unaffected():
    selected = pd.Series({"A": 1.0, "B": 1.0})
    weights = portfolio.size_positions(
        selected, None, "equal_weight", max_position_weight=0.9
    )
    assert weights.sum() == pytest.approx(1.0)


def test_size_positions_volatility_scaled_requires_full_coverage():
    selected = pd.Series({"A": 1.0, "B": 1.0})
    vol = pd.Series({"A": 0.1})  # missing B
    with pytest.raises(ValueError):
        portfolio.size_positions(selected, vol, "volatility_scaled", max_position_weight=1.0)


def test_min_variance_weights_favor_lower_variance_uncorrelated_asset():
    rng = pd.Series(range(100)).astype(float)
    returns = pd.DataFrame(
        {
            "LOW_VOL": (rng % 2 - 0.5).to_numpy() * 0.001,  # tiny, alternating
            "HIGH_VOL": (rng % 5 - 2).to_numpy() * 0.05,  # much larger swings
        }
    )

    weights = portfolio._min_variance_weights(returns)

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert weights["LOW_VOL"] > weights["HIGH_VOL"]


def test_min_variance_weights_falls_back_to_equal_weight_on_degenerate_covariance():
    returns = pd.DataFrame({"A": [0.0] * 10, "B": [0.0] * 10})
    weights = portfolio._min_variance_weights(returns)
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)


def test_size_positions_min_variance_requires_dataframe():
    selected = pd.Series({"A": 1.0, "B": 1.0})
    vol_series = pd.Series({"A": 0.1, "B": 0.2})  # wrong type for min_variance
    with pytest.raises(ValueError):
        portfolio.size_positions(selected, vol_series, "min_variance", max_position_weight=1.0)


def test_size_positions_min_variance_requires_full_coverage():
    selected = pd.Series({"A": 1.0, "B": 1.0})
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03]})  # missing B
    with pytest.raises(ValueError):
        portfolio.size_positions(selected, returns, "min_variance", max_position_weight=1.0)


def test_size_positions_min_variance_matches_direct_computation():
    selected = pd.Series({"A": 1.0, "B": 1.0})
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0, 0.02], "B": [0.05, -0.10, 0.08, -0.03, 0.06]})

    result = portfolio.size_positions(selected, returns, "min_variance", max_position_weight=1.0)
    expected = portfolio._min_variance_weights(returns)

    assert result["A"] == pytest.approx(expected["A"])
    assert result["B"] == pytest.approx(expected["B"])


def test_apply_sector_cap_scales_down_overweight_sector():
    weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    sector_map = {"A": "tech", "B": "tech", "C": "energy"}
    capped = portfolio.apply_sector_cap(weights, sector_map, max_sector_weight=0.4)
    assert capped["A"] + capped["B"] == pytest.approx(0.4)
    assert capped["C"] == pytest.approx(0.4)  # untouched, already at cap


def test_apply_sector_cap_leaves_unknown_symbols_alone():
    weights = pd.Series({"A": 0.6})
    capped = portfolio.apply_sector_cap(weights, sector_map={}, max_sector_weight=0.3)
    assert capped["A"] == pytest.approx(0.6)


def test_stopped_out_positions():
    entries = {"A": 100.0, "B": 50.0}
    current = {"A": 84.0, "B": 49.0}  # A down 16%, B down 2%
    stopped = risk.stopped_out_positions(entries, current, stop_loss_pct=0.15)
    assert stopped == {"A"}


def test_circuit_breaker_triggers_on_drawdown():
    equity = pd.Series([100, 120, 110, 90])  # peak 120 -> 90 is -25%
    assert risk.circuit_breaker_triggered(equity, max_drawdown_pct=0.20) is True


def test_circuit_breaker_not_triggered_within_tolerance():
    equity = pd.Series([100, 120, 110])  # peak 120 -> 110 is ~-8.3%
    assert risk.circuit_breaker_triggered(equity, max_drawdown_pct=0.20) is False


def _benchmark_df(closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000},
        index=dates,
    )


def test_regime_is_risk_on_true_when_price_above_moving_average():
    closes = [100.0] * 20 + [150.0]  # sharp jump above its own trailing average
    df = _benchmark_df(closes)
    assert risk.regime_is_risk_on(df, as_of=df.index[-1], ma_window=20) is True


def test_regime_is_risk_on_false_when_price_below_moving_average():
    closes = [100.0] * 20 + [50.0]  # sharp drop below its own trailing average
    df = _benchmark_df(closes)
    assert risk.regime_is_risk_on(df, as_of=df.index[-1], ma_window=20) is False


def test_regime_is_risk_on_defaults_true_with_insufficient_history():
    closes = [50.0] * 5  # far fewer than ma_window days
    df = _benchmark_df(closes)
    assert risk.regime_is_risk_on(df, as_of=df.index[-1], ma_window=20) is True


def test_regime_is_risk_on_respects_as_of_no_lookahead():
    # mild uptrend (clearly above its own trailing MA at as_of), THEN a
    # crash after as_of -- the crash must not affect the as_of judgment
    closes = list(np.linspace(100.0, 120.0, 25)) + [10.0] * 5
    df = _benchmark_df(closes)
    as_of = df.index[24]  # last day before the crash
    assert risk.regime_is_risk_on(df, as_of=as_of, ma_window=20) is True


def _price_series_with_shock(lookback_returns: list[float], shock_return: float) -> pd.DataFrame:
    """A price history whose daily returns equal `lookback_returns` in
    order, followed by one final day with `shock_return` -- lets a test
    control both the trailing distribution and the outlier move exactly.
    """
    closes = [100.0]
    for r in lookback_returns:
        closes.append(closes[-1] * (1 + r))
    closes.append(closes[-1] * (1 + shock_return))
    return _benchmark_df(closes)


def test_anomalous_drop_positions_flags_outlier_drop():
    lookback = [0.01 if i % 2 == 0 else -0.01 for i in range(60)]  # ~1% typical daily swing
    df = _price_series_with_shock(lookback, shock_return=-0.20)  # a huge one-day crash
    flagged = risk.anomalous_drop_positions({"A": df}, {"A"}, as_of=df.index[-1], z_threshold=3.0, lookback_days=60)
    assert flagged == {"A"}


def test_anomalous_drop_positions_does_not_flag_ordinary_move():
    lookback = [0.01 if i % 2 == 0 else -0.01 for i in range(60)]
    df = _price_series_with_shock(lookback, shock_return=-0.005)  # in line with its normal daily swing
    flagged = risk.anomalous_drop_positions({"A": df}, {"A"}, as_of=df.index[-1], z_threshold=3.0, lookback_days=60)
    assert flagged == set()


def test_anomalous_drop_positions_ignores_upside_spike():
    lookback = [0.01 if i % 2 == 0 else -0.01 for i in range(60)]
    df = _price_series_with_shock(lookback, shock_return=0.20)  # a huge one-day POP, not a drop
    flagged = risk.anomalous_drop_positions({"A": df}, {"A"}, as_of=df.index[-1], z_threshold=3.0, lookback_days=60)
    assert flagged == set()


def test_anomalous_drop_positions_requires_full_lookback_history():
    lookback = [0.01 if i % 2 == 0 else -0.01 for i in range(10)]  # far short of lookback_days=60
    df = _price_series_with_shock(lookback, shock_return=-0.20)
    flagged = risk.anomalous_drop_positions({"A": df}, {"A"}, as_of=df.index[-1], z_threshold=3.0, lookback_days=60)
    assert flagged == set()


def test_anomalous_drop_positions_respects_z_threshold():
    lookback = [0.01 if i % 2 == 0 else -0.01 for i in range(60)]
    df = _price_series_with_shock(lookback, shock_return=-0.03)  # a moderate, non-extreme drop
    loose = risk.anomalous_drop_positions({"A": df}, {"A"}, as_of=df.index[-1], z_threshold=1.0, lookback_days=60)
    strict = risk.anomalous_drop_positions({"A": df}, {"A"}, as_of=df.index[-1], z_threshold=10.0, lookback_days=60)
    assert loose == {"A"}
    assert strict == set()


def test_modulate_theme_cap_positive_sentiment_loosens_cap():
    base = 0.35
    loosened = portfolio.modulate_theme_cap(base, sentiment=1.0, confidence=1.0, sensitivity=0.5)
    assert loosened > base


def test_modulate_theme_cap_negative_sentiment_tightens_cap():
    base = 0.35
    tightened = portfolio.modulate_theme_cap(base, sentiment=-1.0, confidence=1.0, sensitivity=0.5)
    assert tightened < base


def test_modulate_theme_cap_low_confidence_barely_moves_cap():
    base = 0.35
    # Strong sentiment but near-zero confidence should land close to base.
    result = portfolio.modulate_theme_cap(base, sentiment=1.0, confidence=0.01, sensitivity=0.5)
    assert abs(result - base) < 0.01


def test_modulate_theme_cap_neutral_sentiment_is_a_no_op():
    base = 0.35
    assert portfolio.modulate_theme_cap(base, sentiment=0.0, confidence=1.0) == base


def test_modulate_theme_cap_clips_to_bounds():
    # Extreme sentiment/confidence/sensitivity would blow past any
    # reasonable cap without the clip.
    result = portfolio.modulate_theme_cap(
        0.35, sentiment=1.0, confidence=1.0, sensitivity=10.0, min_weight=0.15, max_weight=0.50
    )
    assert result == 0.50

    result = portfolio.modulate_theme_cap(
        0.35, sentiment=-1.0, confidence=1.0, sensitivity=10.0, min_weight=0.15, max_weight=0.50
    )
    assert result == 0.15


def test_scale_gross_exposure_full_exposure_is_a_no_op():
    weights = pd.Series({"AAA": 0.3, "BBB": 0.2})
    result = portfolio.scale_gross_exposure(weights, 1.0)
    pd.testing.assert_series_equal(result, weights)


def test_scale_gross_exposure_scales_down_and_leaves_remainder_as_cash():
    weights = pd.Series({"AAA": 0.3, "BBB": 0.2})
    result = portfolio.scale_gross_exposure(weights, 0.5)
    pd.testing.assert_series_equal(result, pd.Series({"AAA": 0.15, "BBB": 0.1}))
    assert result.sum() < weights.sum()


def test_scale_gross_exposure_zero_flattens_to_all_cash():
    weights = pd.Series({"AAA": 0.3, "BBB": 0.2})
    result = portfolio.scale_gross_exposure(weights, 0.0)
    assert (result == 0.0).all()
