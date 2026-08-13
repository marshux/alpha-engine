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
