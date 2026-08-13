from __future__ import annotations

import pandas as pd
import pytest

from alpha_engine.backtest import metrics


def _curve(closes: list[float]) -> pd.Series:
    dates = pd.bdate_range("2020-01-01", periods=len(closes))
    return pd.Series(closes, index=dates)


def test_downside_deviation_zero_when_no_negative_returns():
    curve = _curve([100, 110, 121, 133.1])
    assert metrics.downside_deviation(curve, periods_per_year=52) == 0.0


def test_downside_deviation_positive_when_negative_returns_present():
    curve = _curve([100, 105, 95, 110, 90])
    dd = metrics.downside_deviation(curve, periods_per_year=52)
    returns = curve.pct_change().dropna()
    expected = returns[returns < 0].std() * (52 ** 0.5)
    assert dd == pytest.approx(expected)
    assert dd > 0


def test_sortino_zero_on_flat_curve():
    curve = _curve([100, 100, 100, 100])
    assert metrics.sortino_ratio(curve, periods_per_year=52) == 0.0


def test_sortino_positive_when_returns_trend_up():
    curve = _curve([100, 105, 103, 110, 108, 115])
    assert metrics.sortino_ratio(curve, periods_per_year=52) > 0


def test_calmar_matches_cagr_over_abs_drawdown():
    curve = _curve([100, 150, 90, 180])
    calmar = metrics.calmar_ratio(curve)
    expected = metrics.cagr(curve) / abs(metrics.max_drawdown(curve))
    assert calmar == pytest.approx(expected)


def test_calmar_zero_when_no_drawdown():
    curve = _curve([100, 100, 100])
    assert metrics.calmar_ratio(curve) == 0.0


def test_win_rate_counts_positive_periods():
    curve = _curve([100, 110, 121, 133.1, 146.41, 140])  # 4 up, 1 down
    assert metrics.win_rate(curve) == pytest.approx(0.8)


def test_win_rate_empty_curve_is_zero():
    assert metrics.win_rate(pd.Series(dtype=float)) == 0.0


def test_summarize_includes_new_metrics():
    curve = _curve([100, 105, 103, 110, 108, 115])
    summary = metrics.summarize(curve, benchmark_curve=None, periods_per_year=52)
    for key in ("sortino", "calmar", "downside_deviation", "win_rate"):
        assert key in summary
