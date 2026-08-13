"""Tests for signals/ai_theme.py and decision/portfolio.apply_theme_cap --
the AI directional-tilt signal and its concentration-cap "hedge".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.decision import portfolio
from alpha_engine.signals import ai_theme


def _series_df(closes: np.ndarray, n_days: int) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000},
        index=dates,
    )


def test_ai_theme_favors_higher_intensity_members():
    n = 30
    prices = {s: _series_df(np.full(n, 100.0), n) for s in ["CORE", "PERIPHERAL", "UNTAGGED"]}
    theme_weights = {"CORE": 1.0, "PERIPHERAL": 0.5}

    scores = ai_theme.compute(prices, as_of=prices["CORE"].index[-1], theme_weights=theme_weights)

    assert "UNTAGGED" not in scores.index  # absent, not zero -- combine_scores treats absent as neutral
    assert scores["CORE"] > scores["PERIPHERAL"]


def test_ai_theme_ignores_tagged_symbols_not_in_current_universe():
    n = 10
    prices = {"A": _series_df(np.full(n, 100.0), n)}
    theme_weights = {"A": 1.0, "NOT_IN_UNIVERSE": 1.0}

    scores = ai_theme.compute(prices, as_of=prices["A"].index[-1], theme_weights=theme_weights)

    assert list(scores.index) == ["A"]


def test_ai_theme_empty_when_no_overlap():
    n = 10
    prices = {"A": _series_df(np.full(n, 100.0), n)}
    scores = ai_theme.compute(prices, as_of=prices["A"].index[-1], theme_weights={"OTHER": 1.0})
    assert scores.empty


def test_apply_theme_cap_scales_down_overweight_theme():
    weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
    theme = {"A": 1.0, "B": 0.5}  # A and B are theme members, C is not

    capped = portfolio.apply_theme_cap(weights, theme, max_theme_weight=0.4)

    assert capped["A"] + capped["B"] == pytest.approx(0.4)
    assert capped["C"] == pytest.approx(0.4)  # untouched, not a theme member


def test_apply_theme_cap_noop_when_under_cap():
    weights = pd.Series({"A": 0.1, "B": 0.2})
    capped = portfolio.apply_theme_cap(weights, {"A": 1.0}, max_theme_weight=0.5)
    assert capped["A"] == pytest.approx(0.1)
    assert capped["B"] == pytest.approx(0.2)


def test_apply_theme_cap_noop_with_empty_theme():
    weights = pd.Series({"A": 0.6})
    capped = portfolio.apply_theme_cap(weights, {}, max_theme_weight=0.3)
    assert capped["A"] == pytest.approx(0.6)


def test_build_portfolio_applies_theme_cap_when_configured():
    scores = pd.Series({"A": 2.0, "B": 1.0, "C": 0.5})
    config = {
        "top_n": 3,
        "max_position_weight": 1.0,
        "max_sector_weight": 1.0,
        "max_theme_weight": 0.3,
        "sizing": "equal_weight",
    }
    theme_map = {"A": 1.0, "B": 1.0}  # A and B (2/3 of an equal-weight book) are theme members

    weights = portfolio.build_portfolio(scores, None, sector_map={}, config=config, theme_map=theme_map)

    assert weights["A"] + weights["B"] == pytest.approx(0.3)


def test_build_portfolio_skips_theme_cap_when_not_configured():
    scores = pd.Series({"A": 2.0, "B": 1.0})
    config = {
        "top_n": 2,
        "max_position_weight": 1.0,
        "max_sector_weight": 1.0,
        "sizing": "equal_weight",
        # no max_theme_weight key at all
    }
    weights = portfolio.build_portfolio(
        scores, None, sector_map={}, config=config, theme_map={"A": 1.0, "B": 1.0}
    )
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)
