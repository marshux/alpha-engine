"""Unit tests for signals/* against synthetic price series with known shapes,
so each assertion is about a directional relationship we can reason about
by hand rather than a black-box comparison against real market data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.signals import mean_reversion, momentum, volatility


def _series_df(closes: np.ndarray, n_days: int) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": 1_000_000,
        },
        index=dates,
    )


def test_momentum_ranks_uptrend_above_downtrend():
    n = 280
    up = _series_df(100 * (1.001 ** np.arange(n)), n)
    down = _series_df(100 * (0.999 ** np.arange(n)), n)
    flat = _series_df(np.full(n, 100.0), n)
    prices = {"UP": up, "DOWN": down, "FLAT": flat}

    scores = momentum.compute(prices, as_of=up.index[-1])

    assert scores["UP"] > scores["FLAT"] > scores["DOWN"]


def test_momentum_skips_illiquid_history():
    n = 50  # shorter than LOOKBACK_DAYS + 1
    short = _series_df(np.full(n, 100.0), n)
    scores = momentum.compute({"SHORT": short}, as_of=short.index[-1])
    assert "SHORT" not in scores.index


def test_mean_reversion_favors_recent_dip_over_recent_spike():
    n = 40
    base = np.full(n, 100.0)
    spiked = base.copy()
    spiked[-1] = 130.0  # sharp spike right before as_of
    dipped = base.copy()
    dipped[-1] = 70.0  # sharp dip right before as_of
    prices = {
        "SPIKED": _series_df(spiked, n),
        "DIPPED": _series_df(dipped, n),
    }
    as_of = prices["SPIKED"].index[-1]

    scores = mean_reversion.compute(prices, as_of=as_of)

    # dip -> expect bounce -> higher score; spike -> expect pullback -> lower score
    assert scores["DIPPED"] > scores["SPIKED"]


def test_volatility_favors_steady_over_erratic():
    n = 40
    rng = np.random.default_rng(42)
    steady = _series_df(100 + np.cumsum(rng.normal(0, 0.05, n)), n)
    erratic = _series_df(100 + np.cumsum(rng.normal(0, 5.0, n)), n)
    prices = {"STEADY": steady, "ERRATIC": erratic}
    as_of = steady.index[-1]

    scores = volatility.compute(prices, as_of=as_of)

    assert scores["STEADY"] > scores["ERRATIC"]


def test_zscore_output_is_cross_sectionally_centered():
    n = 280
    up = _series_df(100 * (1.001 ** np.arange(n)), n)
    down = _series_df(100 * (0.999 ** np.arange(n)), n)
    prices = {"UP": up, "DOWN": down}

    scores = momentum.compute(prices, as_of=up.index[-1])

    assert abs(scores.mean()) < 1e-9
