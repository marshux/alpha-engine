from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest import metrics
from alpha_engine.backtest.engine import rebalance_dates, run_backtest


def _series_df(closes: np.ndarray, n_days: int, start="2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000},
        index=dates,
    )


def _base_strategy_config(**overrides) -> dict:
    cfg = {
        "rebalance": "weekly",
        "signal_weights": {"momentum": 1.0, "mean_reversion": 0.0, "volatility": 0.0},
        "portfolio": {
            "top_n": 2,
            "max_position_weight": 0.6,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
        "risk": {"stop_loss_pct": 0.5, "max_drawdown_pct": 0.99},
        "costs": {"commission_bps": 0.0, "slippage_bps": 0.0},
        "backtest": {"initial_capital": 100_000, "benchmark": "SPY"},
    }
    cfg.update(overrides)
    return cfg


def test_rebalance_dates_weekly_snaps_to_trading_days():
    all_dates = pd.bdate_range("2020-01-01", periods=60)
    dates = rebalance_dates(all_dates, "weekly")
    assert len(dates) > 0
    assert dates.isin(all_dates).all()
    assert dates.is_monotonic_increasing


def test_rebalance_dates_rejects_unknown_cadence():
    all_dates = pd.bdate_range("2020-01-01", periods=10)
    with pytest.raises(ValueError):
        rebalance_dates(all_dates, "daily")


def test_backtest_grows_equity_when_all_assets_trend_up():
    n = 300
    up_a = _series_df(100 * (1.001 ** np.arange(n)), n)
    up_b = _series_df(50 * (1.0015 ** np.arange(n)), n)
    prices = {"A": up_a, "B": up_b}
    config = _base_strategy_config()

    result = run_backtest(prices, config["signal_weights"], config)

    assert len(result.equity_curve) > 1
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]


def test_backtest_no_lookahead_first_period_flat():
    # Equity should not change between the very first rebalance point and
    # the initial capital, since there's no prior-period weight to have
    # accrued a return with yet.
    n = 300
    prices = {"A": _series_df(100 * (1.001 ** np.arange(n)), n)}
    config = _base_strategy_config()
    config["portfolio"]["top_n"] = 1

    result = run_backtest(prices, config["signal_weights"], config)

    first_equity = result.equity_curve.iloc[0]
    assert first_equity == pytest.approx(config["backtest"]["initial_capital"])


def test_backtest_costs_reduce_final_equity():
    n = 300
    rng = np.random.default_rng(7)
    a = _series_df(100 * np.cumprod(1 + rng.normal(0.001, 0.01, n)), n)
    b = _series_df(80 * np.cumprod(1 + rng.normal(0.001, 0.01, n)), n)
    prices = {"A": a, "B": b}

    cheap = _base_strategy_config()
    expensive = _base_strategy_config(costs={"commission_bps": 50.0, "slippage_bps": 100.0})

    cheap_result = run_backtest(prices, cheap["signal_weights"], cheap)
    expensive_result = run_backtest(prices, expensive["signal_weights"], expensive)

    assert expensive_result.equity_curve.iloc[-1] < cheap_result.equity_curve.iloc[-1]


def test_circuit_breaker_flattens_and_halts_losses():
    n = 300
    # sharp crash partway through
    closes = np.concatenate([np.full(150, 100.0), np.linspace(100.0, 20.0, n - 150)])
    prices = {"A": _series_df(closes, n)}
    config = _base_strategy_config(
        risk={"stop_loss_pct": 0.99, "max_drawdown_pct": 0.10},
        portfolio={
            "top_n": 1,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
    )

    result = run_backtest(prices, config["signal_weights"], config)

    # once flattened, equity should stay constant (no further losses accrue)
    tail = result.equity_curve.iloc[-5:]
    assert tail.max() == pytest.approx(tail.min(), rel=1e-9)
    # some weights_history entries should be empty (flattened to cash)
    assert any(w.empty for w in result.weights_history.values())


def test_metrics_cagr_and_drawdown_on_known_curve():
    dates = pd.bdate_range("2020-01-01", periods=3)
    curve = pd.Series([100.0, 150.0, 90.0], index=dates)

    dd = metrics.max_drawdown(curve)
    assert dd == pytest.approx(90.0 / 150.0 - 1.0)

    flat = pd.Series([100.0, 100.0, 100.0], index=dates)
    assert metrics.sharpe_ratio(flat, periods_per_year=52) == 0.0
