from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.engine import PERIODS_PER_YEAR, run_backtest
from alpha_engine.backtest.metrics import sharpe_ratio
from alpha_engine.backtest.walkforward import (
    Window,
    _grid_combinations,
    generate_windows,
    run_walk_forward,
)
from alpha_engine.config import apply_overrides


def _series_df(closes: np.ndarray, n_days: int) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000},
        index=dates,
    )


def _base_strategy_config(**overrides) -> dict:
    cfg = {
        "rebalance": "weekly",
        "signal_weights": {"momentum": 1.0, "mean_reversion": 0.0, "volatility": 0.0},
        "portfolio": {
            "top_n": 1,
            "max_position_weight": 1.0,
            "max_sector_weight": 1.0,
            "sizing": "equal_weight",
        },
        "risk": {"stop_loss_pct": 0.99, "max_drawdown_pct": 0.99},
        "costs": {"commission_bps": 0.0, "slippage_bps": 0.0},
        "backtest": {"initial_capital": 100_000, "benchmark": "SPY"},
    }
    cfg.update(overrides)
    return cfg


def test_generate_windows_rolls_forward_correctly():
    all_dates = pd.bdate_range("2015-01-01", "2024-12-31")
    windows = generate_windows(all_dates, train_years=3, test_years=1, step_years=1)

    assert len(windows) > 0
    first = windows[0]
    assert first.train_start == all_dates.min()
    assert first.test_start == first.train_end
    assert first.test_end == first.train_end + pd.DateOffset(years=1)
    # every subsequent window starts one step_years later
    assert windows[1].train_start == first.train_start + pd.DateOffset(years=1)
    # no window's test period runs past the available data
    assert all(w.test_end <= all_dates.max() for w in windows)


def test_generate_windows_rejects_nonpositive_params():
    all_dates = pd.bdate_range("2015-01-01", "2020-12-31")
    with pytest.raises(ValueError):
        generate_windows(all_dates, train_years=0, test_years=1, step_years=1)


def test_generate_windows_empty_when_range_too_short():
    all_dates = pd.bdate_range("2015-01-01", "2015-06-30")
    windows = generate_windows(all_dates, train_years=3, test_years=1, step_years=1)
    assert windows == []


def test_apply_overrides_sets_nested_path_without_mutating_base():
    base = {"portfolio": {"top_n": 15}, "signal_weights": {"momentum": 0.4}}
    result = apply_overrides(base, {"portfolio.top_n": 5, "signal_weights.momentum": 0.9})

    assert result["portfolio"]["top_n"] == 5
    assert result["signal_weights"]["momentum"] == 0.9
    # original untouched
    assert base["portfolio"]["top_n"] == 15
    assert base["signal_weights"]["momentum"] == 0.4


def test_grid_combinations_cartesian_product():
    grid = {"a": [1, 2], "b": ["x", "y"]}
    combos = _grid_combinations(grid)
    assert len(combos) == 4
    assert {"a": 1, "b": "x"} in combos
    assert {"a": 2, "b": "y"} in combos


def test_grid_combinations_empty_grid_yields_single_empty_override():
    assert _grid_combinations(None) == [{}]
    assert _grid_combinations({}) == [{}]


def test_run_walk_forward_without_grid_uses_base_config_every_window():
    n = 500
    prices = {"A": _series_df(100 * (1.001 ** np.arange(n)), n)}
    all_dates = pd.DatetimeIndex(prices["A"].index)
    windows = generate_windows(all_dates, train_years=0.5, test_years=0.25, step_years=0.25)
    assert len(windows) > 0

    config = _base_strategy_config()
    results = run_walk_forward(prices, config, windows)

    assert len(results) == len(windows)
    assert (results["winning_overrides"] == {}).all()
    assert "cagr" in results.columns
    assert "sharpe" in results.columns


def test_run_walk_forward_grid_search_picks_higher_train_sharpe_config():
    """Rather than hand-predicting which momentum weight "should" win on
    noisy synthetic data (fragile -- small changes in the price path can
    flip which candidate actually scores higher), independently recompute
    each grid candidate's train-window Sharpe the same way
    run_walk_forward does internally, and confirm the recorded winner for
    each window actually matches whichever candidate scored higher. This
    validates the selection *mechanism*, not a guessed market outcome.
    """
    n = 500
    rng = np.random.default_rng(1)
    a = _series_df(100 * np.cumprod(1 + rng.normal(0.002, 0.006, n)), n)
    b = _series_df(100 * np.cumprod(1 + rng.normal(0.0005, 0.006, n)), n)
    prices = {"A": a, "B": b}
    all_dates = pd.DatetimeIndex(sorted(set(a.index) | set(b.index)))
    windows = generate_windows(all_dates, train_years=1.0, test_years=0.5, step_years=0.5)
    assert len(windows) > 0

    config = _base_strategy_config(
        signal_weights={"momentum": 0.0, "mean_reversion": 1.0, "volatility": 0.0}
    )
    momentum_candidates = [0.0, 3.0]
    param_grid = {"signal_weights.momentum": momentum_candidates}

    results = run_walk_forward(prices, config, windows, param_grid=param_grid)
    assert len(results) == len(windows)

    checked = 0
    for i, window in enumerate(windows):
        scores = {}
        for momentum_weight in momentum_candidates:
            candidate = apply_overrides(config, {"signal_weights.momentum": momentum_weight})
            train_result = run_backtest(
                prices,
                candidate["signal_weights"],
                candidate,
                start=window.train_start,
                end=window.train_end,
            )
            if train_result.equity_curve.empty:
                continue
            scores[momentum_weight] = sharpe_ratio(
                train_result.equity_curve, PERIODS_PER_YEAR[candidate["rebalance"]]
            )
        if len(scores) < len(momentum_candidates):
            continue  # too early in the series for a meaningful comparison
        expected_winner = max(scores, key=scores.get)  # first-max, matching run_walk_forward's `>` tie-break
        assert results.iloc[i]["winning_overrides"] == {"signal_weights.momentum": expected_winner}
        checked += 1

    assert checked > 0, "no window had enough history to compare -- test setup issue"
