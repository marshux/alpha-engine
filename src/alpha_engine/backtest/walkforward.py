"""Walk-forward validation: split history into rolling train/test windows,
optionally grid-searching strategy config on each train window, and always
evaluating out-of-sample on the following test window. This is what
actually answers "does this hold up out-of-sample, or is it curve-fit to
one specific historical period" -- a single full-period backtest can't.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import pandas as pd

from alpha_engine.backtest.engine import PERIODS_PER_YEAR, run_backtest
from alpha_engine.backtest.metrics import sharpe_ratio, summarize


@dataclass
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _years_to_timedelta(years: float) -> pd.Timedelta:
    # pd.DateOffset(years=...) rejects fractional years (dateutil
    # limitation) -- convert to days instead, same 365.25 convention
    # backtest/metrics.cagr already uses for year-fraction math.
    return pd.Timedelta(days=round(years * 365.25))


def generate_windows(
    all_dates: pd.DatetimeIndex, train_years: float, test_years: float, step_years: float
) -> list[Window]:
    """Rolling windows across `all_dates`'s range: a `train_years`-long
    train period immediately followed by a `test_years`-long test period,
    stepping forward `step_years` each time until the next test window
    would run past the end of available data.
    """
    if train_years <= 0 or test_years <= 0 or step_years <= 0:
        raise ValueError("train_years, test_years, and step_years must all be positive")

    range_start = all_dates.min()
    range_end = all_dates.max()
    train_span = _years_to_timedelta(train_years)
    test_span = _years_to_timedelta(test_years)
    step_span = _years_to_timedelta(step_years)

    windows: list[Window] = []
    train_start = range_start
    while True:
        train_end = train_start + train_span
        test_start = train_end
        test_end = test_start + test_span
        if test_end > range_end:
            break
        windows.append(Window(train_start, train_end, test_start, test_end))
        train_start = train_start + step_span
    return windows


def _apply_overrides(base_config: dict, overrides: dict[str, object]) -> dict:
    """`overrides`: dotted-path -> value, e.g. {"portfolio.top_n": 10}.
    Returns a deep copy of `base_config` with each path set; never mutates
    `base_config` (each grid candidate needs its own independent config).
    """
    config = copy.deepcopy(base_config)
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        node = config
        for key in parents:
            node = node[key]
        node[leaf] = value
    return config


def _grid_combinations(param_grid: dict[str, list] | None) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(param_grid[k] for k in keys))]


def run_walk_forward(
    prices: dict[str, pd.DataFrame],
    base_config: dict,
    windows: list[Window],
    param_grid: dict[str, list] | None = None,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per window: the config that scored best (by Sharpe) on the
    train sub-range -- or `base_config` unchanged if `param_grid` is
    None/empty, in which case this is a pure out-of-sample consistency
    check across regimes rather than an optimization -- and that config's
    out-of-sample metrics on the following test sub-range. `prices` is
    passed through unsliced to every `run_backtest` call; only the
    rebalance-date range changes, so signals never lose lookback history
    at a window boundary (see run_backtest's `start`/`end` docstring).
    """
    rows = []
    for window in windows:
        best_config = base_config
        best_overrides: dict = {}

        if param_grid:
            best_score = float("-inf")
            for overrides in _grid_combinations(param_grid):
                candidate = _apply_overrides(base_config, overrides)
                train_result = run_backtest(
                    prices,
                    candidate["signal_weights"],
                    candidate,
                    sector_map=sector_map,
                    start=window.train_start,
                    end=window.train_end,
                )
                if train_result.equity_curve.empty:
                    continue
                score = sharpe_ratio(
                    train_result.equity_curve, PERIODS_PER_YEAR[candidate["rebalance"]]
                )
                if score > best_score:
                    best_score, best_config, best_overrides = score, candidate, overrides

        test_result = run_backtest(
            prices,
            best_config["signal_weights"],
            best_config,
            sector_map=sector_map,
            start=window.test_start,
            end=window.test_end,
        )
        summary = summarize(test_result.equity_curve, None, PERIODS_PER_YEAR[best_config["rebalance"]])

        rows.append(
            {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "winning_overrides": best_overrides,
                **summary,
            }
        )
    return pd.DataFrame(rows)
