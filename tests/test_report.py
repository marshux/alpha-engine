"""Smoke test for tear-sheet generation -- confirms the figure renders and
saves without erroring on realistic-shaped inputs. Not asserting on pixel
content, just that the reporting path doesn't crash.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_engine.backtest.report import generate_tearsheet


def test_generate_tearsheet_writes_a_nonempty_file(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=60, freq="W-FRI")
    rng = np.random.default_rng(1)
    equity = pd.Series(100_000 * np.cumprod(1 + rng.normal(0.001, 0.01, len(dates))), index=dates)
    benchmark = pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0008, 0.008, len(dates))), index=dates)
    turnover = {d: float(rng.uniform(0, 1)) for d in dates}

    output_path = Path(tmp_path) / "tearsheet.png"
    generate_tearsheet(
        equity_curve=equity,
        benchmark_curves={"SPY": benchmark},
        turnover_history=turnover,
        periods_per_year=52,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_tearsheet_handles_no_benchmarks_or_turnover(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=10, freq="W-FRI")
    equity = pd.Series(100_000 * np.linspace(1.0, 1.1, len(dates)), index=dates)

    output_path = Path(tmp_path) / "tearsheet.png"
    generate_tearsheet(
        equity_curve=equity,
        benchmark_curves={},
        turnover_history={},
        periods_per_year=52,
        output_path=output_path,
    )

    assert output_path.exists()
