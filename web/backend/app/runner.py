"""Executes a backtest/walk-forward run in the background and persists
results to the DB. This is the integration surface between the web
backend and the existing alpha_engine package -- everything downstream of
fetching data and calling run_backtest/run_walk_forward is identical to
what cli.py's cmd_backtest/cmd_walkforward already do; this just persists
structured rows instead of printing + writing local CSV/PNG files.

Runs as a FastAPI BackgroundTasks callable, so it owns its own DB session
(the request-scoped one from app.database.get_db is long gone by the time
this actually executes) and must not raise -- any exception is caught and
recorded on the Run row rather than crashing the background task silently.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import pandas as pd

from alpha_engine.backtest.engine import PERIODS_PER_YEAR, run_backtest
from alpha_engine.backtest.metrics import benchmark_equity_curve, summarize
from alpha_engine.backtest.report import generate_tearsheet
from alpha_engine.backtest.walkforward import generate_windows, run_walk_forward, total_walkforward_steps
from alpha_engine.config import load_sector_map, load_strategy, load_themes, load_universe, universe_symbols
from alpha_engine.data.equities import get_universe_prices
from alpha_engine.data.fundamentals import get_universe_fundamentals

from app.database import REPO_ROOT, SessionLocal
from app.models import BenchmarkPoint, EquityPoint, Run, WalkForwardWindow
from app.schemas import RunCreateRequest

TEARSHEET_DIR = REPO_ROOT / "data" / "web_tearsheets"


def _deep_update(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def execute_run(run_id: int, payload: RunCreateRequest, session_factory=SessionLocal) -> None:
    """`session_factory` defaults to the app's real SessionLocal; tests
    pass a session factory bound to a temp DB instead, so this can be
    tested directly without monkeypatching module internals.
    """
    db = session_factory()
    try:
        run = db.get(Run, run_id)
        run.status = "running"
        db.commit()

        universe = load_universe()
        strategy = load_strategy()
        if payload.config_overrides:
            _deep_update(strategy, payload.config_overrides)
        sector_map = load_sector_map()
        theme_weights = load_themes().get("ai", {})

        symbols = payload.symbols or universe_symbols(universe)
        primary_benchmark = strategy["backtest"]["benchmark"]
        benchmark_symbols = universe.get("benchmarks") or [primary_benchmark]

        prices = get_universe_prices(symbols, payload.start_date, payload.end_date)
        if not prices:
            raise ValueError("No price data fetched for any symbol")
        benchmark_prices = get_universe_prices(benchmark_symbols, payload.start_date, payload.end_date)

        fundamentals = get_universe_fundamentals(symbols) if payload.fundamentals else None

        if payload.run_type == "backtest":
            _run_single_backtest(
                db, run, prices, benchmark_prices, strategy, sector_map, theme_weights,
                fundamentals, primary_benchmark,
            )
        elif payload.run_type == "walkforward":
            _run_walkforward(
                db, run, prices, benchmark_prices.get(primary_benchmark), strategy, sector_map,
                theme_weights, fundamentals, payload,
            )
        else:
            raise ValueError(f"unknown run_type: {payload.run_type!r}")

        run.status = "completed"
        db.commit()
    except Exception:
        db.rollback()
        failed_run = db.get(Run, run_id)
        failed_run.status = "failed"
        failed_run.error_message = traceback.format_exc()
        db.commit()
    finally:
        db.close()


def _run_single_backtest(
    db, run, prices, benchmark_prices, strategy, sector_map, theme_weights, fundamentals, primary_benchmark
) -> None:
    result = run_backtest(
        prices,
        strategy["signal_weights"],
        strategy,
        sector_map=sector_map,
        fundamentals=fundamentals,
        benchmark_prices=benchmark_prices.get(primary_benchmark),
        theme_weights=theme_weights,
    )
    if result.equity_curve.empty:
        raise ValueError("Backtest produced no equity points -- check date range vs data availability")

    periods_per_year = PERIODS_PER_YEAR[strategy["rebalance"]]
    initial_capital = strategy["backtest"]["initial_capital"]
    bench_curves = {
        symbol: benchmark_equity_curve(df, result.equity_curve.index, initial_capital)
        for symbol, df in benchmark_prices.items()
    }
    summary = summarize(result.equity_curve, bench_curves.get(primary_benchmark), periods_per_year)

    run.cagr = summary["cagr"]
    run.sharpe = summary["sharpe"]
    run.sortino = summary["sortino"]
    run.calmar = summary["calmar"]
    run.max_drawdown = summary["max_drawdown"]
    run.win_rate = summary["win_rate"]
    run.final_equity = summary["final_equity"]
    run.alpha_vs_primary_benchmark = summary.get("alpha_cagr")

    for date, equity in result.equity_curve.items():
        db.add(EquityPoint(run_id=run.id, date=_date_str(date), equity=float(equity)))
    for symbol, curve in bench_curves.items():
        for date, equity in curve.items():
            db.add(BenchmarkPoint(run_id=run.id, symbol=symbol, date=_date_str(date), equity=float(equity)))

    TEARSHEET_DIR.mkdir(parents=True, exist_ok=True)
    tearsheet_path = TEARSHEET_DIR / f"run_{run.id}.png"
    generate_tearsheet(
        result.equity_curve, bench_curves, result.turnover_history, periods_per_year, tearsheet_path
    )
    run.tearsheet_path = str(tearsheet_path)


def _db_progress_updater(db, run, min_interval_seconds: float = 2.0):
    """A progress_callback for run_walk_forward that persists progress onto
    the Run row, throttled by wall-clock time (a full sweep can be
    thousands of steps; committing on every single one would be a lot of
    SQLite writes for no visible benefit) -- always commits on the final
    step so the row never gets stuck showing a stale non-100% value.
    """
    last_commit = 0.0

    def _callback(done: int, total: int) -> None:
        nonlocal last_commit
        now = time.monotonic()
        if done < total and now - last_commit < min_interval_seconds:
            return
        last_commit = now
        run.progress_current = done
        db.commit()

    return _callback


def _run_walkforward(
    db, run, prices, benchmark_prices, strategy, sector_map, theme_weights, fundamentals, payload
) -> None:
    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    windows = generate_windows(
        all_dates, payload.train_years or 3.0, payload.test_years or 1.0, payload.step_years or 1.0
    )
    if not windows:
        raise ValueError("No windows fit in the given date range")

    run.progress_current = 0
    run.progress_total = total_walkforward_steps(windows, payload.param_grid)
    db.commit()

    results = run_walk_forward(
        prices,
        strategy,
        windows,
        param_grid=payload.param_grid,
        sector_map=sector_map,
        fundamentals=fundamentals,
        benchmark_prices=benchmark_prices,
        theme_weights=theme_weights,
        progress_callback=_db_progress_updater(db, run),
    )

    for i, row in results.iterrows():
        db.add(
            WalkForwardWindow(
                run_id=run.id,
                window_index=int(i),
                train_start=_date_str(row["train_start"]),
                train_end=_date_str(row["train_end"]),
                test_start=_date_str(row["test_start"]),
                test_end=_date_str(row["test_end"]),
                winning_overrides_json=json.dumps(row["winning_overrides"], default=str),
                cagr=_none_if_nan(row.get("cagr")),
                sharpe=_none_if_nan(row.get("sharpe")),
                sortino=_none_if_nan(row.get("sortino")),
                calmar=_none_if_nan(row.get("calmar")),
                max_drawdown=_none_if_nan(row.get("max_drawdown")),
                win_rate=_none_if_nan(row.get("win_rate")),
            )
        )

    # Summary on the Run itself: mean across windows -- a rough headline
    # figure for the run-history list; the per-window detail (with real
    # spread, not just the average) lives in walkforward_windows.
    run.cagr = _none_if_nan(results["cagr"].mean())
    run.sharpe = _none_if_nan(results["sharpe"].mean())
    run.sortino = _none_if_nan(results["sortino"].mean())
    run.calmar = _none_if_nan(results["calmar"].mean())
    run.max_drawdown = _none_if_nan(results["max_drawdown"].mean())
    run.win_rate = _none_if_nan(results["win_rate"].mean())


def _date_str(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _none_if_nan(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
