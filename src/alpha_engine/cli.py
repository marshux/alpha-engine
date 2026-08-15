"""CLI entrypoint. Usage: `alpha-engine backtest --start 2018-01-01 --end 2025-01-01`."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from alpha_engine.backtest.engine import PERIODS_PER_YEAR, run_backtest
from alpha_engine.backtest.metrics import benchmark_equity_curve, cagr, max_drawdown, summarize
from alpha_engine.backtest.report import generate_tearsheet
from alpha_engine.backtest.walkforward import generate_windows, run_walk_forward, total_walkforward_steps
from alpha_engine.config import load_sector_map, load_strategy, load_themes, load_universe, universe_symbols
from alpha_engine.data.equities import get_universe_prices
from alpha_engine.data.fundamentals import get_universe_fundamentals
from alpha_engine.data.macro import get_macro_data


def cmd_backtest(args: argparse.Namespace) -> int:
    universe = load_universe()
    strategy = load_strategy(Path(args.config)) if args.config else load_strategy()
    sector_map = load_sector_map()
    theme_weights = load_themes().get("ai", {})

    symbols = args.symbols if args.symbols else universe_symbols(universe)
    # Compare against every configured benchmark (SPY/VOO/VTI), not just the
    # one strategy.yaml names as "primary" -- they're highly correlated but
    # not identical, and it's cheap to show all three.
    benchmark_symbols = universe.get("benchmarks") or [strategy["backtest"]["benchmark"]]
    primary_benchmark = strategy["backtest"]["benchmark"]

    print(f"Fetching prices for {len(symbols)} symbols + benchmarks {', '.join(benchmark_symbols)}...", flush=True)
    prices = get_universe_prices(symbols, args.start, args.end)
    if not prices:
        print("No price data fetched for any symbol -- aborting.", file=sys.stderr)
        return 1
    benchmark_prices = get_universe_prices(benchmark_symbols, args.start, args.end)

    fundamentals = None
    if args.fundamentals:
        print(f"Fetching fundamentals for {len(symbols)} symbols (coverage will be thin for older dates)...", flush=True)
        fundamentals = get_universe_fundamentals(symbols)

    # Only fetched when actually needed (regime.mode == "hmm" and/or a
    # --ml-model is given) -- a plain backtest with neither shouldn't pay
    # for a FRED network round-trip it has no use for.
    macro_data = None
    if strategy.get("regime", {}).get("mode") == "hmm" or args.ml_model:
        print("Fetching macro data (FRED)...", flush=True)
        macro_data = get_macro_data()

    ml_model = None
    if args.ml_model:
        # Deferred: only reached when --ml-model is actually passed, so a
        # plain backtest-only install never needs the `ml` extra
        # (lightgbm) installed just to run `alpha-engine backtest`.
        from alpha_engine.ml.alpha_model import AlphaModel

        ml_model = AlphaModel.load(args.ml_model)

    result = run_backtest(
        prices,
        strategy["signal_weights"],
        strategy,
        sector_map=sector_map,
        fundamentals=fundamentals,
        benchmark_prices=benchmark_prices.get(primary_benchmark),
        theme_weights=theme_weights,
        macro_data=macro_data,
        ml_model=ml_model,
    )
    if result.equity_curve.empty:
        print("Backtest produced no equity points -- check date range vs data availability.", file=sys.stderr)
        return 1

    periods_per_year = PERIODS_PER_YEAR[strategy["rebalance"]]
    initial_capital = strategy["backtest"]["initial_capital"]
    bench_curves = {
        symbol: benchmark_equity_curve(df, result.equity_curve.index, initial_capital)
        for symbol, df in benchmark_prices.items()
    }

    summary = summarize(result.equity_curve, bench_curves.get(primary_benchmark), periods_per_year)

    print(f"\n{'Metric':<20}{'Strategy':>15}")
    print(f"{'CAGR':<20}{summary['cagr']:>14.2%}")
    print(f"{'Max Drawdown':<20}{summary['max_drawdown']:>14.2%}")
    print(f"{'Sharpe':<20}{summary['sharpe']:>14.2f}")
    print(f"{'Sortino':<20}{summary['sortino']:>14.2f}")
    print(f"{'Calmar':<20}{summary['calmar']:>14.2f}")
    print(f"{'Win Rate':<20}{summary['win_rate']:>14.2%}")
    print(f"{'Final Equity':<20}${summary['final_equity']:>13,.0f}")

    if bench_curves:
        print(f"\n{'Benchmark':<12}{'CAGR':>12}{'Max DD':>12}{'Alpha (CAGR)':>16}")
        for symbol, curve in bench_curves.items():
            bench_cagr = cagr(curve)
            print(
                f"{symbol:<12}{bench_cagr:>12.2%}"
                f"{max_drawdown(curve):>12.2%}"
                f"{summary['cagr'] - bench_cagr:>16.2%}"
            )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "equity_curve.csv"
    combined = pd.DataFrame({"strategy": result.equity_curve})
    for symbol, curve in bench_curves.items():
        combined[symbol] = curve
    combined.to_csv(out_path)
    print(f"\nEquity curve written to {out_path}")

    tearsheet_path = output_dir / "tearsheet.png"
    generate_tearsheet(
        result.equity_curve, bench_curves, result.turnover_history, periods_per_year, tearsheet_path
    )
    print(f"Tear sheet written to {tearsheet_path}")
    return 0


def _cli_progress_printer(min_interval_seconds: float = 5.0) -> Callable[[int, int], None]:
    """A progress_callback for run_walk_forward that prints one line per
    update, throttled to at most one line every `min_interval_seconds` (plus
    always the first and last step) -- a multi-hour grid sweep would
    otherwise either print nothing until it finishes (the original problem:
    no way to tell if it's stuck or just slow) or spam a line per backtest,
    which is just as useless when there are thousands of steps. Always
    prints a fresh line (not \\r-overwritten) so this is equally readable
    live in a terminal and after the fact via `tail`/`cat` on a redirected
    log file.
    """
    start_time = time.monotonic()
    last_print = 0.0

    def _callback(done: int, total: int) -> None:
        nonlocal last_print
        now = time.monotonic()
        if done < total and now - last_print < min_interval_seconds:
            return
        last_print = now
        elapsed = now - start_time
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else float("nan")
        eta = f", ~{remaining / 60:.1f} min remaining" if rate > 0 else ""
        # flush=True: stdout is block-buffered (not line-buffered) once
        # redirected to a file/pipe (e.g. `nohup ... > log.txt`), so without
        # this these lines would sit invisible in Python's internal buffer
        # for a long time rather than actually showing up in the log.
        print(f"[walkforward] {done}/{total} backtests done ({100 * done / total:.1f}%){eta}", flush=True)

    return _callback


def cmd_walkforward(args: argparse.Namespace) -> int:
    universe = load_universe()
    strategy = load_strategy(Path(args.config)) if args.config else load_strategy()
    sector_map = load_sector_map()
    theme_weights = load_themes().get("ai", {})

    symbols = args.symbols if args.symbols else universe_symbols(universe)
    primary_benchmark = strategy["backtest"]["benchmark"]
    print(f"Fetching prices for {len(symbols)} symbols + benchmark {primary_benchmark}...", flush=True)
    prices = get_universe_prices(symbols, args.start, args.end)
    if not prices:
        print("No price data fetched for any symbol -- aborting.", file=sys.stderr)
        return 1
    benchmark_prices = get_universe_prices([primary_benchmark], args.start, args.end).get(primary_benchmark)

    fundamentals = None
    if args.fundamentals:
        print(f"Fetching fundamentals for {len(symbols)} symbols (coverage will be thin for older dates)...", flush=True)
        fundamentals = get_universe_fundamentals(symbols)

    # Only fetched when actually needed -- same reasoning as cmd_backtest.
    macro_data = None
    if strategy.get("regime", {}).get("mode") == "hmm" or args.train_ml_alpha:
        print("Fetching macro data (FRED)...", flush=True)
        macro_data = get_macro_data()

    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    windows = generate_windows(all_dates, args.train_years, args.test_years, args.step_years)
    if not windows:
        print(
            "No windows fit in the given date range -- shorten --train-years/--test-years/"
            "--step-years or widen --start/--end.",
            file=sys.stderr,
        )
        return 1

    param_grid = None
    if args.param_grid:
        with open(args.param_grid) as f:
            param_grid = yaml.safe_load(f)

    steps_total = total_walkforward_steps(windows, param_grid)
    combos = steps_total // len(windows) - 1
    print(f"Running walk-forward validation over {len(windows)} window(s), {combos} config(s) each...", flush=True)
    if args.train_ml_alpha:
        print("(fitting a fresh ml_alpha model per window -- requires the `ml` extra)", flush=True)
    results = run_walk_forward(
        prices,
        strategy,
        windows,
        param_grid=param_grid,
        sector_map=sector_map,
        fundamentals=fundamentals,
        benchmark_prices=benchmark_prices,
        theme_weights=theme_weights,
        macro_data=macro_data,
        train_ml_alpha=args.train_ml_alpha,
        progress_callback=_cli_progress_printer(),
    )

    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(f"\n{results.to_string(index=False)}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "walkforward.csv"
    results.to_csv(out_path, index=False)
    print(f"\nWalk-forward results written to {out_path}")
    return 0


def cmd_train_ml_alpha(args: argparse.Namespace) -> int:
    # Imported here, not at module top -- alpha_engine.ml.alpha_model
    # depends on the optional `ml` extra (lightgbm), which a plain
    # backtest-only install shouldn't be forced to pull in just to import
    # cli.py at all (same reasoning as cmd_live_rebalance's deferred
    # import of the `live` extra).
    from alpha_engine.ml.alpha_model import AlphaModel
    from alpha_engine.ml.training import build_training_set

    universe = load_universe()
    strategy = load_strategy(Path(args.config)) if args.config else load_strategy()
    symbols = args.symbols if args.symbols else universe_symbols(universe)
    primary_benchmark = strategy["backtest"]["benchmark"]

    print(f"Fetching prices for {len(symbols)} symbols + benchmark {primary_benchmark}...", flush=True)
    prices = get_universe_prices(symbols, args.start, args.end)
    if not prices:
        print("No price data fetched for any symbol -- aborting.", file=sys.stderr)
        return 1
    benchmark_prices = get_universe_prices([primary_benchmark], args.start, args.end).get(primary_benchmark)
    if benchmark_prices is None:
        print(f"No price data fetched for benchmark {primary_benchmark} -- aborting.", file=sys.stderr)
        return 1

    print(f"Fetching fundamentals for {len(symbols)} symbols...", flush=True)
    fundamentals = get_universe_fundamentals(symbols)

    print("Fetching macro data (FRED)...", flush=True)
    macro_data = get_macro_data()

    train_start, train_end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    print(f"Building training set ({train_start.date()} to {train_end.date()})...", flush=True)
    X, y = build_training_set(
        prices,
        fundamentals,
        macro_data,
        benchmark_prices,
        symbols=symbols,
        train_start=train_start,
        train_end=train_end,
        cadence=strategy["rebalance"],
        forward_days=args.forward_days,
    )
    if X.empty:
        print(
            "No training rows produced -- date range too narrow relative to --forward-days.",
            file=sys.stderr,
        )
        return 1
    print(f"Training on {len(X)} (date, symbol) rows...", flush=True)

    model = AlphaModel().train(X, y, trained_through=str(train_end.date()))

    out_path = Path(args.out)
    model.save(out_path)
    print(f"Model saved to {out_path} (+ {out_path.name}.meta.json)")
    return 0


def cmd_live_rebalance(args: argparse.Namespace) -> int:
    # Imported here, not at module top -- alpha_engine.live depends on the
    # optional `live` extra (alpaca-py), which a backtest-only/CI install
    # shouldn't be forced to pull in just to import cli.py at all.
    from alpha_engine.live.runner import run_live_rebalance

    result = run_live_rebalance(dry_run=args.dry_run)

    if result.skipped_reason:
        print(f"Skipped: {result.skipped_reason}")
        return 0

    print(f"Account equity: ${result.account_equity:,.2f}")
    print(f"Target weights ({len(result.target_weights)} positions):")
    for symbol, weight in result.target_weights.sort_values(ascending=False).items():
        print(f"  {symbol:<8} {weight:.2%}")

    label = "Would submit" if result.dry_run else "Submitted"
    if not result.orders:
        print(f"{label}: no orders (already at target allocation)")
    else:
        print(f"{label} {len(result.orders)} order(s):")
        for order in result.orders:
            print(f"  {order['side']:<4} {order['symbol']:<8} ${order['notional']:>10,.2f}  [{order['status']}]")
    return 0


def cmd_anomaly_check(args: argparse.Namespace) -> int:
    # Imported here for the same reason as cmd_live_rebalance -- keeps the
    # optional `live` extra (alpaca-py) out of a backtest-only install.
    from alpha_engine.live.anomaly_guard import run_anomaly_check

    result = run_anomaly_check(dry_run=args.dry_run)

    if not result.orders:
        print("No anomalous drops detected -- no positions closed.")
        return 0

    label = "Would close" if result.dry_run else "Closed"
    print(f"{label} {len(result.orders)} position(s):")
    for order in result.orders:
        print(f"  {order['symbol']:<8} ${order['notional']:>10,.2f}  [{order['status']}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alpha-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Run a backtest over a date range")
    bt.add_argument("--start", required=True)
    bt.add_argument("--end", required=True)
    bt.add_argument("--config", default=None, help="Path to strategy.yaml (default: config/strategy.yaml)")
    bt.add_argument("--symbols", nargs="*", default=None, help="Override the configured universe")
    bt.add_argument(
        "--fundamentals",
        action="store_true",
        help="Fetch fundamentals and include value/quality signals (slower; thin coverage on older dates)",
    )
    bt.add_argument(
        "--ml-model",
        default=None,
        help="Path to a trained AlphaModel (see `train-ml-alpha`) -- adds the ml_alpha signal (requires the `ml` extra)",
    )
    bt.set_defaults(func=cmd_backtest)

    wf = sub.add_parser("walkforward", help="Rolling train/test window validation")
    wf.add_argument("--start", required=True)
    wf.add_argument("--end", required=True)
    wf.add_argument("--config", default=None, help="Path to strategy.yaml (default: config/strategy.yaml)")
    wf.add_argument("--symbols", nargs="*", default=None, help="Override the configured universe")
    wf.add_argument("--train-years", type=float, default=3.0)
    wf.add_argument("--test-years", type=float, default=1.0)
    wf.add_argument("--step-years", type=float, default=1.0)
    wf.add_argument(
        "--param-grid",
        default=None,
        help="Path to a YAML file of dotted-config-path -> list of candidate values to grid-search on each train window",
    )
    wf.add_argument(
        "--fundamentals",
        action="store_true",
        help="Fetch fundamentals and include value/quality signals (slower; thin coverage on older dates)",
    )
    wf.add_argument(
        "--train-ml-alpha",
        action="store_true",
        help="Fit a fresh ml_alpha model per window on that window's train range only, walk-forward-safe (requires the `ml` extra)",
    )
    wf.set_defaults(func=cmd_walkforward)

    train_ml = sub.add_parser(
        "train-ml-alpha",
        help="Fit a LightGBM ml_alpha model on historical data and save it to disk (requires the `ml` extra)",
    )
    train_ml.add_argument("--start", required=True)
    train_ml.add_argument("--end", required=True)
    train_ml.add_argument("--config", default=None, help="Path to strategy.yaml (default: config/strategy.yaml)")
    train_ml.add_argument("--symbols", nargs="*", default=None, help="Override the configured universe")
    train_ml.add_argument(
        "--forward-days", type=int, default=21, help="Forward-return horizon used to label training rows"
    )
    train_ml.add_argument(
        "--out", default="data/models/ml_alpha.txt", help="Output path for the trained model"
    )
    train_ml.set_defaults(func=cmd_train_ml_alpha)

    live = sub.add_parser(
        "live-rebalance", help="Run one live/paper rebalance against Alpaca (requires the `live` extra)"
    )
    live.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute target weights and orders but do not submit them",
    )
    live.set_defaults(func=cmd_live_rebalance)

    anomaly = sub.add_parser(
        "anomaly-check",
        help="Check currently-held positions for an abnormal single-day drop and force-close any flagged (requires the `live` extra)",
    )
    anomaly.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be closed but do not submit orders",
    )
    anomaly.set_defaults(func=cmd_anomaly_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
