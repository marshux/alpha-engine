"""CLI entrypoint. Usage: `alpha-engine backtest --start 2018-01-01 --end 2025-01-01`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from alpha_engine.backtest.engine import PERIODS_PER_YEAR, run_backtest
from alpha_engine.backtest.metrics import benchmark_equity_curve, cagr, max_drawdown, summarize
from alpha_engine.backtest.report import generate_tearsheet
from alpha_engine.backtest.walkforward import generate_windows, run_walk_forward
from alpha_engine.config import load_sector_map, load_strategy, load_universe, universe_symbols
from alpha_engine.data.equities import get_universe_prices
from alpha_engine.data.fundamentals import get_universe_fundamentals


def cmd_backtest(args: argparse.Namespace) -> int:
    universe = load_universe()
    strategy = load_strategy(Path(args.config)) if args.config else load_strategy()
    sector_map = load_sector_map()

    symbols = args.symbols if args.symbols else universe_symbols(universe)
    # Compare against every configured benchmark (SPY/VOO/VTI), not just the
    # one strategy.yaml names as "primary" -- they're highly correlated but
    # not identical, and it's cheap to show all three.
    benchmark_symbols = universe.get("benchmarks") or [strategy["backtest"]["benchmark"]]
    primary_benchmark = strategy["backtest"]["benchmark"]

    print(f"Fetching prices for {len(symbols)} symbols + benchmarks {', '.join(benchmark_symbols)}...")
    prices = get_universe_prices(symbols, args.start, args.end)
    if not prices:
        print("No price data fetched for any symbol -- aborting.", file=sys.stderr)
        return 1
    benchmark_prices = get_universe_prices(benchmark_symbols, args.start, args.end)

    fundamentals = None
    if args.fundamentals:
        print(f"Fetching fundamentals for {len(symbols)} symbols (coverage will be thin for older dates)...")
        fundamentals = get_universe_fundamentals(symbols)

    result = run_backtest(
        prices, strategy["signal_weights"], strategy, sector_map=sector_map, fundamentals=fundamentals
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


def cmd_walkforward(args: argparse.Namespace) -> int:
    universe = load_universe()
    strategy = load_strategy(Path(args.config)) if args.config else load_strategy()
    sector_map = load_sector_map()

    symbols = args.symbols if args.symbols else universe_symbols(universe)
    print(f"Fetching prices for {len(symbols)} symbols...")
    prices = get_universe_prices(symbols, args.start, args.end)
    if not prices:
        print("No price data fetched for any symbol -- aborting.", file=sys.stderr)
        return 1

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

    print(f"Running walk-forward validation over {len(windows)} window(s)...")
    results = run_walk_forward(prices, strategy, windows, param_grid=param_grid, sector_map=sector_map)

    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(f"\n{results.to_string(index=False)}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "walkforward.csv"
    results.to_csv(out_path, index=False)
    print(f"\nWalk-forward results written to {out_path}")
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
    wf.set_defaults(func=cmd_walkforward)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
