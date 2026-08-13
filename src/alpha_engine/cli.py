"""CLI entrypoint. Usage: `alpha-engine backtest --start 2018-01-01 --end 2025-01-01`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from alpha_engine.backtest.engine import PERIODS_PER_YEAR, run_backtest
from alpha_engine.backtest.metrics import benchmark_equity_curve, summarize
from alpha_engine.config import load_strategy, load_universe, universe_symbols
from alpha_engine.data.equities import get_universe_prices


def cmd_backtest(args: argparse.Namespace) -> int:
    universe = load_universe()
    strategy = load_strategy(Path(args.config)) if args.config else load_strategy()

    symbols = args.symbols if args.symbols else universe_symbols(universe)
    benchmark_symbol = strategy["backtest"]["benchmark"]

    print(f"Fetching prices for {len(symbols)} symbols + benchmark {benchmark_symbol}...")
    prices = get_universe_prices(symbols, args.start, args.end)
    if not prices:
        print("No price data fetched for any symbol -- aborting.", file=sys.stderr)
        return 1
    benchmark_prices = get_universe_prices([benchmark_symbol], args.start, args.end).get(
        benchmark_symbol
    )

    result = run_backtest(prices, strategy["signal_weights"], strategy)
    if result.equity_curve.empty:
        print("Backtest produced no equity points -- check date range vs data availability.", file=sys.stderr)
        return 1

    periods_per_year = PERIODS_PER_YEAR[strategy["rebalance"]]
    bench_curve = None
    if benchmark_prices is not None:
        bench_curve = benchmark_equity_curve(
            benchmark_prices, result.equity_curve.index, strategy["backtest"]["initial_capital"]
        )

    summary = summarize(result.equity_curve, bench_curve, periods_per_year)

    print(f"\n{'Metric':<20}{'Strategy':>15}{'Benchmark':>15}")
    print(
        f"{'CAGR':<20}{summary['cagr']:>14.2%} "
        f"{summary.get('benchmark_cagr', float('nan')):>14.2%}"
    )
    print(
        f"{'Max Drawdown':<20}{summary['max_drawdown']:>14.2%} "
        f"{summary.get('benchmark_max_drawdown', float('nan')):>14.2%}"
    )
    print(f"{'Sharpe':<20}{summary['sharpe']:>14.2f}")
    print(f"{'Final Equity':<20}${summary['final_equity']:>13,.0f}")
    if "alpha_cagr" in summary:
        print(f"{'Alpha (CAGR)':<20}{summary['alpha_cagr']:>14.2%}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "equity_curve.csv"
    combined = pd.DataFrame({"strategy": result.equity_curve})
    if bench_curve is not None:
        combined["benchmark"] = bench_curve
    combined.to_csv(out_path)
    print(f"\nEquity curve written to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alpha-engine")
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="Run a backtest over a date range")
    bt.add_argument("--start", required=True)
    bt.add_argument("--end", required=True)
    bt.add_argument("--config", default=None, help="Path to strategy.yaml (default: config/strategy.yaml)")
    bt.add_argument("--symbols", nargs="*", default=None, help="Override the configured universe")
    bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
