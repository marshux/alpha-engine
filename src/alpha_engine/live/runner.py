"""Live/paper rebalance orchestrator, run once per real-world rebalance
(weekly, matching config/strategy.yaml's `rebalance` cadence -- via the
CLI by hand for now, see cli.py's `live-rebalance` command). Fetches
current market data, computes signals, and runs through the SAME
deterministic decision pipeline backtest/engine.py uses --
decision/scorer.py, decision/portfolio.py, decision/risk.py are reused
completely unchanged -- then turns the resulting target weights into real
orders via broker/alpaca.py.

This is also the first and only place signals/news_ai.py's live AI
sector-sentiment research ever actually gets called (see that module's
docstring for why it can never run inside the backtester). Its output is a
single scalar, not a per-symbol signal -- it modulates the AI-theme
concentration cap via decision/portfolio.py::modulate_theme_cap (a plain
deterministic formula) rather than feeding decision/scorer.py's per-stock
weighted sum. The "AI researches, never decides" boundary holds here the
same way: the AI supplies a sentiment/confidence pair, the cap-scaling math
is ordinary code.

The risk-check control flow below (cooldown -> circuit breaker -> regime
filter -> stop-loss) deliberately mirrors run_backtest's per-date loop in
backtest/engine.py line for line, just evaluated once per call instead of
once per historical date, with `breaker_peak`/`cooldown_remaining`
persisted across calls via live/state.py instead of carried in a Python
loop variable (a live rebalance has no in-memory loop connecting one
week's invocation to the next).

Deliberately does NOT live in web/backend/ -- both the CLI
(`alpha-engine live-rebalance`) and a future web-backend scheduler job
call this same function, sharing the same persisted state
(live/state.py), without the CLI needing to depend on the web backend or
vice versa.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha_engine.broker import alpaca as broker
from alpha_engine.config import load_sector_map, load_strategy, load_themes, load_universe, universe_symbols
from alpha_engine.data.equities import get_universe_prices
from alpha_engine.data.fundamentals import get_universe_fundamentals
from alpha_engine.decision.portfolio import build_portfolio, modulate_theme_cap
from alpha_engine.decision.risk import regime_is_risk_on, stopped_out_positions
from alpha_engine.decision.scorer import combine_scores
from alpha_engine.live.state import STATE_PATH, load_state, save_state
from alpha_engine.signals import ai_theme, mean_reversion, momentum, news_ai, quality, value, volatility

SIGNAL_MODULES = {
    "momentum": momentum,
    "mean_reversion": mean_reversion,
    "volatility": volatility,
}

DEFAULT_CIRCUIT_BREAKER_COOLDOWN_PERIODS = 8
# Plenty of lookback for momentum's ~273-trading-day window plus margin --
# a single live rebalance only needs history up through today, unlike a
# full backtest which needs the whole configured date range.
LOOKBACK_DAYS = 730

logger = logging.getLogger(__name__)


@dataclass
class LiveRebalanceResult:
    skipped_reason: str | None = None  # e.g. "already ran today"
    target_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    orders: list[dict] = field(default_factory=list)
    account_equity: float | None = None
    dry_run: bool = False


def _current_prices(prices: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol, df in prices.items():
        hist = df.loc[:as_of]
        if not hist.empty:
            out[symbol] = float(hist["close"].iloc[-1])
    return out


def run_live_rebalance(
    dry_run: bool = False,
    as_of: pd.Timestamp | None = None,
    broker_client=None,
    state_path: Path | None = None,
) -> LiveRebalanceResult:
    """`broker_client`, if provided: an already-constructed alpaca-py
    `TradingClient` (tests pass a mock here instead of monkeypatching
    module internals). Defaults to `broker.get_client()`, which reads
    ALPACA_API_KEY/ALPACA_SECRET_KEY from the environment.

    `state_path`, if provided: overrides where live/state.py reads/writes
    its JSON state file. Defaults to `live.state.STATE_PATH` (the real
    `data/live_state.json`) -- tests always pass a tmp path here instead,
    the same way web/backend/app/runner.py's `execute_run` takes a
    `session_factory` param so tests never touch real persisted state.
    """
    as_of = as_of or pd.Timestamp.now().normalize()
    today_str = as_of.date().isoformat()
    state_path = state_path or STATE_PATH

    state = load_state(state_path)
    if state.last_rebalance_date == today_str:
        logger.info("Already rebalanced today (%s) -- skipping.", today_str)
        return LiveRebalanceResult(skipped_reason="already ran today")

    strategy = load_strategy()
    universe = load_universe()
    sector_map = load_sector_map()
    theme_weights = load_themes().get("ai", {})
    signal_weights = strategy["signal_weights"]
    portfolio_cfg = strategy["portfolio"]
    risk_cfg = strategy["risk"]
    regime_cfg = strategy.get("regime", {})

    symbols = universe_symbols(universe)
    primary_benchmark = strategy["backtest"]["benchmark"]
    benchmark_symbols = universe.get("benchmarks") or [primary_benchmark]

    end = today_str
    start = (as_of - pd.Timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    prices = get_universe_prices(symbols, start, end)
    if not prices:
        raise ValueError("No price data fetched for any symbol")
    benchmark_prices = get_universe_prices(benchmark_symbols, start, end).get(primary_benchmark)
    fundamentals = get_universe_fundamentals(symbols)

    client = broker_client or broker.get_client()
    account = broker.get_account(client)
    positions = broker.get_positions(client)
    equity = account["equity"]

    cooldown_periods = risk_cfg.get(
        "circuit_breaker_cooldown_periods", DEFAULT_CIRCUIT_BREAKER_COOLDOWN_PERIODS
    )
    breaker_peak = state.breaker_peak if state.breaker_peak is not None else equity

    target_weights = pd.Series(dtype=float)

    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1
        if state.cooldown_remaining == 0:
            breaker_peak = equity  # drawdown clock restarts from here
    else:
        breaker_peak = max(breaker_peak, equity)
        drawdown = equity / breaker_peak - 1.0
        if drawdown <= -risk_cfg["max_drawdown_pct"]:
            state.cooldown_remaining = cooldown_periods
        elif (
            benchmark_prices is not None
            and regime_cfg.get("enabled", False)
            and not regime_is_risk_on(benchmark_prices, as_of, regime_cfg["ma_window_days"])
        ):
            pass  # flatten, no cooldown -- re-evaluated fresh next rebalance
        else:
            entry_prices = {s: p["avg_entry_price"] for s, p in positions.items()}
            current_prices = _current_prices(prices, as_of)
            stopped = stopped_out_positions(entry_prices, current_prices, risk_cfg["stop_loss_pct"])

            signal_table = pd.DataFrame(
                {name: mod.compute(prices, as_of) for name, mod in SIGNAL_MODULES.items()}
            )
            signal_table["value"] = value.compute(prices, fundamentals, as_of)
            signal_table["quality"] = quality.compute(fundamentals, as_of)
            if theme_weights:
                signal_table["ai_theme"] = ai_theme.compute(prices, as_of, theme_weights)

            scores = combine_scores(signal_table, signal_weights)
            if stopped:
                scores = scores.drop(index=[s for s in stopped if s in scores.index])

            sizing_data = None
            if portfolio_cfg["sizing"] == "volatility_scaled":
                sizing_data = volatility.raw_annualized_vol(prices, as_of)
                scores = scores[scores.index.isin(sizing_data.index)]
            elif portfolio_cfg["sizing"] == "min_variance":
                sizing_data = volatility.trailing_returns_matrix(prices, list(prices.keys()), as_of)
                scores = scores[scores.index.isin(sizing_data.columns)]

            # AI sentiment is a portfolio-level risk input (how much room to
            # give the AI-theme tilt), not a per-stock ranking signal -- it
            # modulates max_theme_weight, it doesn't join signal_table.
            effective_portfolio_cfg = portfolio_cfg
            if theme_weights and "max_theme_weight" in portfolio_cfg:
                sentiment = news_ai.compute_ai_sector_sentiment(as_of)
                adjusted_cap = modulate_theme_cap(
                    portfolio_cfg["max_theme_weight"], sentiment["sentiment"], sentiment["confidence"]
                )
                effective_portfolio_cfg = {**portfolio_cfg, "max_theme_weight": adjusted_cap}

            target_weights = build_portfolio(
                scores,
                sizing_data,
                sector_map=sector_map,
                config=effective_portfolio_cfg,
                theme_map=theme_weights,
            )

    state.breaker_peak = breaker_peak

    orders = broker.compute_orders(target_weights, positions, equity)
    if dry_run:
        submitted = [
            {"symbol": o.symbol, "side": o.side.value, "notional": o.notional, "status": "DRY_RUN"}
            for o in orders
        ]
    else:
        submitted = broker.submit_orders(client, orders)

    state.last_rebalance_date = today_str
    save_state(state, state_path)

    return LiveRebalanceResult(
        target_weights=target_weights,
        orders=submitted,
        account_equity=equity,
        dry_run=dry_run,
    )
