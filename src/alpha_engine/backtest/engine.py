"""Walk-forward backtest loop.

At each rebalance date, signals are computed only from price history up to
that date (no lookahead), combined via the deterministic decision engine
(decision/scorer.py, decision/portfolio.py), and the simulated portfolio is
updated.

Positions are tracked as portfolio *weights*, not share counts -- this
keeps the engine simple and avoids share-rounding/cash-drag bookkeeping
that isn't the point of a v1 research backtester. One consequence: risk
checks (circuit breaker, stop-loss) are only evaluated at rebalance
granularity (weekly by default), not intraday/daily. That's a known
limitation of this simplification, not an oversight -- a stop-loss can be
breached and only get acted on at the next rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from alpha_engine.backtest.costs import turnover_cost
from alpha_engine.decision.portfolio import build_portfolio
from alpha_engine.decision.risk import regime_is_risk_on, stopped_out_positions
from alpha_engine.decision.scorer import combine_scores
from alpha_engine.signals import ai_theme, mean_reversion, momentum, quality, value, volatility

SIGNAL_MODULES = {
    "momentum": momentum,
    "mean_reversion": mean_reversion,
    "volatility": volatility,
}

_CADENCE_FREQ = {"weekly": "W-FRI", "biweekly": "2W-FRI", "monthly": "ME"}

PERIODS_PER_YEAR = {"weekly": 52, "biweekly": 26, "monthly": 12}

DEFAULT_CIRCUIT_BREAKER_COOLDOWN_PERIODS = 8


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    weights_history: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    turnover_history: dict[pd.Timestamp, float] = field(default_factory=dict)


def rebalance_dates(all_dates: pd.DatetimeIndex, cadence: str) -> pd.DatetimeIndex:
    """Snap each period-end (per `cadence`) to the nearest available
    trading date on or before it, so rebalances always land on a real
    market day even when the calendar period-end is a weekend/holiday.
    """
    if cadence not in _CADENCE_FREQ:
        raise ValueError(f"unknown cadence: {cadence}")
    period_ends = pd.date_range(all_dates.min(), all_dates.max(), freq=_CADENCE_FREQ[cadence])
    snapped = []
    for d in period_ends:
        available = all_dates[all_dates <= d]
        if len(available):
            snapped.append(available[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def _prices_at(prices: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol, df in prices.items():
        hist = df.loc[:as_of]
        if not hist.empty:
            out[symbol] = float(hist["close"].iloc[-1])
    return out


def run_backtest(
    prices: dict[str, pd.DataFrame],
    signal_weights: dict[str, float],
    strategy_config: dict,
    sector_map: dict[str, str] | None = None,
    fundamentals: dict[str, pd.DataFrame] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    benchmark_prices: pd.DataFrame | None = None,
    theme_weights: dict[str, float] | None = None,
) -> BacktestResult:
    """`sector_map`, if provided: symbol -> sector name (config/sectors.yaml),
    used for the portfolio's sector-exposure cap.

    `fundamentals`, if provided: symbol -> quarterly fundamentals DataFrame
    (data.fundamentals.get_universe_fundamentals) -- when given, adds the
    `value`/`quality` signals to the signal table for this run. Symbols
    without fundamentals data (e.g. ETFs) simply score neutral on these,
    via the same missing-signal handling as everywhere else.

    `start`/`end`, if provided: restrict rebalancing to this sub-range of
    dates. Deliberately does NOT slice `prices` -- signals still see full
    lookback history from before `start` (momentum needs ~273 trading days
    of it). This is what backtest/walkforward.py uses to evaluate a single
    train/test window without starving signals of history at the window's
    start; slicing `prices` per-window would do exactly that.

    `benchmark_prices`, if provided together with `strategy_config["regime"]
    ["enabled"]`: OHLCV for the trend/regime filter (decision.risk.
    regime_is_risk_on) -- when the benchmark is below its own trailing
    moving average, the portfolio holds cash regardless of what individual
    stock signals say, until the benchmark recovers above it again.

    `theme_weights`, if provided: symbol -> AI-exposure intensity
    (config/themes.yaml's `ai` mapping) -- adds the `ai_theme` signal
    (signals/ai_theme.py) to the signal table, and the resulting
    concentration in tagged names is capped by
    `strategy_config["portfolio"]["max_theme_weight"]` if set (see
    decision/portfolio.apply_theme_cap).
    """
    sector_map = sector_map or {}
    regime_cfg = strategy_config.get("regime", {})
    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    dates = rebalance_dates(all_dates, strategy_config["rebalance"])
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates <= end]

    portfolio_cfg = strategy_config["portfolio"]
    risk_cfg = strategy_config["risk"]
    cost_cfg = strategy_config["costs"]

    equity = float(strategy_config["backtest"]["initial_capital"])
    weights: pd.Series = pd.Series(dtype=float)
    entry_prices: dict[str, float] = {}

    equity_points: dict[pd.Timestamp, float] = {}
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    turnover_history: dict[pd.Timestamp, float] = {}

    prev_prices: dict[str, float] = {}
    has_prev = False

    # Circuit-breaker state: `breaker_peak` is the reference high-water mark
    # used for the drawdown check, tracked separately from the full equity
    # curve. This matters once the breaker has fired: while flattened to
    # cash, equity is frozen, so drawdown-from-the-pre-crash-peak would
    # never recover on its own and the strategy would be locked out of the
    # market forever. Instead: stay in cash for a fixed cooldown, then
    # re-enter and reset `breaker_peak` to the post-cooldown equity level,
    # so the drawdown clock restarts from where the strategy actually is
    # rather than an unrecoverable historical high.
    breaker_peak = equity
    cooldown_remaining = 0
    cooldown_periods = risk_cfg.get(
        "circuit_breaker_cooldown_periods", DEFAULT_CIRCUIT_BREAKER_COOLDOWN_PERIODS
    )

    for as_of in dates:
        current_prices = _prices_at(prices, as_of)

        # Accrue return over the period since the last rebalance using the
        # PRIOR period's target weights -- weights were decided last period
        # using only data available then, applied here to this period's
        # realized return. No lookahead.
        if has_prev and not weights.empty:
            period_return = sum(
                w * (current_prices[s] / prev_prices[s] - 1.0)
                for s, w in weights.items()
                if s in current_prices and s in prev_prices
            )
            equity *= 1.0 + period_return

        equity_points[as_of] = equity

        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            weights = pd.Series(dtype=float)
            entry_prices = {}
            weights_history[as_of] = weights
            turnover_history[as_of] = 0.0
            if cooldown_remaining == 0:
                breaker_peak = equity  # drawdown clock restarts from here
            prev_prices, has_prev = current_prices, True
            continue

        breaker_peak = max(breaker_peak, equity)
        drawdown = equity / breaker_peak - 1.0
        if drawdown <= -risk_cfg["max_drawdown_pct"]:
            turnover = weights.abs().sum()
            equity -= turnover_cost(turnover, equity, cost_cfg)
            equity_points[as_of] = equity
            weights = pd.Series(dtype=float)
            entry_prices = {}
            weights_history[as_of] = weights
            turnover_history[as_of] = turnover
            cooldown_remaining = cooldown_periods
            prev_prices, has_prev = current_prices, True
            continue

        if (
            benchmark_prices is not None
            and regime_cfg.get("enabled", False)
            and not regime_is_risk_on(benchmark_prices, as_of, regime_cfg["ma_window_days"])
        ):
            # Unlike the circuit breaker, no cooldown here -- re-evaluated
            # fresh every rebalance, so the portfolio resumes investing the
            # moment the benchmark crosses back above its moving average.
            turnover = weights.abs().sum()
            equity -= turnover_cost(turnover, equity, cost_cfg)
            equity_points[as_of] = equity
            weights = pd.Series(dtype=float)
            entry_prices = {}
            weights_history[as_of] = weights
            turnover_history[as_of] = turnover
            prev_prices, has_prev = current_prices, True
            continue

        stopped = stopped_out_positions(entry_prices, current_prices, risk_cfg["stop_loss_pct"])

        signal_table = pd.DataFrame(
            {name: mod.compute(prices, as_of) for name, mod in SIGNAL_MODULES.items()}
        )
        if fundamentals is not None:
            signal_table["value"] = value.compute(prices, fundamentals, as_of)
            signal_table["quality"] = quality.compute(fundamentals, as_of)
        if theme_weights is not None:
            signal_table["ai_theme"] = ai_theme.compute(prices, as_of, theme_weights)

        scores = combine_scores(signal_table, signal_weights)
        if stopped:
            scores = scores.drop(index=[s for s in stopped if s in scores.index])

        sizing_data = None
        if portfolio_cfg["sizing"] == "volatility_scaled":
            sizing_data = volatility.raw_annualized_vol(prices, as_of)
            # Only candidates with enough trailing history to be sized are
            # eligible for selection at all. Without this, a signal that
            # doesn't need price history itself (e.g. ai_theme, which only
            # checks dict membership) can populate `scores` before any
            # symbol has the ~20 days volatility_scaled sizing needs --
            # size_positions would then correctly refuse to size a
            # selected-but-unsizeable symbol (a real invariant it should
            # keep enforcing), so the filtering has to happen up here,
            # before selection, not by loosening that guard.
            scores = scores[scores.index.isin(sizing_data.index)]
        elif portfolio_cfg["sizing"] == "min_variance":
            sizing_data = volatility.trailing_returns_matrix(prices, list(prices.keys()), as_of)
            scores = scores[scores.index.isin(sizing_data.columns)]

        target_weights = build_portfolio(
            scores, sizing_data, sector_map=sector_map, config=portfolio_cfg, theme_map=theme_weights
        )

        turnover = weights.add(-target_weights, fill_value=0.0).abs().sum()
        equity -= turnover_cost(turnover, equity, cost_cfg)
        equity_points[as_of] = equity
        turnover_history[as_of] = turnover

        for s in target_weights.index:
            if weights.get(s, 0.0) == 0.0:
                entry_prices[s] = current_prices.get(s, entry_prices.get(s, 0.0))
        for s in list(entry_prices.keys()):
            if s not in target_weights.index:
                entry_prices.pop(s, None)

        weights = target_weights
        weights_history[as_of] = weights
        prev_prices, has_prev = current_prices, True

    return BacktestResult(
        equity_curve=pd.Series(equity_points).sort_index(),
        weights_history=weights_history,
        turnover_history=turnover_history,
    )
