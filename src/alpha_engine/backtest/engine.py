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
from alpha_engine.decision.risk import circuit_breaker_triggered, stopped_out_positions
from alpha_engine.decision.scorer import combine_scores
from alpha_engine.signals import mean_reversion, momentum, volatility

SIGNAL_MODULES = {
    "momentum": momentum,
    "mean_reversion": mean_reversion,
    "volatility": volatility,
}

_CADENCE_FREQ = {"weekly": "W-FRI", "biweekly": "2W-FRI", "monthly": "ME"}

PERIODS_PER_YEAR = {"weekly": 52, "biweekly": 26, "monthly": 12}


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    weights_history: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)


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
    news_signal: pd.DataFrame | None = None,
) -> BacktestResult:
    """`news_signal`, if provided: DataFrame indexed by rebalance date with
    one column per symbol, holding precomputed/cached AI news scores
    (already z-scored) -- kept separate from `prices` since it comes from a
    different data source (signals/news_ai.py) fetched independently.
    """
    all_dates = pd.DatetimeIndex(sorted(set().union(*(df.index for df in prices.values()))))
    dates = rebalance_dates(all_dates, strategy_config["rebalance"])

    portfolio_cfg = strategy_config["portfolio"]
    risk_cfg = strategy_config["risk"]
    cost_cfg = strategy_config["costs"]

    equity = float(strategy_config["backtest"]["initial_capital"])
    weights: pd.Series = pd.Series(dtype=float)
    entry_prices: dict[str, float] = {}

    equity_points: dict[pd.Timestamp, float] = {}
    weights_history: dict[pd.Timestamp, pd.Series] = {}

    prev_prices: dict[str, float] = {}
    has_prev = False

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
        curve_so_far = pd.Series(equity_points).sort_index()

        if circuit_breaker_triggered(curve_so_far, risk_cfg["max_drawdown_pct"]):
            turnover = weights.abs().sum()
            equity -= turnover_cost(turnover, equity, cost_cfg)
            weights = pd.Series(dtype=float)
            entry_prices = {}
            equity_points[as_of] = equity
            weights_history[as_of] = weights
            prev_prices, has_prev = current_prices, True
            continue

        stopped = stopped_out_positions(entry_prices, current_prices, risk_cfg["stop_loss_pct"])

        signal_table = pd.DataFrame(
            {name: mod.compute(prices, as_of) for name, mod in SIGNAL_MODULES.items()}
        )
        if news_signal is not None and as_of in news_signal.index:
            signal_table["news_ai"] = news_signal.loc[as_of]

        scores = combine_scores(signal_table, signal_weights)
        if stopped:
            scores = scores.drop(index=[s for s in stopped if s in scores.index])

        vol_scores = None
        if portfolio_cfg["sizing"] == "volatility_scaled":
            vol_scores = volatility.raw_annualized_vol(prices, as_of)

        target_weights = build_portfolio(scores, vol_scores, sector_map={}, config=portfolio_cfg)

        turnover = weights.add(-target_weights, fill_value=0.0).abs().sum()
        equity -= turnover_cost(turnover, equity, cost_cfg)
        equity_points[as_of] = equity

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
    )
