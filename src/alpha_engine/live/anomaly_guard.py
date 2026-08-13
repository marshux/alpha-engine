"""Daily per-position anomaly check, entirely independent of the weekly
rebalance cadence: flags any currently-held position whose latest daily
move is a volatility-adjusted outlier for that specific stock (see
decision/risk.py::anomalous_drop_positions) and immediately liquidates it
via broker/alpaca.py. This is a faster-acting complement to the
cooldown/circuit-breaker/regime/stop-loss checks in live/runner.py, which
are only evaluated once a week -- a stock can crash mid-week and this
catches it without waiting for the next rebalance.

Deliberately separate from run_live_rebalance(): it only ever touches
currently-held positions (cheap -- a handful of price lookups, not a
full-universe fetch), and it can fire and sell independent of whether
today happens to be a rebalance day. No shared state with live/state.py:
it operates purely off Alpaca's own current positions, so a position sold
here simply won't appear in the next check (or the next rebalance's
position diff) -- naturally idempotent, no separate "already handled"
flag needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from alpha_engine.broker import alpaca as broker
from alpha_engine.config import load_strategy
from alpha_engine.data.equities import get_universe_prices
from alpha_engine.decision.risk import anomalous_drop_positions

DEFAULT_Z_THRESHOLD = 3.0
DEFAULT_LOOKBACK_DAYS = 60
# Calendar-day price-fetch window: comfortably covers DEFAULT_LOOKBACK_DAYS
# trading days (~85 calendar days) plus margin for holidays/weekends.
FETCH_LOOKBACK_DAYS = 120

logger = logging.getLogger(__name__)


@dataclass
class AnomalyCheckResult:
    flagged: list[str] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    dry_run: bool = False


def run_anomaly_check(
    dry_run: bool = False,
    as_of: pd.Timestamp | None = None,
    broker_client=None,
) -> AnomalyCheckResult:
    """`broker_client`/`dry_run` follow the same conventions as
    live/runner.py::run_live_rebalance -- tests pass a mock client instead
    of monkeypatching module internals, `dry_run` computes what would be
    sold without submitting it.
    """
    as_of = as_of or pd.Timestamp.now().normalize()
    strategy = load_strategy()
    risk_cfg = strategy.get("risk", {})
    z_threshold = risk_cfg.get("anomaly_z_threshold", DEFAULT_Z_THRESHOLD)
    lookback_days = risk_cfg.get("anomaly_lookback_days", DEFAULT_LOOKBACK_DAYS)

    client = broker_client or broker.get_client()
    positions = broker.get_positions(client)
    if not positions:
        return AnomalyCheckResult(dry_run=dry_run)

    held_symbols = set(positions.keys())
    start = (as_of - pd.Timedelta(days=lookback_days + FETCH_LOOKBACK_DAYS)).date().isoformat()
    end = as_of.date().isoformat()
    prices = get_universe_prices(list(held_symbols), start, end)

    flagged = anomalous_drop_positions(
        prices, held_symbols, as_of, z_threshold=z_threshold, lookback_days=lookback_days
    )
    if not flagged:
        return AnomalyCheckResult(dry_run=dry_run)

    logger.warning("Anomalous drop flagged for: %s", sorted(flagged))

    orders = []
    for symbol in sorted(flagged):
        notional = positions[symbol]["market_value"]
        if dry_run:
            orders.append({"symbol": symbol, "side": "sell", "notional": notional, "status": "DRY_RUN"})
            continue
        order = broker.close_position(client, symbol)
        orders.append({**order, "notional": notional})

    return AnomalyCheckResult(flagged=sorted(flagged), orders=orders, dry_run=dry_run)
