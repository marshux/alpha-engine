"""Thin wrapper around alpaca-py's TradingClient for paper-trading order
execution -- fetching account/position state and turning target weights
(from decision/portfolio.py) into real orders.

`paper=True` is hardcoded in `_client()`, not a runtime toggle. Going live
with real money later is meant to be a deliberate code change, not a
config flip, given the stakes.

Only ever called by live/runner.py -- never by the backtester, same
isolation as signals/news_ai.py (see that module's docstring for why
live-only code must not leak into the backtest path).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

DEFAULT_CASH_BUFFER_PCT = 0.02
# Alpaca rejects notional orders below $1; skip anything smaller than this
# rather than round it up into a trade the rebalance never intended.
MIN_ORDER_NOTIONAL = 1.0


@dataclass
class Order:
    symbol: str
    side: OrderSide
    notional: float  # dollar amount, always positive


def get_client(api_key: str | None = None, secret_key: str | None = None) -> TradingClient:
    return TradingClient(
        api_key=api_key or os.environ["ALPACA_API_KEY"],
        secret_key=secret_key or os.environ["ALPACA_SECRET_KEY"],
        paper=True,
    )


def get_account(client: TradingClient) -> dict:
    account = client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
    }


def get_positions(client: TradingClient) -> dict[str, dict]:
    """symbol -> {qty, avg_entry_price, market_value}. `avg_entry_price` is
    Alpaca's own record of each position's cost basis -- used directly as
    the stop-loss's entry-price source (decision/risk.py's
    stopped_out_positions), so nothing needs to separately persist entry
    prices locally.
    """
    positions = client.get_all_positions()
    return {
        p.symbol: {
            "qty": float(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
        }
        for p in positions
    }


def compute_orders(
    target_weights: pd.Series,
    positions: dict[str, dict],
    equity: float,
    cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT,
) -> list[Order]:
    """Diffs target weights against currently-held positions to produce the
    orders needed to reach the target allocation.

    Unions current-position symbols with `target_weights.index` -- a
    symbol held but absent from `target_weights` gets an implicit target
    of 0, which is what actually generates the sell order when a position
    is dropped from top_n, or when every position needs to close because
    a risk check (circuit breaker, regime filter) flattened the portfolio
    entirely and `target_weights` is empty. Iterating `target_weights`
    alone would silently leave those positions open.

    Returns notional (dollar-denominated) orders rather than computing
    qty = dollars / price and rounding -- sidesteps fractional-share
    rounding entirely for any fractionable symbol (true of essentially
    every symbol in config/universe.yaml: large-caps and sector ETFs).
    `cash_buffer_pct` reserves a small slice of equity undeployed --
    intentionally implemented here, not in decision/portfolio.py, so the
    backtested decision layer's weights stay untouched by this
    live-execution-only concern.
    """
    symbols = set(positions.keys()) | set(target_weights.index)
    deployable = equity * (1.0 - cash_buffer_pct)

    orders = []
    for symbol in sorted(symbols):
        target_dollars = float(target_weights.get(symbol, 0.0)) * deployable
        current_dollars = positions.get(symbol, {}).get("market_value", 0.0)
        delta = target_dollars - current_dollars
        if abs(delta) < MIN_ORDER_NOTIONAL:
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        orders.append(Order(symbol=symbol, side=side, notional=round(abs(delta), 2)))
    return orders


def submit_orders(client: TradingClient, orders: list[Order]) -> list[dict]:
    """Submits sells/reduces first, then buys/increases -- otherwise a buy
    can be spuriously rejected for insufficient buying power even when the
    net target allocation is within 100% of equity, since freed-up cash
    from a same-batch sell isn't reflected until that sell is processed.
    Does not wait for sells to *fill* (only to be accepted) before
    submitting buys -- a known v1 simplification, consistent with treating
    order submission as the audit record rather than building a full
    fill-reconciliation state machine (see live/runner.py's docstring).
    """
    sells = [o for o in orders if o.side == OrderSide.SELL]
    buys = [o for o in orders if o.side == OrderSide.BUY]

    submitted = []
    for order in sells + buys:
        request = MarketOrderRequest(
            symbol=order.symbol,
            notional=order.notional,
            side=order.side,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(request)
        submitted.append(
            {
                "symbol": order.symbol,
                "side": order.side.value,
                "notional": order.notional,
                "alpaca_order_id": str(result.id),
                "status": str(result.status),
            }
        )
    return submitted


def flatten_all(client: TradingClient) -> None:
    """Liquidates every position and cancels every open order immediately --
    a manual safety action (e.g. the dashboard's "flatten to cash" button),
    not something the deterministic rebalance logic calls itself. Uses
    alpaca-py's own bulk close endpoint rather than compute_orders/
    submit_orders, since this is an unconditional "get to zero now", not a
    diff against a target allocation.
    """
    client.close_all_positions(cancel_orders=True)
