"""Tests for broker/alpaca.py's order-diffing and submission logic, using
a mocked TradingClient -- no real network calls, no Alpaca account needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pandas as pd
from alpaca.trading.enums import OrderSide

from alpha_engine.broker import alpaca as broker
from alpha_engine.broker.alpaca import Order


def test_compute_orders_sells_position_dropped_from_target():
    # AAPL is currently held but absent from target_weights entirely --
    # this is the "dropped from top_n" / "circuit breaker flattened
    # everything" case. Iterating target_weights alone would miss it.
    positions = {"AAPL": {"qty": 10.0, "avg_entry_price": 100.0, "market_value": 1000.0}}
    target_weights = pd.Series({"MSFT": 0.5})

    orders = broker.compute_orders(target_weights, positions, equity=10_000.0, cash_buffer_pct=0.02)

    by_symbol = {o.symbol: o for o in orders}
    assert by_symbol["AAPL"].side == OrderSide.SELL
    assert by_symbol["AAPL"].notional == 1000.0
    assert by_symbol["MSFT"].side == OrderSide.BUY
    assert by_symbol["MSFT"].notional == 4900.0  # 0.5 * (10000 * 0.98)


def test_compute_orders_skips_position_already_at_target():
    positions = {"MSFT": {"qty": 10.0, "avg_entry_price": 400.0, "market_value": 4900.0}}
    target_weights = pd.Series({"MSFT": 0.5})

    orders = broker.compute_orders(target_weights, positions, equity=10_000.0, cash_buffer_pct=0.02)

    assert orders == []


def test_compute_orders_skips_sub_dollar_deltas():
    positions: dict = {}
    target_weights = pd.Series({"MSFT": 0.0001})  # 0.0001 * 9800 = $0.98

    orders = broker.compute_orders(target_weights, positions, equity=10_000.0, cash_buffer_pct=0.02)

    assert orders == []


def test_compute_orders_empty_target_flattens_everything():
    positions = {
        "AAPL": {"qty": 10.0, "avg_entry_price": 100.0, "market_value": 1000.0},
        "MSFT": {"qty": 5.0, "avg_entry_price": 400.0, "market_value": 2000.0},
    }
    target_weights = pd.Series(dtype=float)  # circuit breaker / regime filter tripped

    orders = broker.compute_orders(target_weights, positions, equity=3_000.0)

    assert {o.symbol for o in orders} == {"AAPL", "MSFT"}
    assert all(o.side == OrderSide.SELL for o in orders)


@dataclass
class _FakeSubmittedOrder:
    id: str
    status: str


def test_submit_orders_sells_before_buys():
    call_order: list[str] = []

    def fake_submit(request):
        call_order.append(request.side.value)
        return _FakeSubmittedOrder(id="abc123", status="accepted")

    client = MagicMock()
    client.submit_order.side_effect = fake_submit

    # Deliberately passed in buy-then-sell order -- submit_orders must
    # still submit the sell first.
    orders = [
        Order(symbol="MSFT", side=OrderSide.BUY, notional=500.0),
        Order(symbol="AAPL", side=OrderSide.SELL, notional=300.0),
    ]

    submitted = broker.submit_orders(client, orders)

    assert call_order == ["sell", "buy"]
    assert [o["symbol"] for o in submitted] == ["AAPL", "MSFT"]
    assert all(o["alpaca_order_id"] == "abc123" for o in submitted)


def test_submit_orders_uses_notional_market_orders():
    captured = []

    def fake_submit(request):
        captured.append(request)
        return _FakeSubmittedOrder(id="1", status="accepted")

    client = MagicMock()
    client.submit_order.side_effect = fake_submit

    broker.submit_orders(client, [Order(symbol="AAPL", side=OrderSide.BUY, notional=123.45)])

    assert captured[0].symbol == "AAPL"
    assert captured[0].notional == 123.45
    assert captured[0].qty is None


def test_flatten_all_calls_close_all_positions_with_cancel_orders():
    client = MagicMock()

    broker.flatten_all(client)

    client.close_all_positions.assert_called_once_with(cancel_orders=True)
