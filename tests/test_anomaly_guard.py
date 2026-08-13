"""Tests for live/anomaly_guard.py's orchestration -- config/price/broker
fetches are all monkeypatched (no network calls, no real Alpaca account).
The anomaly-detection math itself (decision/risk.py::anomalous_drop_positions)
is covered by tests/test_decision.py; these tests patch that function
directly to control which symbols get flagged.
"""
from __future__ import annotations

from alpha_engine.live import anomaly_guard


def _base_strategy_config(**risk_overrides) -> dict:
    return {"risk": {"anomaly_z_threshold": 3.0, "anomaly_lookback_days": 60, **risk_overrides}}


def _patch_config(monkeypatch, risk_overrides=None):
    monkeypatch.setattr(anomaly_guard, "load_strategy", lambda: _base_strategy_config(**(risk_overrides or {})))


def _patch_prices(monkeypatch):
    monkeypatch.setattr(anomaly_guard, "get_universe_prices", lambda symbols, start, end: {})


def test_no_positions_returns_empty_result_without_fetching_prices(monkeypatch):
    _patch_config(monkeypatch)

    def _fail_if_called(*a, **k):
        raise AssertionError("get_universe_prices must not be called with no held positions")

    monkeypatch.setattr(anomaly_guard, "get_universe_prices", _fail_if_called)
    monkeypatch.setattr(anomaly_guard.broker, "get_positions", lambda client: {})

    result = anomaly_guard.run_anomaly_check(broker_client=object())

    assert result.flagged == []
    assert result.orders == []


def test_flagged_position_is_closed_and_notional_reflects_market_value(monkeypatch):
    _patch_config(monkeypatch)
    _patch_prices(monkeypatch)

    monkeypatch.setattr(
        anomaly_guard.broker,
        "get_positions",
        lambda client: {
            "AAPL": {"qty": 10.0, "avg_entry_price": 100.0, "market_value": 850.0},
            "MSFT": {"qty": 5.0, "avg_entry_price": 300.0, "market_value": 1500.0},
        },
    )
    monkeypatch.setattr(
        anomaly_guard, "anomalous_drop_positions", lambda prices, symbols, as_of, z_threshold, lookback_days: {"AAPL"}
    )

    closed = []
    monkeypatch.setattr(
        anomaly_guard.broker,
        "close_position",
        lambda client, symbol: closed.append(symbol)
        or {"symbol": symbol, "side": "sell", "alpaca_order_id": "1", "status": "accepted"},
    )

    result = anomaly_guard.run_anomaly_check(broker_client=object())

    assert closed == ["AAPL"]  # MSFT untouched -- not flagged
    assert result.flagged == ["AAPL"]
    assert result.orders == [
        {"symbol": "AAPL", "side": "sell", "alpaca_order_id": "1", "status": "accepted", "notional": 850.0}
    ]


def test_no_flagged_positions_closes_nothing(monkeypatch):
    _patch_config(monkeypatch)
    _patch_prices(monkeypatch)
    monkeypatch.setattr(
        anomaly_guard.broker,
        "get_positions",
        lambda client: {"AAPL": {"qty": 10.0, "avg_entry_price": 100.0, "market_value": 850.0}},
    )
    monkeypatch.setattr(anomaly_guard, "anomalous_drop_positions", lambda *a, **k: set())

    def _fail_if_called(*a, **k):
        raise AssertionError("close_position must not be called when nothing is flagged")

    monkeypatch.setattr(anomaly_guard.broker, "close_position", _fail_if_called)

    result = anomaly_guard.run_anomaly_check(broker_client=object())

    assert result.flagged == []
    assert result.orders == []


def test_dry_run_never_closes_positions(monkeypatch):
    _patch_config(monkeypatch)
    _patch_prices(monkeypatch)
    monkeypatch.setattr(
        anomaly_guard.broker,
        "get_positions",
        lambda client: {"AAPL": {"qty": 10.0, "avg_entry_price": 100.0, "market_value": 850.0}},
    )
    monkeypatch.setattr(anomaly_guard, "anomalous_drop_positions", lambda *a, **k: {"AAPL"})

    def _fail_if_called(*a, **k):
        raise AssertionError("close_position must never be called in dry-run mode")

    monkeypatch.setattr(anomaly_guard.broker, "close_position", _fail_if_called)

    result = anomaly_guard.run_anomaly_check(dry_run=True, broker_client=object())

    assert result.dry_run is True
    assert result.flagged == ["AAPL"]
    assert result.orders == [{"symbol": "AAPL", "side": "sell", "notional": 850.0, "status": "DRY_RUN"}]


def test_strategy_config_thresholds_are_passed_through(monkeypatch):
    _patch_config(monkeypatch, risk_overrides={"anomaly_z_threshold": 2.5, "anomaly_lookback_days": 30})
    _patch_prices(monkeypatch)
    monkeypatch.setattr(
        anomaly_guard.broker,
        "get_positions",
        lambda client: {"AAPL": {"qty": 10.0, "avg_entry_price": 100.0, "market_value": 850.0}},
    )

    captured = {}

    def _capture(prices, symbols, as_of, z_threshold, lookback_days):
        captured["z_threshold"] = z_threshold
        captured["lookback_days"] = lookback_days
        return set()

    monkeypatch.setattr(anomaly_guard, "anomalous_drop_positions", _capture)

    anomaly_guard.run_anomaly_check(broker_client=object())

    assert captured == {"z_threshold": 2.5, "lookback_days": 30}
