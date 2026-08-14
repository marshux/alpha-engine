"""HTTP-level tests for app/routes/live.py -- broker calls are monkeypatched
(no real Alpaca account needed), and execute_paper_rebalance is monkeypatched
in the rebalance-now test since it's dispatched as a background task. Every
route is strategy-scoped (/api/live/{strategy_id}/...), so each test first
creates a Strategy row and sets its (fake) credential env vars.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.models import AiUsageEvent, EquityPoint, Strategy
from app.routes import live as live_routes


def _make_strategy(db_session, monkeypatch, **overrides) -> Strategy:
    defaults = dict(
        name="test_strategy",
        display_name="Test Strategy",
        description="A strategy used only in tests.",
        risk_tier="traditional",
        config_overrides_json="{}",
        universe_symbols_json=None,
        alpaca_key_env="ALPACA_API_KEY_TEST",
        alpaca_secret_env="ALPACA_SECRET_KEY_TEST",
        enabled=True,
    )
    defaults.update(overrides)
    strategy = Strategy(**defaults)
    db_session.add(strategy)
    db_session.commit()
    monkeypatch.setenv(strategy.alpaca_key_env, "fake-key")
    monkeypatch.setenv(strategy.alpaca_secret_env, "fake-secret")
    return strategy


def test_positions_requires_auth(client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    response = client.get(f"/api/live/{strategy.id}/positions")
    assert response.status_code == 401


def test_positions_returns_live_account_and_positions(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    monkeypatch.setattr(live_routes.broker, "get_client", lambda **kwargs: MagicMock())
    monkeypatch.setattr(
        live_routes.broker,
        "get_account",
        lambda client: {"equity": 100_000.0, "cash": 50_000.0, "buying_power": 200_000.0},
    )
    monkeypatch.setattr(
        live_routes.broker,
        "get_positions",
        lambda client: {"AAPL": {"qty": 10.0, "avg_entry_price": 150.0, "market_value": 1500.0}},
    )

    response = authed_client.get(f"/api/live/{strategy.id}/positions")

    assert response.status_code == 200
    body = response.json()
    assert body["equity"] == 100_000.0
    assert body["positions"] == [
        {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 150.0, "market_value": 1500.0}
    ]


def test_positions_unknown_strategy_returns_404(authed_client):
    response = authed_client.get("/api/live/999999/positions")
    assert response.status_code == 404


def test_positions_missing_credentials_returns_503(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    monkeypatch.delenv(strategy.alpaca_key_env, raising=False)

    response = authed_client.get(f"/api/live/{strategy.id}/positions")

    assert response.status_code == 503


def test_start_creates_active_paper_trade_run(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)

    response = authed_client.post(f"/api/live/{strategy.id}/start")

    assert response.status_code == 201
    body = response.json()
    assert body["run_type"] == "paper_trade"
    assert body["status"] == "active"


def test_start_is_idempotent(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)

    first = authed_client.post(f"/api/live/{strategy.id}/start").json()
    second = authed_client.post(f"/api/live/{strategy.id}/start").json()

    assert first["id"] == second["id"]


def test_rebalance_now_schedules_execution(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    calls = []
    monkeypatch.setattr(
        "app.routes.live.execute_paper_rebalance", lambda strategy_id: calls.append(strategy_id)
    )

    response = authed_client.post(f"/api/live/{strategy.id}/rebalance-now")

    assert response.status_code == 200
    assert response.json()["run_type"] == "paper_trade"
    assert calls == [strategy.id]  # TestClient runs background tasks synchronously


def test_anomaly_check_now_schedules_execution(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    calls = []
    monkeypatch.setattr(
        "app.routes.live.execute_anomaly_check", lambda strategy_id: calls.append(strategy_id)
    )

    response = authed_client.post(f"/api/live/{strategy.id}/anomaly-check-now")

    assert response.status_code == 200
    assert response.json()["run_type"] == "paper_trade"
    assert calls == [strategy.id]  # TestClient runs background tasks synchronously


def test_ai_usage_requires_auth(client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    response = client.get(f"/api/live/{strategy.id}/ai-usage")
    assert response.status_code == 401


def test_ai_usage_aggregates_events(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    run = live_routes.get_or_create_strategy_run(db_session, strategy)
    db_session.add(
        AiUsageEvent(
            run_id=run.id,
            date="2026-08-10",
            source="news_ai_sentiment",
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            web_search_requests=1,
        )
    )
    db_session.add(
        AiUsageEvent(
            run_id=run.id,
            date="2026-08-11",
            source="news_ai_sentiment",
            input_tokens=2000,
            output_tokens=400,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            web_search_requests=1,
        )
    )
    db_session.commit()

    response = authed_client.get(f"/api/live/{strategy.id}/ai-usage")

    assert response.status_code == 200
    body = response.json()
    assert body["total_calls"] == 2
    assert body["total_input_tokens"] == 3000
    assert body["total_output_tokens"] == 600
    assert body["total_web_search_requests"] == 2
    assert body["total_estimated_cost_usd"] > 0
    assert len(body["events"]) == 2
    # Newest first
    assert body["events"][0]["date"] == "2026-08-11"


def test_ai_usage_does_not_include_another_strategys_events(authed_client, db_session, monkeypatch):
    strategy_a = _make_strategy(db_session, monkeypatch, name="a")
    strategy_b = _make_strategy(db_session, monkeypatch, name="b")
    run_a = live_routes.get_or_create_strategy_run(db_session, strategy_a)
    live_routes.get_or_create_strategy_run(db_session, strategy_b)
    db_session.add(
        AiUsageEvent(
            run_id=run_a.id,
            date="2026-08-10",
            source="news_ai_sentiment",
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            web_search_requests=1,
        )
    )
    db_session.commit()

    response = authed_client.get(f"/api/live/{strategy_b.id}/ai-usage")

    assert response.status_code == 200
    assert response.json()["total_calls"] == 0


def test_ai_usage_empty_returns_zeros(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)

    response = authed_client.get(f"/api/live/{strategy.id}/ai-usage")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "total_calls": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_web_search_requests": 0,
        "total_estimated_cost_usd": 0.0,
        "events": [],
    }


def test_benchmark_pnl_requires_auth(client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    response = client.get(f"/api/live/{strategy.id}/benchmark-pnl")
    assert response.status_code == 401


def test_benchmark_pnl_with_no_equity_points_returns_nulls(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)

    response = authed_client.get(f"/api/live/{strategy.id}/benchmark-pnl")

    assert response.status_code == 200
    assert response.json() == {"baseline_date": None, "baseline_equity": None, "benchmarks": []}


def test_benchmark_pnl_computed_from_same_baseline_as_strategy_pnl(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    run = live_routes.get_or_create_strategy_run(db_session, strategy)
    db_session.add(EquityPoint(run_id=run.id, date="2026-08-01", equity=100_000.0))
    db_session.commit()

    def fake_prices(symbols, start, end):
        assert start == "2026-08-01"
        # SPY up 10%, VOO up 5% -- distinct per symbol so the test can tell
        # them apart in the response.
        closes = {"SPY": [400.0, 440.0], "VOO": [500.0, 525.0], "VTI": [300.0, 300.0]}
        return {
            symbol: pd.DataFrame({"close": closes[symbol]}, index=pd.to_datetime(["2026-08-01", "2026-08-13"]))
            for symbol in symbols
        }

    monkeypatch.setattr(live_routes, "get_universe_prices", fake_prices)
    monkeypatch.setattr(live_routes, "load_universe", lambda: {"benchmarks": ["SPY", "VOO", "VTI"]})

    response = authed_client.get(f"/api/live/{strategy.id}/benchmark-pnl")

    assert response.status_code == 200
    body = response.json()
    assert body["baseline_date"] == "2026-08-01"
    assert body["baseline_equity"] == 100_000.0
    by_symbol = {b["symbol"]: b for b in body["benchmarks"]}
    assert by_symbol["SPY"]["pnl_pct"] == pytest.approx(0.10)
    assert by_symbol["SPY"]["pnl_usd"] == pytest.approx(10_000.0)
    assert by_symbol["VOO"]["pnl_pct"] == pytest.approx(0.05)
    assert by_symbol["VOO"]["pnl_usd"] == pytest.approx(5_000.0)
    assert by_symbol["VTI"]["pnl_pct"] == pytest.approx(0.0)


def test_benchmark_pnl_widens_fetch_range_when_baseline_is_today(authed_client, db_session, monkeypatch):
    # Regression: yfinance's `end` is exclusive, so a session started
    # today (baseline.date == today) would fetch a zero-width start==end
    # range and silently return no benchmarks at all if the route passed
    # today's date straight through as `end`.
    import datetime as dt

    today_str = dt.date.today().isoformat()
    strategy = _make_strategy(db_session, monkeypatch)
    run = live_routes.get_or_create_strategy_run(db_session, strategy)
    db_session.add(EquityPoint(run_id=run.id, date=today_str, equity=100_000.0))
    db_session.commit()

    def fake_prices(symbols, start, end):
        assert start == today_str
        assert end != today_str  # widened past the exclusive-end trap
        return {
            symbol: pd.DataFrame({"close": [400.0, 404.0]}, index=pd.to_datetime([today_str, today_str]))
            for symbol in symbols
        }

    monkeypatch.setattr(live_routes, "get_universe_prices", fake_prices)
    monkeypatch.setattr(live_routes, "load_universe", lambda: {"benchmarks": ["SPY"]})

    response = authed_client.get(f"/api/live/{strategy.id}/benchmark-pnl")

    assert response.status_code == 200
    assert len(response.json()["benchmarks"]) == 1


def test_flatten_requires_auth(client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    response = client.post(f"/api/live/{strategy.id}/flatten")
    assert response.status_code == 401


def test_flatten_calls_broker(authed_client, db_session, monkeypatch):
    strategy = _make_strategy(db_session, monkeypatch)
    flatten_calls = []
    monkeypatch.setattr(live_routes.broker, "get_client", lambda **kwargs: MagicMock())
    monkeypatch.setattr(live_routes.broker, "flatten_all", lambda client: flatten_calls.append(client))

    response = authed_client.post(f"/api/live/{strategy.id}/flatten")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(flatten_calls) == 1
