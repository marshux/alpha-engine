"""HTTP-level tests for app/routes/live.py -- broker calls are monkeypatched
(no real Alpaca account needed), and execute_paper_rebalance is monkeypatched
in the rebalance-now test since it's dispatched as a background task.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.models import AiUsageEvent
from app.routes import live as live_routes


def test_positions_requires_auth(client):
    response = client.get("/api/live/positions")
    assert response.status_code == 401


def test_positions_returns_live_account_and_positions(authed_client, monkeypatch):
    monkeypatch.setattr(live_routes.broker, "get_client", lambda: MagicMock())
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

    response = authed_client.get("/api/live/positions")

    assert response.status_code == 200
    body = response.json()
    assert body["equity"] == 100_000.0
    assert body["positions"] == [
        {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 150.0, "market_value": 1500.0}
    ]


def test_positions_missing_credentials_returns_503(authed_client, monkeypatch):
    def _raise():
        raise KeyError("ALPACA_API_KEY")

    monkeypatch.setattr(live_routes.broker, "get_client", _raise)

    response = authed_client.get("/api/live/positions")

    assert response.status_code == 503


def test_start_creates_active_paper_trade_run(authed_client):
    response = authed_client.post("/api/live/start")

    assert response.status_code == 201
    body = response.json()
    assert body["run_type"] == "paper_trade"
    assert body["status"] == "active"


def test_start_is_idempotent(authed_client):
    first = authed_client.post("/api/live/start").json()
    second = authed_client.post("/api/live/start").json()

    assert first["id"] == second["id"]


def test_rebalance_now_schedules_execution(authed_client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.routes.live.execute_paper_rebalance", lambda: calls.append("executed")
    )

    response = authed_client.post("/api/live/rebalance-now")

    assert response.status_code == 200
    assert response.json()["run_type"] == "paper_trade"
    assert calls == ["executed"]  # TestClient runs background tasks synchronously


def test_anomaly_check_now_schedules_execution(authed_client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "app.routes.live.execute_anomaly_check", lambda: calls.append("executed")
    )

    response = authed_client.post("/api/live/anomaly-check-now")

    assert response.status_code == 200
    assert response.json()["run_type"] == "paper_trade"
    assert calls == ["executed"]  # TestClient runs background tasks synchronously


def test_ai_usage_requires_auth(client):
    response = client.get("/api/live/ai-usage")
    assert response.status_code == 401


def test_ai_usage_aggregates_events(authed_client, db_session):
    db_session.add(
        AiUsageEvent(
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

    response = authed_client.get("/api/live/ai-usage")

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


def test_ai_usage_empty_returns_zeros(authed_client):
    response = authed_client.get("/api/live/ai-usage")

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


def test_flatten_requires_auth(client):
    response = client.post("/api/live/flatten")
    assert response.status_code == 401


def test_flatten_calls_broker(authed_client, monkeypatch):
    flatten_calls = []
    monkeypatch.setattr(live_routes.broker, "get_client", lambda: MagicMock())
    monkeypatch.setattr(live_routes.broker, "flatten_all", lambda client: flatten_calls.append(client))

    response = authed_client.post("/api/live/flatten")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(flatten_calls) == 1
