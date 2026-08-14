"""HTTP-level tests for app/routes/strategies.py -- the main /live list
view's backing endpoints. Read-only: no broker calls, no credentials
needed.
"""
from __future__ import annotations

import pytest

from app.models import EquityPoint, Strategy
from app.routes import live as live_routes


def _make_strategy(db_session, **overrides) -> Strategy:
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
    return strategy


def test_list_strategies_requires_auth(client):
    response = client.get("/api/strategies")
    assert response.status_code == 401


def test_list_strategies_empty(authed_client):
    response = authed_client.get("/api/strategies")
    assert response.status_code == 200
    assert response.json() == []


def test_list_strategies_with_no_run_yet_has_null_pnl(authed_client, db_session):
    _make_strategy(db_session, name="a")

    response = authed_client.get("/api/strategies")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] is None
    assert body[0]["equity"] is None
    assert body[0]["total_pnl_usd"] is None
    assert body[0]["total_pnl_pct"] is None


def test_list_strategies_computes_pnl_from_equity_points(authed_client, db_session):
    strategy = _make_strategy(db_session, name="a")
    run = live_routes.get_or_create_strategy_run(db_session, strategy)
    db_session.add(EquityPoint(run_id=run.id, date="2026-08-01", equity=100_000.0))
    db_session.add(EquityPoint(run_id=run.id, date="2026-08-05", equity=110_000.0))
    db_session.commit()

    response = authed_client.get("/api/strategies")

    assert response.status_code == 200
    body = response.json()[0]
    assert body["run_id"] == run.id
    assert body["status"] == "active"
    assert body["equity"] == 110_000.0
    assert body["total_pnl_usd"] == 10_000.0
    assert body["total_pnl_pct"] == pytest.approx(0.1)


def test_list_strategies_keeps_strategies_independent(authed_client, db_session):
    strategy_a = _make_strategy(db_session, name="a")
    strategy_b = _make_strategy(db_session, name="b")
    run_a = live_routes.get_or_create_strategy_run(db_session, strategy_a)
    db_session.add(EquityPoint(run_id=run_a.id, date="2026-08-01", equity=100_000.0))
    db_session.add(EquityPoint(run_id=run_a.id, date="2026-08-05", equity=90_000.0))
    db_session.commit()
    # strategy_b never rebalanced -- no run, no equity points

    response = authed_client.get("/api/strategies")

    assert response.status_code == 200
    by_name = {s["name"]: s for s in response.json()}
    assert by_name["a"]["total_pnl_usd"] == -10_000.0
    assert by_name["b"]["run_id"] is None
    assert by_name["b"]["total_pnl_usd"] is None


def test_get_strategy_detail(authed_client, db_session):
    strategy = _make_strategy(db_session, name="a", display_name="Strategy A", risk_tier="aggressive")

    response = authed_client.get(f"/api/strategies/{strategy.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "a"
    assert body["display_name"] == "Strategy A"
    assert body["risk_tier"] == "aggressive"


def test_get_strategy_unknown_id_returns_404(authed_client):
    response = authed_client.get("/api/strategies/999999")
    assert response.status_code == 404


def test_get_strategy_requires_auth(client, db_session):
    strategy = _make_strategy(db_session)
    response = client.get(f"/api/strategies/{strategy.id}")
    assert response.status_code == 401
