from __future__ import annotations

from datetime import date, timedelta

from app.models import BenchmarkPoint, EquityPoint, Run
from app.routes.public import PUBLIC_DASHBOARD_DELAY_DAYS


def _make_run(db_session, *, is_public: bool, status: str = "completed", run_type: str = "backtest") -> Run:
    run = Run(
        run_type=run_type,
        status=status,
        is_public=is_public,
        config_json='{"signal_weights": {"momentum": 0.9}}',
        universe_json='["AAPL"]',
        start_date="2019-01-01",
        end_date="2024-12-31",
        cagr=0.12,
        sharpe=0.9,
    )
    db_session.add(run)
    db_session.flush()

    old_date = (date.today() - timedelta(days=PUBLIC_DASHBOARD_DELAY_DAYS + 10)).isoformat()
    recent_date = (date.today() - timedelta(days=1)).isoformat()
    db_session.add(EquityPoint(run_id=run.id, date=old_date, equity=100000.0))
    db_session.add(EquityPoint(run_id=run.id, date=recent_date, equity=120000.0))
    db_session.add(BenchmarkPoint(run_id=run.id, symbol="SPY", date=old_date, equity=100000.0))
    db_session.add(BenchmarkPoint(run_id=run.id, symbol="SPY", date=recent_date, equity=110000.0))
    db_session.commit()
    return run


def test_public_list_excludes_non_public_runs(client, db_session):
    _make_run(db_session, is_public=False)
    response = client.get("/api/public/runs")
    assert response.status_code == 200
    assert response.json() == []


def test_public_list_needs_no_auth(client, db_session):
    _make_run(db_session, is_public=True)
    response = client.get("/api/public/runs")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_public_list_excludes_incomplete_runs(client, db_session):
    _make_run(db_session, is_public=True, status="running")
    response = client.get("/api/public/runs")
    assert response.json() == []


def test_public_detail_404s_for_non_public_run(client, db_session):
    run = _make_run(db_session, is_public=False)
    response = client.get(f"/api/public/runs/{run.id}")
    assert response.status_code == 404


def test_public_detail_never_includes_config_or_universe(client, db_session):
    run = _make_run(db_session, is_public=True)
    response = client.get(f"/api/public/runs/{run.id}")
    body = response.json()
    assert "config_json" not in body
    assert "universe_json" not in body
    assert "tearsheet_path" not in body
    assert "walkforward_windows" not in body


def test_public_detail_redacts_recent_equity_points(client, db_session):
    run = _make_run(db_session, is_public=True)
    response = client.get(f"/api/public/runs/{run.id}")
    body = response.json()
    # Two EquityPoints were seeded: one old (past the delay window), one
    # from yesterday -- only the old one should survive.
    assert len(body["equity_points"]) == 1
    assert len(body["benchmark_points"]) == 1
    assert body["equity_points"][0]["equity"] == 100000.0


def test_patch_run_visibility_requires_auth(client, db_session):
    run = _make_run(db_session, is_public=False)
    response = client.patch(f"/api/runs/{run.id}", json={"is_public": True})
    assert response.status_code == 401


def test_patch_run_visibility_toggles_flag(authed_client, db_session):
    run = _make_run(db_session, is_public=False)
    response = authed_client.patch(f"/api/runs/{run.id}", json={"is_public": True})
    assert response.status_code == 200
    assert response.json()["is_public"] is True

    # And it's now visible on the public list.
    public_response = authed_client.get("/api/public/runs")
    assert len(public_response.json()) == 1


def test_public_list_excludes_paper_trade_runs_even_if_public_and_completed(client, db_session):
    # Defense in depth: even a hypothetically mis-set paper_trade row
    # (is_public=True, status="completed" -- status alone wouldn't
    # normally exclude it if that combination ever existed) must never
    # appear on the public dashboard. Positions/order timing must never
    # leak, regardless of how the flags got set.
    _make_run(db_session, is_public=True, status="completed", run_type="paper_trade")
    response = client.get("/api/public/runs")
    assert response.json() == []


def test_public_detail_404s_for_paper_trade_run_even_if_public_and_completed(client, db_session):
    run = _make_run(db_session, is_public=True, status="completed", run_type="paper_trade")
    response = client.get(f"/api/public/runs/{run.id}")
    assert response.status_code == 404
