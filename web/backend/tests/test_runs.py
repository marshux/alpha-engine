from __future__ import annotations

import json
from datetime import datetime

from app.models import BenchmarkPoint, EquityPoint, Run, WalkForwardWindow


def _make_backtest_run(db_session) -> Run:
    run = Run(
        run_type="backtest",
        status="completed",
        created_at=datetime(2024, 1, 1),
        config_json=json.dumps({"signal_weights": {"momentum": 0.3}}),
        universe_json=json.dumps(["AAPL", "MSFT"]),
        start_date="2019-01-01",
        end_date="2024-12-31",
        cagr=0.15,
        sharpe=1.0,
    )
    db_session.add(run)
    db_session.flush()
    db_session.add(EquityPoint(run_id=run.id, date="2019-01-04", equity=100000.0))
    db_session.add(BenchmarkPoint(run_id=run.id, symbol="SPY", date="2019-01-04", equity=100000.0))
    db_session.commit()
    return run


def test_list_runs_requires_auth(client):
    response = client.get("/api/runs")
    assert response.status_code == 401


def test_get_run_requires_auth(client, db_session):
    run = _make_backtest_run(db_session)
    response = client.get(f"/api/runs/{run.id}")
    assert response.status_code == 401


def test_list_runs_returns_seeded_data(authed_client, db_session):
    _make_backtest_run(db_session)
    response = authed_client.get("/api/runs")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_type"] == "backtest"
    assert body[0]["cagr"] == 0.15


def test_get_run_includes_time_series(authed_client, db_session):
    run = _make_backtest_run(db_session)
    response = authed_client.get(f"/api/runs/{run.id}")
    assert response.status_code == 200
    body = response.json()
    assert len(body["equity_points"]) == 1
    assert len(body["benchmark_points"]) == 1
    assert body["equity_points"][0]["equity"] == 100000.0


def test_get_missing_run_is_404(authed_client):
    response = authed_client.get("/api/runs/999")
    assert response.status_code == 404


def test_get_tearsheet_404_when_not_set(authed_client, db_session):
    run = _make_backtest_run(db_session)
    response = authed_client.get(f"/api/runs/{run.id}/tearsheet")
    assert response.status_code == 404


def test_walkforward_run_includes_windows(authed_client, db_session):
    run = Run(
        run_type="walkforward",
        status="completed",
        config_json="{}",
        universe_json="[]",
        start_date="2008-01-01",
        end_date="2024-12-31",
    )
    db_session.add(run)
    db_session.flush()
    db_session.add(
        WalkForwardWindow(
            run_id=run.id,
            window_index=0,
            train_start="2008-01-02",
            train_end="2011-01-02",
            test_start="2011-01-02",
            test_end="2012-01-02",
            winning_overrides_json="{}",
            sharpe=0.5,
        )
    )
    db_session.commit()

    response = authed_client.get(f"/api/runs/{run.id}")
    assert response.status_code == 200
    windows = response.json()["walkforward_windows"]
    assert len(windows) == 1
    assert windows[0]["sharpe"] == 0.5


def test_runs_ordered_newest_first(authed_client, db_session):
    older = Run(
        run_type="backtest", status="completed", created_at=datetime(2024, 1, 1),
        config_json="{}", universe_json="[]", start_date="2019-01-01", end_date="2024-12-31",
    )
    newer = Run(
        run_type="backtest", status="completed", created_at=datetime(2024, 6, 1),
        config_json="{}", universe_json="[]", start_date="2019-01-01", end_date="2024-12-31",
    )
    db_session.add_all([older, newer])
    db_session.commit()

    response = authed_client.get("/api/runs")
    ids = [row["id"] for row in response.json()]
    assert ids == [newer.id, older.id]


def test_create_run_requires_auth(client):
    response = client.post(
        "/api/runs",
        json={"run_type": "backtest", "start_date": "2019-01-01", "end_date": "2024-12-31"},
    )
    assert response.status_code == 401


def test_create_run_persists_pending_row_and_schedules_execution(authed_client, monkeypatch):
    # execute_run is what actually runs the backtest -- mocked here so this
    # test verifies the route's own behavior (row creation, response shape,
    # scheduling) without running a real backtest over HTTP. runner.py has
    # its own direct tests for the execution logic itself.
    calls = []
    monkeypatch.setattr(
        "app.routes.runs.execute_run", lambda run_id, payload: calls.append((run_id, payload))
    )

    response = authed_client.post(
        "/api/runs",
        json={
            "run_type": "backtest",
            "start_date": "2019-01-01",
            "end_date": "2024-12-31",
            "symbols": ["AAPL", "MSFT"],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert body["run_type"] == "backtest"
    assert len(calls) == 1
    run_id, payload = calls[0]
    assert run_id == body["id"]
    assert payload.symbols == ["AAPL", "MSFT"]


def test_create_run_rejects_unknown_run_type(authed_client):
    response = authed_client.post(
        "/api/runs",
        json={"run_type": "not_a_real_type", "start_date": "2019-01-01", "end_date": "2024-12-31"},
    )
    assert response.status_code == 422


def test_default_tuning_grid_requires_auth(client):
    response = client.get("/api/runs/tuning-grid/default")
    assert response.status_code == 401


def test_default_tuning_grid_returns_real_config(authed_client):
    # This is the actual config/tuning_grid.yaml the CLI's --param-grid
    # flag reads -- the "New run" form's grid-search checkbox sends this
    # straight back as param_grid, so it must be real dotted-path -> value
    # candidates, not a placeholder shape.
    response = authed_client.get("/api/runs/tuning-grid/default")
    assert response.status_code == 200
    body = response.json()
    assert "signal_weights" in body
    assert isinstance(body["signal_weights"], list)
    assert "portfolio.top_n" in body
