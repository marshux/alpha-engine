"""Direct unit tests for app/runner.py's execution + persistence logic --
bypasses the HTTP layer entirely (that's covered in test_runs.py, with
execute_run mocked there). Price/fundamentals fetching is monkeypatched to
synthetic data so these tests need no network access and are fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import EquityPoint, Run, WalkForwardWindow
from app.runner import execute_run
from app.schemas import RunCreateRequest


def _series_df(closes: np.ndarray, n_days: int, start: str = "2019-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_days)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1_000_000},
        index=dates,
    )


def _fake_get_universe_prices(symbols, start, end):
    rng = np.random.default_rng(1)
    n = 320
    return {s: _series_df(100 * np.cumprod(1 + rng.normal(0.001, 0.01, n)), n) for s in symbols}


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/runner_test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _make_pending_run(session_factory, **kwargs) -> int:
    db = session_factory()
    run = Run(
        run_type=kwargs.get("run_type", "backtest"),
        status="pending",
        config_json="{}",
        universe_json="[]",
        start_date=kwargs.get("start_date", "2019-01-01"),
        end_date=kwargs.get("end_date", "2020-06-01"),
    )
    db.add(run)
    db.commit()
    run_id = run.id
    db.close()
    return run_id


def test_execute_run_persists_successful_backtest(monkeypatch, session_factory, tmp_path):
    monkeypatch.setattr("app.runner.get_universe_prices", _fake_get_universe_prices)
    monkeypatch.setattr("app.runner.TEARSHEET_DIR", tmp_path / "tearsheets")

    run_id = _make_pending_run(session_factory)
    payload = RunCreateRequest(
        run_type="backtest", start_date="2019-01-01", end_date="2020-06-01", symbols=["AAPL", "MSFT"]
    )

    execute_run(run_id, payload, session_factory=session_factory)

    db = session_factory()
    run = db.get(Run, run_id)
    assert run.status == "completed"
    assert run.error_message is None
    assert run.cagr is not None
    assert run.sharpe is not None
    assert run.tearsheet_path is not None
    import os

    assert os.path.exists(run.tearsheet_path)
    points = db.query(EquityPoint).filter_by(run_id=run_id).all()
    assert len(points) > 0


def test_execute_run_records_failure_on_exception(monkeypatch, session_factory):
    monkeypatch.setattr("app.runner.get_universe_prices", lambda symbols, start, end: {})

    run_id = _make_pending_run(session_factory)
    payload = RunCreateRequest(
        run_type="backtest", start_date="2019-01-01", end_date="2020-06-01", symbols=["AAPL"]
    )

    execute_run(run_id, payload, session_factory=session_factory)

    db = session_factory()
    run = db.get(Run, run_id)
    assert run.status == "failed"
    assert run.error_message is not None
    assert "No price data fetched" in run.error_message


def test_execute_run_persists_walkforward_windows(monkeypatch, session_factory, tmp_path):
    monkeypatch.setattr("app.runner.get_universe_prices", _fake_get_universe_prices)
    monkeypatch.setattr("app.runner.TEARSHEET_DIR", tmp_path / "tearsheets")

    run_id = _make_pending_run(session_factory, run_type="walkforward")
    payload = RunCreateRequest(
        run_type="walkforward",
        start_date="2019-01-01",
        end_date="2020-06-01",
        symbols=["AAPL", "MSFT"],
        train_years=0.75,
        test_years=0.25,
        step_years=0.25,
    )

    execute_run(run_id, payload, session_factory=session_factory)

    db = session_factory()
    run = db.get(Run, run_id)
    assert run.status == "completed"
    windows = db.query(WalkForwardWindow).filter_by(run_id=run_id).all()
    assert len(windows) > 0
    assert run.sharpe is not None


def test_execute_run_unknown_run_type_fails_gracefully(monkeypatch, session_factory):
    monkeypatch.setattr("app.runner.get_universe_prices", _fake_get_universe_prices)

    run_id = _make_pending_run(session_factory)
    payload = RunCreateRequest(
        run_type="not_real", start_date="2019-01-01", end_date="2020-06-01", symbols=["AAPL"]
    )

    execute_run(run_id, payload, session_factory=session_factory)

    db = session_factory()
    run = db.get(Run, run_id)
    assert run.status == "failed"
    assert "unknown run_type" in run.error_message
