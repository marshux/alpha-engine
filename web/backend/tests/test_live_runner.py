"""Direct unit tests for app/live_runner.py's execution + persistence logic
-- bypasses the HTTP layer entirely, and monkeypatches
alpha_engine.live.runner.run_live_rebalance so these need no real Alpaca or
Anthropic credentials.
"""
from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha_engine.live.runner import LiveRebalanceResult

from app.database import Base
from app.live_runner import execute_paper_rebalance, get_or_create_paper_trade_run
from app.models import EquityPoint, LiveOrder, Run


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/live_runner_test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_get_or_create_creates_a_new_run(session_factory):
    db = session_factory()
    run = get_or_create_paper_trade_run(db)

    assert run.run_type == "paper_trade"
    assert run.status == "active"
    assert run.end_date is None
    db.close()


def test_get_or_create_reuses_existing_active_run(session_factory):
    db = session_factory()
    first = get_or_create_paper_trade_run(db)
    second = get_or_create_paper_trade_run(db)

    assert first.id == second.id
    assert db.query(Run).filter_by(run_type="paper_trade").count() == 1
    db.close()


def test_execute_paper_rebalance_persists_equity_and_orders(session_factory, monkeypatch):
    result = LiveRebalanceResult(
        target_weights=pd.Series({"AAPL": 0.5}),
        orders=[{"symbol": "AAPL", "side": "buy", "notional": 500.0, "alpaca_order_id": "abc", "status": "accepted"}],
        account_equity=100_500.0,
        dry_run=False,
    )
    monkeypatch.setattr("app.live_runner.run_live_rebalance", lambda: result)

    execute_paper_rebalance(session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(run_type="paper_trade").one()
    assert run.status == "active"
    assert run.error_message is None
    assert run.final_equity == 100_500.0
    assert db.query(EquityPoint).filter_by(run_id=run.id).count() == 1
    orders = db.query(LiveOrder).filter_by(run_id=run.id).all()
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].alpaca_order_id == "abc"
    db.close()


def test_execute_paper_rebalance_records_error_but_stays_active(session_factory, monkeypatch):
    def _raise():
        raise RuntimeError("Alpaca is down")

    monkeypatch.setattr("app.live_runner.run_live_rebalance", _raise)

    execute_paper_rebalance(session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(run_type="paper_trade").one()
    # A bad week doesn't kill a persistent session the way it would a
    # one-shot backtest -- it should still be here to try again next week.
    assert run.status == "active"
    assert "RuntimeError" in run.error_message
    assert db.query(EquityPoint).filter_by(run_id=run.id).count() == 0
    db.close()


def test_execute_paper_rebalance_skip_is_a_noop(session_factory, monkeypatch):
    result = LiveRebalanceResult(skipped_reason="already ran today")
    monkeypatch.setattr("app.live_runner.run_live_rebalance", lambda: result)

    # Pre-create the session so we can assert nothing changed about it.
    db = session_factory()
    run = get_or_create_paper_trade_run(db)
    run_id = run.id
    db.close()

    execute_paper_rebalance(session_factory=session_factory)

    db = session_factory()
    assert db.query(EquityPoint).filter_by(run_id=run_id).count() == 0
    assert db.query(LiveOrder).filter_by(run_id=run_id).count() == 0
    db.close()
