"""Direct unit tests for app/live_runner.py's execution + persistence logic
-- bypasses the HTTP layer entirely, and monkeypatches
alpha_engine.live.runner.run_live_rebalance so these need no real Alpaca or
Anthropic credentials. `_broker_client` is stubbed out everywhere too, so
no test needs real ALPACA_*_ENV environment variables set.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from alpha_engine.live.anomaly_guard import AnomalyCheckResult
from alpha_engine.live.runner import LiveRebalanceResult

from app.database import Base
from app.live_runner import (
    execute_anomaly_check,
    execute_paper_rebalance,
    get_or_create_strategy_run,
    record_daily_equity_snapshot,
)
from app.models import AiUsageEvent, EquityPoint, LiveOrder, Run, Strategy


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/live_runner_test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _strategy_kwargs(**overrides) -> dict:
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
    return defaults


def _make_strategy_id(session_factory, **overrides) -> int:
    """Creates a Strategy row in its own session and returns the plain id
    -- matches how execute_paper_rebalance/etc. take strategy_id: int, not
    an ORM object (see app/live_runner.py's module docstring for why)."""
    db = session_factory()
    strategy = Strategy(**_strategy_kwargs(**overrides))
    db.add(strategy)
    db.commit()
    strategy_id = strategy.id
    db.close()
    return strategy_id


def _patch_broker_client(monkeypatch):
    monkeypatch.setattr("app.live_runner._broker_client", lambda strategy: object())


def test_get_or_create_creates_a_new_run(session_factory):
    db = session_factory()
    strategy = Strategy(**_strategy_kwargs())
    db.add(strategy)
    db.commit()

    run = get_or_create_strategy_run(db, strategy)

    assert run.run_type == "paper_trade"
    assert run.strategy_id == strategy.id
    assert run.status == "active"
    assert run.end_date is None
    db.close()


def test_get_or_create_reuses_existing_active_run(session_factory):
    db = session_factory()
    strategy = Strategy(**_strategy_kwargs())
    db.add(strategy)
    db.commit()

    first = get_or_create_strategy_run(db, strategy)
    second = get_or_create_strategy_run(db, strategy)

    assert first.id == second.id
    assert db.query(Run).filter_by(strategy_id=strategy.id).count() == 1
    db.close()


def test_get_or_create_does_not_reuse_another_strategys_run(session_factory):
    db = session_factory()
    strategy_a = Strategy(**_strategy_kwargs(name="a"))
    strategy_b = Strategy(**_strategy_kwargs(name="b"))
    db.add_all([strategy_a, strategy_b])
    db.commit()

    run_a = get_or_create_strategy_run(db, strategy_a)
    run_b = get_or_create_strategy_run(db, strategy_b)

    assert run_a.id != run_b.id
    assert run_a.strategy_id == strategy_a.id
    assert run_b.strategy_id == strategy_b.id
    db.close()


def test_execute_paper_rebalance_persists_equity_and_orders(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    _patch_broker_client(monkeypatch)
    result = LiveRebalanceResult(
        target_weights=pd.Series({"AAPL": 0.5}),
        orders=[{"symbol": "AAPL", "side": "buy", "notional": 500.0, "alpaca_order_id": "abc", "status": "accepted"}],
        account_equity=100_500.0,
        dry_run=False,
    )
    monkeypatch.setattr("app.live_runner.run_live_rebalance", lambda **kwargs: result)

    execute_paper_rebalance(strategy_id, session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    assert run.status == "active"
    assert run.error_message is None
    assert run.final_equity == 100_500.0
    assert db.query(EquityPoint).filter_by(run_id=run.id).count() == 1
    orders = db.query(LiveOrder).filter_by(run_id=run.id).all()
    assert len(orders) == 1
    assert orders[0].symbol == "AAPL"
    assert orders[0].alpaca_order_id == "abc"
    assert orders[0].trigger == "rebalance"
    assert db.query(AiUsageEvent).count() == 0  # no ai_usage on this result -- nothing to record
    db.close()


def test_execute_paper_rebalance_persists_ai_usage_event(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    _patch_broker_client(monkeypatch)
    result = LiveRebalanceResult(
        target_weights=pd.Series({"AAPL": 0.5}),
        orders=[],
        account_equity=100_500.0,
        dry_run=False,
        ai_usage={
            "input_tokens": 1200,
            "output_tokens": 300,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "web_search_requests": 2,
        },
    )
    monkeypatch.setattr("app.live_runner.run_live_rebalance", lambda **kwargs: result)

    execute_paper_rebalance(strategy_id, session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    events = db.query(AiUsageEvent).filter_by(run_id=run.id).all()
    assert len(events) == 1
    assert events[0].source == "news_ai_sentiment"
    assert events[0].input_tokens == 1200
    assert events[0].output_tokens == 300
    assert events[0].web_search_requests == 2
    db.close()


def test_execute_paper_rebalance_passes_strategy_config_through(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(
        session_factory,
        config_overrides_json='{"portfolio.top_n": 4}',
        universe_symbols_json='["XLK", "XLF"]',
    )
    _patch_broker_client(monkeypatch)

    captured = {}

    def _fake_run_live_rebalance(**kwargs):
        captured.update(kwargs)
        return LiveRebalanceResult(account_equity=100_000.0, dry_run=False)

    monkeypatch.setattr("app.live_runner.run_live_rebalance", _fake_run_live_rebalance)

    execute_paper_rebalance(strategy_id, session_factory=session_factory)

    assert captured["strategy_overrides"] == {"portfolio.top_n": 4}
    assert captured["symbols_override"] == ["XLK", "XLF"]


def test_execute_anomaly_check_persists_orders_tagged_anomaly(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    _patch_broker_client(monkeypatch)
    result = AnomalyCheckResult(
        flagged=["TSLA"],
        orders=[{"symbol": "TSLA", "side": "sell", "notional": 850.0, "alpaca_order_id": "xyz", "status": "accepted"}],
        dry_run=False,
    )
    monkeypatch.setattr("app.live_runner.run_anomaly_check", lambda **kwargs: result)

    execute_anomaly_check(strategy_id, session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    assert run.status == "active"
    assert run.error_message is None
    orders = db.query(LiveOrder).filter_by(run_id=run.id).all()
    assert len(orders) == 1
    assert orders[0].symbol == "TSLA"
    assert orders[0].trigger == "anomaly"
    assert orders[0].notional == 850.0
    # An anomaly check doesn't snapshot equity -- that's the weekly
    # rebalance's job, not a per-position forced sell.
    assert db.query(EquityPoint).filter_by(run_id=run.id).count() == 0
    db.close()


def test_execute_anomaly_check_records_error_but_stays_active(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    _patch_broker_client(monkeypatch)

    def _raise(**kwargs):
        raise RuntimeError("Alpaca is down")

    monkeypatch.setattr("app.live_runner.run_anomaly_check", _raise)

    execute_anomaly_check(strategy_id, session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    assert run.status == "active"
    assert "RuntimeError" in run.error_message
    db.close()


def test_execute_anomaly_check_noop_when_nothing_flagged(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    _patch_broker_client(monkeypatch)
    result = AnomalyCheckResult(flagged=[], orders=[], dry_run=False)
    monkeypatch.setattr("app.live_runner.run_anomaly_check", lambda **kwargs: result)

    execute_anomaly_check(strategy_id, session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    assert db.query(LiveOrder).filter_by(run_id=run.id).count() == 0
    db.close()


def test_execute_paper_rebalance_records_error_but_stays_active(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    _patch_broker_client(monkeypatch)

    def _raise(**kwargs):
        raise RuntimeError("Alpaca is down")

    monkeypatch.setattr("app.live_runner.run_live_rebalance", _raise)

    execute_paper_rebalance(strategy_id, session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    # A bad week doesn't kill a persistent session the way it would a
    # one-shot backtest -- it should still be here to try again next week.
    assert run.status == "active"
    assert "RuntimeError" in run.error_message
    assert db.query(EquityPoint).filter_by(run_id=run.id).count() == 0
    db.close()


def test_execute_paper_rebalance_skip_is_a_noop(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    _patch_broker_client(monkeypatch)
    result = LiveRebalanceResult(skipped_reason="already ran today")
    monkeypatch.setattr("app.live_runner.run_live_rebalance", lambda **kwargs: result)

    execute_paper_rebalance(strategy_id, session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    assert db.query(EquityPoint).filter_by(run_id=run.id).count() == 0
    assert db.query(LiveOrder).filter_by(run_id=run.id).count() == 0
    db.close()


def test_record_daily_equity_snapshot_creates_a_point(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    _patch_broker_client(monkeypatch)
    monkeypatch.setattr("app.live_runner.broker.get_account", lambda client: {"equity": 101_234.5})

    record_daily_equity_snapshot(strategy_id, session_factory=session_factory)

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    points = db.query(EquityPoint).filter_by(run_id=run.id).all()
    assert len(points) == 1
    assert points[0].date == date.today().isoformat()
    assert points[0].equity == 101_234.5
    db.close()


def test_record_daily_equity_snapshot_is_a_noop_if_todays_point_exists(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)
    db = session_factory()
    strategy = db.get(Strategy, strategy_id)
    run = get_or_create_strategy_run(db, strategy)
    run_id = run.id
    db.add(EquityPoint(run_id=run_id, date=date.today().isoformat(), equity=99_000.0))
    db.commit()
    db.close()

    def _fail_if_called(strategy):
        raise AssertionError("_broker_client must not be called when today's point already exists")

    monkeypatch.setattr("app.live_runner._broker_client", _fail_if_called)

    record_daily_equity_snapshot(strategy_id, session_factory=session_factory)

    db = session_factory()
    points = db.query(EquityPoint).filter_by(run_id=run_id).all()
    assert len(points) == 1
    assert points[0].equity == 99_000.0  # untouched
    db.close()


def test_record_daily_equity_snapshot_swallows_broker_errors(session_factory, monkeypatch):
    strategy_id = _make_strategy_id(session_factory)

    def _raise(strategy):
        raise RuntimeError("Alpaca is down")

    monkeypatch.setattr("app.live_runner._broker_client", _raise)

    record_daily_equity_snapshot(strategy_id, session_factory=session_factory)  # must not raise

    db = session_factory()
    run = db.query(Run).filter_by(strategy_id=strategy_id).one()
    assert run.error_message is None  # reserved for trading failures, not this
    assert db.query(EquityPoint).filter_by(run_id=run.id).count() == 0
    db.close()
