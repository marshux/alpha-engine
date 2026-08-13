"""Tests for the scheduled job functions (app/jobs/tasks.py) -- called
directly, not through APScheduler, with data fetch and execution
monkeypatched so these need no network access and don't run a real
backtest.
"""
from __future__ import annotations

from app.jobs import scheduler as scheduler_module
from app.jobs import tasks
from app.jobs.scheduler import scheduler, start_scheduler, stop_scheduler


def test_nightly_cache_refresh_fetches_prices_and_fundamentals(monkeypatch):
    price_calls = []
    fundamentals_calls = []
    monkeypatch.setattr(
        tasks, "get_universe_prices", lambda symbols, start, end: price_calls.append((symbols, start, end))
    )
    monkeypatch.setattr(tasks, "get_universe_fundamentals", lambda symbols: fundamentals_calls.append(symbols))

    tasks.nightly_cache_refresh()

    assert len(price_calls) == 1
    symbols, start, end = price_calls[0]
    assert len(symbols) > 0
    assert start < end
    assert len(fundamentals_calls) == 1


def test_weekly_auto_backtest_creates_run_and_executes(monkeypatch, tmp_path):
    monkeypatch.setattr(tasks, "SessionLocal", _sqlite_session_factory(tmp_path))

    calls = []
    monkeypatch.setattr(tasks, "execute_run", lambda run_id, payload: calls.append((run_id, payload)))

    tasks.weekly_auto_backtest()

    assert len(calls) == 1
    run_id, payload = calls[0]
    assert payload.run_type == "backtest"
    assert payload.fundamentals is True


def test_weekly_paper_rebalance_calls_execute_paper_rebalance(monkeypatch):
    calls = []
    monkeypatch.setattr(tasks, "execute_paper_rebalance", lambda: calls.append("executed"))

    tasks.weekly_paper_rebalance()

    assert calls == ["executed"]


def test_daily_anomaly_check_calls_execute_anomaly_check(monkeypatch):
    calls = []
    monkeypatch.setattr(tasks, "execute_anomaly_check", lambda: calls.append("executed"))

    tasks.daily_anomaly_check()

    assert calls == ["executed"]


def test_daily_equity_snapshot_calls_record_daily_equity_snapshot(monkeypatch):
    calls = []
    monkeypatch.setattr(tasks, "record_daily_equity_snapshot", lambda: calls.append("executed"))

    tasks.daily_equity_snapshot()

    assert calls == ["executed"]


def _sqlite_session_factory(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base

    engine = create_engine(f"sqlite:///{tmp_path}/jobs_test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_scheduler_registers_core_jobs_without_live_trading(monkeypatch):
    monkeypatch.setattr(scheduler_module, "ENABLE_LIVE_TRADING", False)
    start_scheduler()
    try:
        job_ids = {job.id for job in scheduler.get_jobs()}
        assert job_ids == {"nightly_cache_refresh", "weekly_auto_backtest"}
    finally:
        stop_scheduler()


def test_scheduler_registers_paper_rebalance_when_live_trading_enabled(monkeypatch):
    monkeypatch.setattr(scheduler_module, "ENABLE_LIVE_TRADING", True)
    start_scheduler()
    try:
        job_ids = {job.id for job in scheduler.get_jobs()}
        assert job_ids == {
            "nightly_cache_refresh",
            "weekly_auto_backtest",
            "weekly_paper_rebalance",
            "daily_anomaly_check",
            "daily_equity_snapshot",
        }
    finally:
        stop_scheduler()
