"""Tests for the scheduled job functions (app/jobs/tasks.py) -- called
directly, not through APScheduler, with data fetch and execution
monkeypatched so these need no network access and don't run a real
backtest.
"""
from __future__ import annotations

from app.jobs import scheduler as scheduler_module
from app.jobs import tasks
from app.jobs.scheduler import scheduler, start_scheduler, stop_scheduler
from app.models import Strategy


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


def _sqlite_session_factory(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base

    engine = create_engine(f"sqlite:///{tmp_path}/jobs_test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _seed_strategies(session_factory) -> tuple[int, int]:
    """One enabled, one disabled -- returns (enabled_id, disabled_id)."""
    db = session_factory()
    enabled = Strategy(**_strategy_kwargs(name="enabled_one", enabled=True))
    disabled = Strategy(**_strategy_kwargs(name="disabled_one", enabled=False))
    db.add_all([enabled, disabled])
    db.commit()
    ids = (enabled.id, disabled.id)
    db.close()
    return ids


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


def test_weekly_paper_rebalance_runs_only_enabled_strategies(monkeypatch, tmp_path):
    session_factory = _sqlite_session_factory(tmp_path)
    enabled_id, disabled_id = _seed_strategies(session_factory)
    monkeypatch.setattr(tasks, "SessionLocal", session_factory)

    calls = []
    monkeypatch.setattr(tasks, "execute_paper_rebalance", lambda strategy_id: calls.append(strategy_id))

    tasks.weekly_paper_rebalance()

    assert calls == [enabled_id]  # disabled_id never called


def test_daily_anomaly_check_runs_only_enabled_strategies(monkeypatch, tmp_path):
    session_factory = _sqlite_session_factory(tmp_path)
    enabled_id, disabled_id = _seed_strategies(session_factory)
    monkeypatch.setattr(tasks, "SessionLocal", session_factory)

    calls = []
    monkeypatch.setattr(tasks, "execute_anomaly_check", lambda strategy_id: calls.append(strategy_id))

    tasks.daily_anomaly_check()

    assert calls == [enabled_id]


def test_daily_equity_snapshot_runs_only_enabled_strategies(monkeypatch, tmp_path):
    session_factory = _sqlite_session_factory(tmp_path)
    enabled_id, disabled_id = _seed_strategies(session_factory)
    monkeypatch.setattr(tasks, "SessionLocal", session_factory)

    calls = []
    monkeypatch.setattr(tasks, "record_daily_equity_snapshot", lambda strategy_id: calls.append(strategy_id))

    tasks.daily_equity_snapshot()

    assert calls == [enabled_id]


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
