"""The actual scheduled job functions, called by app/jobs/scheduler.py on
a cron trigger. Kept separate from scheduler.py so each is independently
callable/testable without APScheduler running at all.
"""
from __future__ import annotations

from datetime import date, timedelta

from alpha_engine.config import load_universe, universe_symbols
from alpha_engine.data.equities import get_universe_prices
from alpha_engine.data.fundamentals import get_universe_fundamentals

from app.database import SessionLocal
from app.live_runner import execute_anomaly_check, execute_paper_rebalance, record_daily_equity_snapshot
from app.models import Run, Strategy
from app.runner import execute_run
from app.schemas import RunCreateRequest

DEFAULT_BACKTEST_YEARS = 6


def _date_range(years_back: int) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    return start.isoformat(), end.isoformat()


def _enabled_strategy_ids() -> list[int]:
    db = SessionLocal()
    try:
        return [s.id for s in db.query(Strategy).filter_by(enabled=True).all()]
    finally:
        db.close()


def nightly_cache_refresh() -> None:
    """Keeps data/cache/ warm so on-demand runs from the dashboard don't
    pay a cold-fetch penalty. No explicit force_refresh needed --
    requesting end=today naturally misses the cache once yesterday's
    cached range no longer covers today, which is true every day this
    runs, so it refetches automatically without redundant network calls
    if run more than once on the same day.
    """
    universe = load_universe()
    symbols = universe_symbols(universe, include_benchmarks=True)
    start, end = _date_range(DEFAULT_BACKTEST_YEARS)
    get_universe_prices(symbols, start, end)
    get_universe_fundamentals(symbols)


def weekly_auto_backtest() -> None:
    """Runs a full-universe backtest with the current strategy.yaml and
    persists it -- an ongoing "how would today's config have performed"
    monitoring signal, not a trading decision.
    """
    start, end = _date_range(DEFAULT_BACKTEST_YEARS)
    payload = RunCreateRequest(run_type="backtest", start_date=start, end_date=end, fundamentals=True)

    db = SessionLocal()
    try:
        run = Run(
            run_type=payload.run_type,
            status="pending",
            config_json="{}",
            universe_json="[]",
            start_date=payload.start_date,
            end_date=payload.end_date,
        )
        db.add(run)
        db.commit()
        run_id = run.id
    finally:
        db.close()

    execute_run(run_id, payload)


def weekly_paper_rebalance() -> None:
    """Runs one real (paper) rebalance for every enabled Strategy (see
    models.py::Strategy) against its own Alpaca account via
    alpha_engine.live.runner, sharing state with the CLI and the manual
    POST /api/live/{strategy_id}/rebalance-now endpoint. Registered by
    jobs/scheduler.py only when ENABLE_LIVE_TRADING is set -- unlike
    nightly_cache_refresh/weekly_auto_backtest (gated only by
    ENABLE_SCHEDULER), real account interaction gets its own explicit
    opt-in even in paper mode. `Strategy.enabled` is a second, per-
    strategy gate on top of that -- a new strategy can exist and be
    backfilled before its automation is trusted.
    """
    for strategy_id in _enabled_strategy_ids():
        execute_paper_rebalance(strategy_id)


def daily_anomaly_check() -> None:
    """Checks currently-held positions for every enabled Strategy for an
    abnormal single-day drop (alpha_engine.live.anomaly_guard) and
    force-closes any flagged position -- entirely independent of
    weekly_paper_rebalance's cadence, a faster-acting safety net for a
    stock that crashes mid-week. Registered by jobs/scheduler.py only
    when ENABLE_LIVE_TRADING is set, same gate as weekly_paper_rebalance
    (real account interaction).
    """
    for strategy_id in _enabled_strategy_ids():
        execute_anomaly_check(strategy_id)


def daily_equity_snapshot() -> None:
    """Records today's real Alpaca account equity for every enabled
    Strategy so the dashboard's equity curve has daily resolution rather
    than only one point per weekly rebalance -- lets the /live page
    render day-to-day change and color-code the ranges between
    rebalances. A read of the account, not a trading action; registered
    alongside daily_anomaly_check under the same ENABLE_LIVE_TRADING gate
    purely because it also calls Alpaca.
    """
    for strategy_id in _enabled_strategy_ids():
        record_daily_equity_snapshot(strategy_id)
