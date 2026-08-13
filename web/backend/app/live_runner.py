"""Web-backend integration for paper trading -- persists
alpha_engine.live.runner.run_live_rebalance results into the dashboard's
DB. Analogous to app/runner.py (backtest/walkforward), but for an ongoing,
long-lived session instead of a one-shot completed run -- see
Run.status="active" and models.py::LiveOrder's docstrings for why.
"""
from __future__ import annotations

import logging
import traceback
from datetime import date

from alpha_engine.broker import alpaca as broker
from alpha_engine.live.anomaly_guard import run_anomaly_check
from alpha_engine.live.runner import run_live_rebalance
from sqlalchemy import select

from app.database import SessionLocal
from app.models import AiUsageEvent, EquityPoint, LiveOrder, Run

PAPER_TRADE_RUN_TYPE = "paper_trade"

logger = logging.getLogger(__name__)


def get_or_create_paper_trade_run(db) -> Run:
    """Idempotent -- returns the existing active paper-trade session if one
    exists, otherwise creates it. There's at most one active session at a
    time: this mirrors alpha_engine.live.state's single persisted state
    file, since there's only one real Alpaca paper account behind either.
    """
    existing = db.scalars(
        select(Run).where(Run.run_type == PAPER_TRADE_RUN_TYPE, Run.status == "active")
    ).first()
    if existing is not None:
        return existing

    run = Run(
        run_type=PAPER_TRADE_RUN_TYPE,
        status="active",
        config_json="{}",
        universe_json="[]",
        start_date=date.today().isoformat(),
        end_date=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def execute_paper_rebalance(session_factory=SessionLocal) -> None:
    """Runs one real rebalance against Alpaca (via alpha_engine.live.runner,
    sharing the same live/state.py the CLI uses) and persists results to
    the paper-trade session's Run row. Called by both the manual
    POST /api/live/rebalance-now endpoint and the weekly scheduler job.

    Unlike app/runner.py's execute_run, a per-rebalance failure does NOT
    flip the session to status="failed" -- this is a persistent session
    meant to keep trying next week, not a one-shot backtest. The error is
    recorded on error_message; status stays "active" either way.
    """
    db = session_factory()
    try:
        run = get_or_create_paper_trade_run(db)

        try:
            result = run_live_rebalance()
        except Exception:
            run.error_message = traceback.format_exc()
            db.commit()
            return

        if result.skipped_reason:
            return  # already ran today -- nothing new to persist

        run.error_message = None
        today_str = date.today().isoformat()
        if result.account_equity is not None:
            db.add(EquityPoint(run_id=run.id, date=today_str, equity=result.account_equity))
            run.final_equity = result.account_equity
        for order in result.orders:
            db.add(
                LiveOrder(
                    run_id=run.id,
                    date=today_str,
                    symbol=order["symbol"],
                    side=order["side"],
                    notional=order["notional"],
                    alpaca_order_id=order.get("alpaca_order_id"),
                    status=order["status"],
                    trigger="rebalance",
                )
            )
        if result.ai_usage is not None:
            db.add(
                AiUsageEvent(
                    date=today_str,
                    source="news_ai_sentiment",
                    input_tokens=result.ai_usage["input_tokens"],
                    output_tokens=result.ai_usage["output_tokens"],
                    cache_creation_input_tokens=result.ai_usage["cache_creation_input_tokens"],
                    cache_read_input_tokens=result.ai_usage["cache_read_input_tokens"],
                    web_search_requests=result.ai_usage["web_search_requests"],
                )
            )
        db.commit()
    finally:
        db.close()


def record_daily_equity_snapshot(session_factory=SessionLocal) -> None:
    """Records today's real Alpaca account equity as an EquityPoint even
    on days with no rebalance and nothing flagged by the anomaly guard --
    gives the dashboard's equity curve daily resolution instead of only
    one point per weekly rebalance. Idempotent per day (a rebalance may
    have already recorded today's point; this is a no-op if so). A read
    of a public/paper account, not a trading action -- failures are
    logged, not surfaced on Run.error_message (that field is reserved for
    failures that affect trading itself), so a transient outage here
    never puts a false alarm on the dashboard.
    """
    db = session_factory()
    try:
        run = get_or_create_paper_trade_run(db)
        today_str = date.today().isoformat()
        existing = db.query(EquityPoint).filter_by(run_id=run.id, date=today_str).first()
        if existing is not None:
            return
        try:
            account = broker.get_account(broker.get_client())
        except Exception:
            logger.warning("Daily equity snapshot failed: %s", traceback.format_exc())
            return
        db.add(EquityPoint(run_id=run.id, date=today_str, equity=account["equity"]))
        db.commit()
    finally:
        db.close()


def execute_anomaly_check(session_factory=SessionLocal) -> None:
    """Runs the daily per-position anomaly check (live/anomaly_guard.py),
    entirely independent of the weekly rebalance, and persists any forced
    sells to the paper-trade session's Run row tagged trigger="anomaly" so
    the dashboard can tell them apart from routine rebalance orders. Same
    non-failing error-handling posture as execute_paper_rebalance -- a
    failed check today shouldn't flip the session to failed, just try
    again on the next scheduled check.
    """
    db = session_factory()
    try:
        run = get_or_create_paper_trade_run(db)

        try:
            result = run_anomaly_check()
        except Exception:
            run.error_message = traceback.format_exc()
            db.commit()
            return

        if not result.orders:
            return

        run.error_message = None
        today_str = date.today().isoformat()
        for order in result.orders:
            db.add(
                LiveOrder(
                    run_id=run.id,
                    date=today_str,
                    symbol=order["symbol"],
                    side=order["side"],
                    notional=order["notional"],
                    alpaca_order_id=order.get("alpaca_order_id"),
                    status=order["status"],
                    trigger="anomaly",
                )
            )
        db.commit()
    finally:
        db.close()
