"""Web-backend integration for paper trading -- persists
alpha_engine.live.runner.run_live_rebalance results into the dashboard's
DB. Analogous to app/runner.py (backtest/walkforward), but for an ongoing,
long-lived session instead of a one-shot completed run -- see
Run.status="active" and models.py::LiveOrder's docstrings for why.
"""
from __future__ import annotations

import traceback
from datetime import date

from alpha_engine.live.runner import run_live_rebalance
from sqlalchemy import select

from app.database import SessionLocal
from app.models import EquityPoint, LiveOrder, Run

PAPER_TRADE_RUN_TYPE = "paper_trade"


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
                )
            )
        db.commit()
    finally:
        db.close()
