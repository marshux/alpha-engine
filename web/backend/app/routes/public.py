"""Unauthenticated read-only routes for runs the owner has explicitly
opted into the public dashboard (Run.is_public, toggled via the owner-only
PATCH /api/runs/{id}). Deliberately never exposes config_json,
universe_json, tearsheet_path, or walkforward_windows (winning_overrides
reveals tuned signal weights) -- only aggregate return metrics and an
equity curve, and even that curve is truncated to hide anything more
recent than PUBLIC_DASHBOARD_DELAY_DAYS, so this can't be used to infer
current positioning or time entries/exits against the live strategy.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Run
from app.schemas import PublicRunDetail, PublicRunSummary

router = APIRouter(prefix="/api/public", tags=["public"])

PUBLIC_DASHBOARD_DELAY_DAYS = int(os.environ.get("PUBLIC_DASHBOARD_DELAY_DAYS", "30"))

# Explicit allowlist, independent of is_public/status -- defense in depth
# so a paper_trade run (real position/order timing) can never leak through
# here even via a future bug (e.g. a generic "make public" toggle reused
# across run types, or a new terminal status that isn't "completed" but
# also isn't "active"). Today status alone already excludes paper_trade
# rows (they're never "completed"), but that's an accident of the current
# status values, not something this route should rely on.
PUBLIC_RUN_TYPES = ("backtest", "walkforward")


@router.get("/meta")
def public_meta() -> dict:
    # So the frontend can display the actual configured delay instead of
    # hardcoding a number that could drift from PUBLIC_DASHBOARD_DELAY_DAYS.
    return {"delay_days": PUBLIC_DASHBOARD_DELAY_DAYS}


def _visible_cutoff() -> str:
    return (date.today() - timedelta(days=PUBLIC_DASHBOARD_DELAY_DAYS)).isoformat()


@router.get("/runs", response_model=list[PublicRunSummary])
def list_public_runs(db: Session = Depends(get_db)) -> list[Run]:
    return list(
        db.scalars(
            select(Run)
            .where(
                Run.is_public.is_(True),
                Run.status == "completed",
                Run.run_type.in_(PUBLIC_RUN_TYPES),
            )
            .order_by(Run.created_at.desc())
        )
    )


@router.get("/runs/{run_id}", response_model=PublicRunDetail)
def get_public_run(run_id: int, db: Session = Depends(get_db)) -> PublicRunDetail:
    run = db.get(Run, run_id)
    if (
        run is None
        or not run.is_public
        or run.status != "completed"
        or run.run_type not in PUBLIC_RUN_TYPES
    ):
        raise HTTPException(status_code=404, detail="Run not found")

    cutoff = _visible_cutoff()
    return PublicRunDetail(
        id=run.id,
        run_type=run.run_type,
        status=run.status,
        created_at=run.created_at,
        completed_at=run.completed_at,
        start_date=run.start_date,
        end_date=run.end_date,
        cagr=run.cagr,
        sharpe=run.sharpe,
        max_drawdown=run.max_drawdown,
        win_rate=run.win_rate,
        final_equity=run.final_equity,
        equity_points=[p for p in run.equity_points if p.date < cutoff],
        benchmark_points=[p for p in run.benchmark_points if p.date < cutoff],
    )
