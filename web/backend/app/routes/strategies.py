"""Strategy listing/detail endpoints -- the main /live dashboard view.
Each Strategy trades its own real Alpaca paper account (see
models.py::Strategy); this module is read-only metadata plus a cheap P/L
summary computed from persisted EquityPoint history, never a live Alpaca
call per row (routes/live.py handles execution and live account reads).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import require_auth
from app.database import get_db
from app.models import EquityPoint, Run, Strategy
from app.schemas import StrategyOut, StrategySummaryOut

router = APIRouter(prefix="/api/strategies", tags=["strategies"], dependencies=[Depends(require_auth)])


def _summarize(db: Session, strategy: Strategy) -> StrategySummaryOut:
    run = db.scalars(
        select(Run).where(
            Run.strategy_id == strategy.id, Run.run_type == "paper_trade", Run.status == "active"
        )
    ).first()

    run_id = status = equity = total_pnl_usd = total_pnl_pct = None
    if run is not None:
        run_id, status = run.id, run.status
        points = db.query(EquityPoint).filter_by(run_id=run.id).order_by(EquityPoint.date).all()
        if points:
            baseline = points[0].equity
            equity = points[-1].equity
            total_pnl_usd = equity - baseline
            total_pnl_pct = (equity / baseline - 1) if baseline else None

    return StrategySummaryOut(
        id=strategy.id,
        name=strategy.name,
        display_name=strategy.display_name,
        description=strategy.description,
        risk_tier=strategy.risk_tier,
        enabled=strategy.enabled,
        run_id=run_id,
        status=status,
        equity=equity,
        total_pnl_usd=total_pnl_usd,
        total_pnl_pct=total_pnl_pct,
    )


@router.get("", response_model=list[StrategySummaryOut])
def list_strategies(db: Session = Depends(get_db)) -> list[StrategySummaryOut]:
    strategies = db.scalars(select(Strategy).order_by(Strategy.id)).all()
    return [_summarize(db, s) for s in strategies]


@router.get("/{strategy_id}", response_model=StrategyOut)
def get_strategy(strategy_id: int, db: Session = Depends(get_db)) -> Strategy:
    strategy = db.get(Strategy, strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy
