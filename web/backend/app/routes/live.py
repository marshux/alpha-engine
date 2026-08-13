"""Paper-trading endpoints: current positions (a live Alpaca proxy, never
persisted), starting a session, triggering a manual rebalance, and the
"flatten to cash" safety action. All require auth -- this exposes real
(paper) account control, not just read access to run history.
"""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from alpha_engine.broker import alpaca as broker

from app.auth import require_auth
from app.database import get_db
from app.live_runner import execute_paper_rebalance, get_or_create_paper_trade_run
from app.models import Run
from app.schemas import LiveAccountOut, RunSummary

router = APIRouter(prefix="/api/live", tags=["live"], dependencies=[Depends(require_auth)])


def _get_client():
    try:
        return broker.get_client()
    except KeyError as e:
        raise HTTPException(
            status_code=503, detail=f"Alpaca credentials not configured (missing {e})"
        ) from e


@router.get("/positions", response_model=LiveAccountOut)
def get_positions() -> LiveAccountOut:
    client = _get_client()
    account = broker.get_account(client)
    positions = broker.get_positions(client)
    return LiveAccountOut(
        **account,
        positions=[{"symbol": symbol, **data} for symbol, data in positions.items()],
    )


@router.post("/start", response_model=RunSummary, status_code=201)
def start_paper_trading(db: Session = Depends(get_db)) -> Run:
    return get_or_create_paper_trade_run(db)


@router.post("/rebalance-now", response_model=RunSummary)
def rebalance_now(background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> Run:
    run = get_or_create_paper_trade_run(db)
    # Runs its own DB session (see app/live_runner.py) -- the request-scoped
    # `db` above is closed long before this actually executes. A rebalance
    # involves real network calls (Alpaca + Anthropic) that can take from
    # several seconds up to a couple minutes, so this doesn't block the
    # response -- poll GET /api/runs/{id} or /api/live/positions afterward.
    background_tasks.add_task(execute_paper_rebalance)
    return run


@router.post("/flatten")
def flatten_to_cash() -> dict:
    client = _get_client()
    broker.flatten_all(client)
    return {"status": "ok"}
