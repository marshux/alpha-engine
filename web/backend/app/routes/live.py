"""Per-strategy paper-trading endpoints: current positions (a live Alpaca
proxy, never persisted), starting a session, triggering a manual
rebalance, and the "flatten to cash" safety action. All require auth --
this exposes real (paper) account control, not just read access to run
history. Every route is scoped to one Strategy (see
models.py::Strategy) -- each strategy trades its own independent real
Alpaca paper account, resolved via that strategy's named credential env
vars, never a shared/global client.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from alpha_engine.broker import alpaca as broker
from alpha_engine.config import load_universe
from alpha_engine.data.equities import get_universe_prices

from app.ai_cost import estimate_cost_usd
from app.auth import require_auth
from app.database import get_db
from app.live_runner import execute_anomaly_check, execute_paper_rebalance, get_or_create_strategy_run
from app.models import AiUsageEvent, EquityPoint, Run, Strategy
from app.schemas import (
    AiUsageEventOut,
    AiUsageSummaryOut,
    BenchmarkPnlEntry,
    BenchmarkPnlOut,
    LiveAccountOut,
    RunSummary,
)

router = APIRouter(prefix="/api/live/{strategy_id}", tags=["live"], dependencies=[Depends(require_auth)])


def _get_strategy(strategy_id: int, db: Session) -> Strategy:
    strategy = db.get(Strategy, strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return strategy


def _get_client(strategy: Strategy):
    try:
        return broker.get_client(
            api_key=os.environ[strategy.alpaca_key_env],
            secret_key=os.environ[strategy.alpaca_secret_env],
        )
    except KeyError as e:
        raise HTTPException(
            status_code=503, detail=f"Alpaca credentials not configured (missing {e})"
        ) from e


@router.get("/positions", response_model=LiveAccountOut)
def get_positions(strategy_id: int, db: Session = Depends(get_db)) -> LiveAccountOut:
    client = _get_client(_get_strategy(strategy_id, db))
    account = broker.get_account(client)
    positions = broker.get_positions(client)
    return LiveAccountOut(
        **account,
        positions=[{"symbol": symbol, **data} for symbol, data in positions.items()],
    )


@router.post("/start", response_model=RunSummary, status_code=201)
def start_paper_trading(strategy_id: int, db: Session = Depends(get_db)) -> Run:
    strategy = _get_strategy(strategy_id, db)
    return get_or_create_strategy_run(db, strategy)


@router.post("/rebalance-now", response_model=RunSummary)
def rebalance_now(strategy_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> Run:
    strategy = _get_strategy(strategy_id, db)
    run = get_or_create_strategy_run(db, strategy)
    # Runs its own DB session (see app/live_runner.py) -- the request-scoped
    # `db` above is closed long before this actually executes. A rebalance
    # involves real network calls (Alpaca + Anthropic) that can take from
    # several seconds up to a couple minutes, so this doesn't block the
    # response -- poll GET /api/runs/{id} or this strategy's /positions
    # afterward.
    background_tasks.add_task(execute_paper_rebalance, strategy.id)
    return run


@router.post("/anomaly-check-now", response_model=RunSummary)
def anomaly_check_now(strategy_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> Run:
    strategy = _get_strategy(strategy_id, db)
    run = get_or_create_strategy_run(db, strategy)
    # Same fire-and-poll pattern as /rebalance-now -- runs its own DB
    # session, doesn't block the response.
    background_tasks.add_task(execute_anomaly_check, strategy.id)
    return run


@router.get("/ai-usage", response_model=AiUsageSummaryOut)
def get_ai_usage(strategy_id: int, db: Session = Depends(get_db)) -> AiUsageSummaryOut:
    """Cost visibility into this strategy's signals/news_ai.py Anthropic
    usage -- the component that previously ran up real, unplanned spend
    before being redesigned (see CLAUDE.md). Estimated, not billed truth:
    dollar figures are computed from raw persisted token/search counts
    via app/ai_cost.py's rate constants, cross-check the Anthropic
    console for exact billing.
    """
    strategy = _get_strategy(strategy_id, db)
    run = get_or_create_strategy_run(db, strategy)
    rows = (
        db.query(AiUsageEvent)
        .filter_by(run_id=run.id)
        .order_by(AiUsageEvent.date.desc(), AiUsageEvent.id.desc())
        .all()
    )
    events = [
        AiUsageEventOut(
            date=row.date,
            source=row.source,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            cache_creation_input_tokens=row.cache_creation_input_tokens,
            cache_read_input_tokens=row.cache_read_input_tokens,
            web_search_requests=row.web_search_requests,
            estimated_cost_usd=estimate_cost_usd(
                row.input_tokens,
                row.output_tokens,
                row.cache_creation_input_tokens,
                row.cache_read_input_tokens,
                row.web_search_requests,
            ),
        )
        for row in rows
    ]
    return AiUsageSummaryOut(
        total_calls=len(events),
        total_input_tokens=sum(e.input_tokens for e in events),
        total_output_tokens=sum(e.output_tokens for e in events),
        total_web_search_requests=sum(e.web_search_requests for e in events),
        total_estimated_cost_usd=sum(e.estimated_cost_usd for e in events),
        events=events,
    )


@router.get("/benchmark-pnl", response_model=BenchmarkPnlOut)
def get_benchmark_pnl(strategy_id: int, db: Session = Depends(get_db)) -> BenchmarkPnlOut:
    """"What if this same starting capital had been put into SPY/VOO/VTI
    instead?" -- computed from the SAME baseline (this strategy's first
    EquityPoint) the dashboard's strategy P/L headline already uses, so
    the two numbers are a fair apples-to-apples comparison rather than
    two different reference points. Not persisted -- cheap enough (a
    handful of cached daily closes) to compute fresh on every request.
    """
    strategy = _get_strategy(strategy_id, db)
    run = get_or_create_strategy_run(db, strategy)
    baseline = db.query(EquityPoint).filter_by(run_id=run.id).order_by(EquityPoint.date).first()
    if baseline is None:
        return BenchmarkPnlOut(baseline_date=None, baseline_equity=None, benchmarks=[])

    universe = load_universe()
    benchmark_symbols = universe.get("benchmarks") or ["SPY", "VOO", "VTI"]
    # yfinance's `end` is exclusive -- a session started today has
    # baseline.date == today, so start==end would fetch zero rows and
    # silently drop every symbol. +1 day guarantees today's close (if the
    # market's open) is actually included.
    fetch_end = (date.today() + timedelta(days=1)).isoformat()
    prices = get_universe_prices(benchmark_symbols, baseline.date, fetch_end)

    benchmarks = []
    for symbol in benchmark_symbols:
        df = prices.get(symbol)
        if df is None or df.empty:
            continue
        price_start = float(df["close"].iloc[0])
        price_now = float(df["close"].iloc[-1])
        pnl_pct = price_now / price_start - 1
        benchmarks.append(
            BenchmarkPnlEntry(symbol=symbol, pnl_usd=baseline.equity * pnl_pct, pnl_pct=pnl_pct)
        )

    return BenchmarkPnlOut(baseline_date=baseline.date, baseline_equity=baseline.equity, benchmarks=benchmarks)


@router.post("/flatten")
def flatten_to_cash(strategy_id: int, db: Session = Depends(get_db)) -> dict:
    client = _get_client(_get_strategy(strategy_id, db))
    broker.flatten_all(client)
    return {"status": "ok"}
