"""Paper-trading endpoints: current positions (a live Alpaca proxy, never
persisted), starting a session, triggering a manual rebalance, and the
"flatten to cash" safety action. All require auth -- this exposes real
(paper) account control, not just read access to run history.
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from alpha_engine.broker import alpaca as broker
from alpha_engine.config import load_universe
from alpha_engine.data.equities import get_universe_prices

from app.ai_cost import estimate_cost_usd
from app.auth import require_auth
from app.database import get_db
from app.live_runner import execute_anomaly_check, execute_paper_rebalance, get_or_create_paper_trade_run
from app.models import AiUsageEvent, EquityPoint, Run
from app.schemas import (
    AiUsageEventOut,
    AiUsageSummaryOut,
    BenchmarkPnlEntry,
    BenchmarkPnlOut,
    LiveAccountOut,
    RunSummary,
)

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


@router.post("/anomaly-check-now", response_model=RunSummary)
def anomaly_check_now(background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> Run:
    run = get_or_create_paper_trade_run(db)
    # Same fire-and-poll pattern as /rebalance-now -- runs its own DB
    # session, doesn't block the response.
    background_tasks.add_task(execute_anomaly_check)
    return run


@router.get("/ai-usage", response_model=AiUsageSummaryOut)
def get_ai_usage(db: Session = Depends(get_db)) -> AiUsageSummaryOut:
    """All-time cost visibility into signals/news_ai.py's Anthropic usage --
    the component that previously ran up real, unplanned spend before
    being redesigned (see CLAUDE.md). Estimated, not billed truth: dollar
    figures are computed from raw persisted token/search counts via
    app/ai_cost.py's rate constants, cross-check the Anthropic console for
    exact billing.
    """
    rows = db.query(AiUsageEvent).order_by(AiUsageEvent.date.desc(), AiUsageEvent.id.desc()).all()
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
def get_benchmark_pnl(db: Session = Depends(get_db)) -> BenchmarkPnlOut:
    """"What if this same starting capital had been put into SPY/VOO/VTI
    instead?" -- computed from the SAME baseline (the session's first
    EquityPoint) the dashboard's strategy P/L headline already uses, so
    the two numbers are a fair apples-to-apples comparison rather than
    two different reference points. Not persisted -- cheap enough (a
    handful of cached daily closes) to compute fresh on every request.
    """
    run = get_or_create_paper_trade_run(db)
    baseline = db.query(EquityPoint).filter_by(run_id=run.id).order_by(EquityPoint.date).first()
    if baseline is None:
        return BenchmarkPnlOut(baseline_date=None, baseline_equity=None, benchmarks=[])

    universe = load_universe()
    benchmark_symbols = universe.get("benchmarks") or ["SPY", "VOO", "VTI"]
    prices = get_universe_prices(benchmark_symbols, baseline.date, date.today().isoformat())

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
def flatten_to_cash() -> dict:
    client = _get_client()
    broker.flatten_all(client)
    return {"status": "ok"}
