"""Web-backend integration for paper trading -- persists
alpha_engine.live.runner.run_live_rebalance results into the dashboard's
DB, one independent session per Strategy (see models.py::Strategy).
Analogous to app/runner.py (backtest/walkforward), but for an ongoing,
long-lived session instead of a one-shot completed run -- see
Run.status="active" and models.py::LiveOrder's docstrings for why.

Every function here takes a `strategy_id` (plain int), not a `Strategy`
ORM object, and re-fetches it within its own session -- same pattern as
app/runner.py::execute_run(run_id, ...). Passing ORM objects across the
session boundary (e.g. one fetched by a scheduler loop, then handed to a
function that opens its own session) risks DetachedInstanceError on any
attribute not already loaded; plain IDs sidestep that entirely.
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import date, timedelta

import pandas as pd
from alpha_engine.broker import alpaca as broker
from alpha_engine.config import load_strategy, load_universe, universe_symbols
from alpha_engine.data.equities import get_universe_prices
from alpha_engine.data.fundamentals import get_universe_fundamentals
from alpha_engine.data.macro import get_macro_data
from alpha_engine.live.anomaly_guard import run_anomaly_check
from alpha_engine.live.runner import run_live_rebalance
from alpha_engine.live.state import REPO_ROOT
from sqlalchemy import select

from app.database import SessionLocal
from app.models import AiUsageEvent, EquityPoint, LiveOrder, Run, Strategy

PAPER_TRADE_RUN_TYPE = "paper_trade"

# How much trailing history a monthly ml_alpha retrain trains on. Not
# imported from jobs/tasks.py's own DEFAULT_BACKTEST_YEARS/_date_range to
# avoid a circular import (jobs/tasks.py already imports from this
# module) -- small enough to duplicate rather than introduce a shared
# utils module just for one four-line helper.
ML_RETRAIN_LOOKBACK_YEARS = 6

logger = logging.getLogger(__name__)


def _date_range(years_back: int) -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    return start.isoformat(), end.isoformat()


def _state_path(strategy: Strategy):
    return REPO_ROOT / "data" / f"live_state_{strategy.name}.json"


def _config_overrides(strategy: Strategy) -> dict:
    return json.loads(strategy.config_overrides_json)


def _universe_symbols(strategy: Strategy) -> list[str] | None:
    return json.loads(strategy.universe_symbols_json) if strategy.universe_symbols_json else None


def _broker_client(strategy: Strategy):
    return broker.get_client(
        api_key=os.environ[strategy.alpaca_key_env],
        secret_key=os.environ[strategy.alpaca_secret_env],
    )


def _uses_ml_alpha(strategy: Strategy) -> bool:
    return bool(_config_overrides(strategy).get("signal_weights.ml_alpha", 0))


def _ml_model_path(strategy: Strategy):
    return REPO_ROOT / "data" / "models" / f"ml_alpha_{strategy.name}.txt"


def _load_ml_model(strategy: Strategy):
    """The strategy's persisted AlphaModel, if `execute_ml_retrain` has
    ever successfully saved one -- None otherwise (a brand-new
    ml_alpha-weighted strategy simply runs without that signal
    participating until its first monthly retrain lands, same "degrade
    gracefully rather than fail" posture alpha_engine.decision.regime and
    backtest/engine.py's own missing-macro-data handling already follow).
    Deferred lightgbm import -- only reached for a strategy that actually
    uses ml_alpha.
    """
    path = _ml_model_path(strategy)
    if not path.exists():
        return None
    from alpha_engine.ml.alpha_model import AlphaModel

    return AlphaModel.load(path)


def get_or_create_strategy_run(db, strategy: Strategy) -> Run:
    """Idempotent -- returns the existing active `Run` for this strategy if
    one exists, otherwise creates it. At most one active session per
    strategy: mirrors live/state.py's one-persisted-state-file-per-
    strategy, since each strategy trades exactly one real Alpaca paper
    account.
    """
    existing = db.scalars(
        select(Run).where(
            Run.run_type == PAPER_TRADE_RUN_TYPE, Run.strategy_id == strategy.id, Run.status == "active"
        )
    ).first()
    if existing is not None:
        return existing

    run = Run(
        run_type=PAPER_TRADE_RUN_TYPE,
        strategy_id=strategy.id,
        status="active",
        config_json=strategy.config_overrides_json,
        universe_json=strategy.universe_symbols_json or "[]",
        start_date=date.today().isoformat(),
        end_date=None,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def execute_paper_rebalance(strategy_id: int, session_factory=SessionLocal) -> None:
    """Runs one real rebalance against this strategy's Alpaca account (via
    alpha_engine.live.runner, sharing that strategy's own live/state.py
    file) and persists results to its `Run` row. Called by both the
    manual POST /api/live/{strategy_id}/rebalance-now endpoint and the
    weekly scheduler job.

    Unlike app/runner.py's execute_run, a per-rebalance failure does NOT
    flip the session to status="failed" -- this is a persistent session
    meant to keep trying next week, not a one-shot backtest. The error is
    recorded on error_message; status stays "active" either way.
    """
    db = session_factory()
    try:
        strategy = db.get(Strategy, strategy_id)
        run = get_or_create_strategy_run(db, strategy)

        try:
            result = run_live_rebalance(
                broker_client=_broker_client(strategy),
                state_path=_state_path(strategy),
                strategy_overrides=_config_overrides(strategy),
                symbols_override=_universe_symbols(strategy),
                macro_data=get_macro_data(),
                ml_model=_load_ml_model(strategy) if _uses_ml_alpha(strategy) else None,
            )
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
                    run_id=run.id,
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


def record_daily_equity_snapshot(strategy_id: int, session_factory=SessionLocal) -> None:
    """Records today's real Alpaca account equity as an EquityPoint even
    on days with no rebalance and nothing flagged by the anomaly guard --
    gives the dashboard's equity curve daily resolution instead of only
    one point per weekly rebalance. Idempotent per day (a rebalance may
    have already recorded today's point; this is a no-op if so). A read
    of a paper account, not a trading action -- failures are logged, not
    surfaced on Run.error_message (that field is reserved for failures
    that affect trading itself), so a transient outage here never puts a
    false alarm on the dashboard.
    """
    db = session_factory()
    try:
        strategy = db.get(Strategy, strategy_id)
        run = get_or_create_strategy_run(db, strategy)
        today_str = date.today().isoformat()
        existing = db.query(EquityPoint).filter_by(run_id=run.id, date=today_str).first()
        if existing is not None:
            return
        try:
            account = broker.get_account(_broker_client(strategy))
        except Exception:
            logger.warning("Daily equity snapshot failed for strategy %s: %s", strategy.name, traceback.format_exc())
            return
        db.add(EquityPoint(run_id=run.id, date=today_str, equity=account["equity"]))
        db.commit()
    finally:
        db.close()


def execute_anomaly_check(strategy_id: int, session_factory=SessionLocal) -> None:
    """Runs the daily per-position anomaly check (live/anomaly_guard.py)
    against this strategy's Alpaca account, entirely independent of the
    weekly rebalance, and persists any forced sells to its `Run` row
    tagged trigger="anomaly" so the dashboard can tell them apart from
    routine rebalance orders. Same non-failing error-handling posture as
    execute_paper_rebalance -- a failed check today shouldn't flip the
    session to failed, just try again on the next scheduled check.
    """
    db = session_factory()
    try:
        strategy = db.get(Strategy, strategy_id)
        run = get_or_create_strategy_run(db, strategy)

        try:
            result = run_anomaly_check(
                broker_client=_broker_client(strategy),
                strategy_overrides=_config_overrides(strategy),
            )
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


def _retrain_ml_alpha_model(strategy: Strategy) -> None:
    # Deferred: only a strategy that actually uses ml_alpha ever reaches
    # here (see execute_ml_retrain's guard below), so a plain deployment
    # with no ml_alpha-weighted strategy never needs the `ml` extra
    # (lightgbm) installed just to import this module.
    from alpha_engine.ml.alpha_model import AlphaModel
    from alpha_engine.ml.training import build_training_set

    symbols = _universe_symbols(strategy) or universe_symbols(load_universe())
    primary_benchmark = load_strategy()["backtest"]["benchmark"]
    start, end = _date_range(ML_RETRAIN_LOOKBACK_YEARS)

    prices = get_universe_prices(symbols, start, end)
    benchmark_prices = get_universe_prices([primary_benchmark], start, end).get(primary_benchmark)
    fundamentals = get_universe_fundamentals(symbols)
    macro_data = get_macro_data()

    train_start, train_end = pd.Timestamp(start), pd.Timestamp(end)
    X, y = build_training_set(
        prices,
        fundamentals,
        macro_data,
        benchmark_prices,
        symbols=symbols,
        train_start=train_start,
        train_end=train_end,
        cadence=load_strategy()["rebalance"],
    )
    if X.empty:
        logger.warning("ML retrain for strategy %s produced no training rows", strategy.name)
        return

    model = AlphaModel().train(X, y, trained_through=str(train_end.date()))
    model.save(_ml_model_path(strategy))
    logger.info("ML retrain for strategy %s saved (%d training rows)", strategy.name, len(X))


def execute_ml_retrain(strategy_id: int, session_factory=SessionLocal) -> None:
    """Refits this strategy's ml_alpha model
    (alpha_engine.ml.alpha_model.AlphaModel) on the trailing
    `ML_RETRAIN_LOOKBACK_YEARS` of history and saves it to
    `_ml_model_path(strategy)` -- `execute_paper_rebalance` picks up the
    new file on its next call automatically, no explicit hand-off needed.

    Pure research/training, no broker interaction at all -- a failure
    here is logged, not surfaced on Run.error_message (that field is
    reserved for failures that affect trading itself, same posture as
    `record_daily_equity_snapshot`'s non-trading reads). A stale or
    missing model just means `ml_alpha` sits out that week's rebalance
    (see `_load_ml_model`'s docstring), never a broken strategy.

    A no-op for a strategy that doesn't weight `ml_alpha` at all --
    `jobs/tasks.py::monthly_ml_retrain` already filters to only such
    strategies, but this stays a safe no-op either way if ever called
    directly (e.g. from a shell for a one-off manual retrain).
    """
    db = session_factory()
    try:
        strategy = db.get(Strategy, strategy_id)
        if not _uses_ml_alpha(strategy):
            return
        try:
            _retrain_ml_alpha_model(strategy)
        except Exception:
            logger.warning(
                "ML retrain failed for strategy %s: %s", strategy.name, traceback.format_exc()
            )
    finally:
        db.close()
