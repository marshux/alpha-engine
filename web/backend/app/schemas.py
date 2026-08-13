"""Pydantic schemas for API request/response bodies."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class EquityPointOut(BaseModel):
    date: str
    equity: float
    model_config = ConfigDict(from_attributes=True)


class BenchmarkPointOut(BaseModel):
    symbol: str
    date: str
    equity: float
    model_config = ConfigDict(from_attributes=True)


class LiveOrderOut(BaseModel):
    date: str
    symbol: str
    side: str
    notional: float
    alpaca_order_id: str | None
    status: str
    trigger: str
    model_config = ConfigDict(from_attributes=True)


class AiUsageEventOut(BaseModel):
    date: str
    source: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    web_search_requests: int
    estimated_cost_usd: float
    model_config = ConfigDict(from_attributes=True)


class AiUsageSummaryOut(BaseModel):
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_web_search_requests: int
    total_estimated_cost_usd: float
    events: list[AiUsageEventOut]


class WalkForwardWindowOut(BaseModel):
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    winning_overrides_json: str
    cagr: float | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    max_drawdown: float | None
    win_rate: float | None
    model_config = ConfigDict(from_attributes=True)


class RunSummary(BaseModel):
    """Lightweight shape for the run-history list -- no time series."""

    id: int
    run_type: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    start_date: str
    end_date: str | None  # null for an active (still-running) paper_trade session
    cagr: float | None
    sharpe: float | None
    max_drawdown: float | None
    final_equity: float | None
    progress_current: int | None
    progress_total: int | None
    is_public: bool
    model_config = ConfigDict(from_attributes=True)


class RunDetail(RunSummary):
    """Full shape for a single run's detail page."""

    error_message: str | None
    config_json: str
    universe_json: str
    sortino: float | None
    calmar: float | None
    win_rate: float | None
    alpha_vs_primary_benchmark: float | None
    tearsheet_path: str | None
    equity_points: list[EquityPointOut]
    benchmark_points: list[BenchmarkPointOut]
    walkforward_windows: list[WalkForwardWindowOut]
    live_orders: list[LiveOrderOut]


class RunCreateRequest(BaseModel):
    """Body for POST /api/runs. `run_type` selects which fields apply --
    train_years/test_years/step_years/param_grid are walkforward-only.
    """

    run_type: str  # "backtest" | "walkforward"
    start_date: str
    end_date: str
    symbols: list[str] | None = None
    fundamentals: bool = False
    config_overrides: dict | None = None
    train_years: float | None = None
    test_years: float | None = None
    step_years: float | None = None
    param_grid: dict | None = None


class RunVisibilityUpdate(BaseModel):
    """Body for PATCH /api/runs/{id} -- the only mutable field on an
    existing run. Owner-only (auth-gated), this is how a run gets listed
    on the unauthenticated public dashboard.
    """

    is_public: bool


class PublicRunSummary(BaseModel):
    """No config_json/universe_json/tearsheet_path -- the public dashboard
    shows aggregate returns, never strategy internals.
    """

    id: int
    run_type: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    start_date: str
    end_date: str
    cagr: float | None
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    final_equity: float | None
    model_config = ConfigDict(from_attributes=True)


class PublicRunDetail(PublicRunSummary):
    """Equity/benchmark points are pre-filtered by the route (recent points
    older than PUBLIC_DASHBOARD_DELAY_DAYS excluded) before this is built --
    this schema doesn't do the filtering itself.
    """

    equity_points: list[EquityPointOut]
    benchmark_points: list[BenchmarkPointOut]


class LivePositionOut(BaseModel):
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float


class LiveAccountOut(BaseModel):
    """GET /api/live/positions's response -- a live proxy straight from
    Alpaca (broker/alpaca.py's get_account/get_positions), never persisted
    or cached here, so it's never stale between rebalances.
    """

    equity: float
    cash: float
    buying_power: float
    positions: list[LivePositionOut]


class LiveRebalanceOut(BaseModel):
    """Response for POST /api/live/rebalance-now -- mirrors
    alpha_engine.live.runner.LiveRebalanceResult.
    """

    skipped_reason: str | None
    account_equity: float | None
    target_weights: dict[str, float]
    orders: list[dict]
