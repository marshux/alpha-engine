"""SQLAlchemy ORM models: run history, the equity/benchmark time series
backing the dashboard's charts, and per-window walk-forward results.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Strategy(Base):
    """A distinct live paper-trading strategy: its own real Alpaca paper
    account (no position collisions between strategies -- each trades an
    independently funded account), its own config overrides, its own
    persisted circuit-breaker state file, and its own `Run` row (see
    `Run.strategy_id`). Everything about *how* a strategy differs from
    another lives here as data, not code -- the same alpha_engine
    execution stack (broker/alpaca.py, live/runner.py,
    live/anomaly_guard.py) runs unchanged for every strategy.
    """

    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)  # slug, e.g. "sector_rotation"
    display_name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    risk_tier: Mapped[str] = mapped_column(String)  # e.g. "traditional" | "aggressive", for UI badges

    # Dotted-path -> value, applied on top of config/strategy.yaml via
    # alpha_engine.config.apply_overrides (see live/runner.py). e.g.
    # {"signal_weights.momentum": 1.0, "portfolio.top_n": 4}.
    config_overrides_json: Mapped[str] = mapped_column(Text)
    # Nullable: null means "trade the full configured universe" (most
    # strategies); a JSON list of symbols restricts to just those (e.g.
    # Sector Rotation's 11 SPDR sector ETFs). Passed through as
    # live/runner.py::run_live_rebalance's `symbols_override`.
    universe_symbols_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # NAMES of the Fly-secret env vars holding this strategy's Alpaca
    # credentials (e.g. "ALPACA_API_KEY_SECTOR_ROTATION") -- never the
    # secrets themselves. The DB stays out of the secrets business, same
    # as every other credential in this project.
    alpaca_key_env: Mapped[str] = mapped_column(String)
    alpaca_secret_env: Mapped[str] = mapped_column(String)

    # Per-strategy on/off, ANDed with the global ENABLE_LIVE_TRADING kill
    # switch in jobs/scheduler.py -- lets a strategy be created and
    # backfilled with history before its scheduled automation is trusted.
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    runs: Mapped[list["Run"]] = relationship(back_populates="strategy")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Plain string, not a hardcoded enum -- "backtest" | "walkforward" today,
    # so a future "live_session" type (see CLAUDE.md's live-trading
    # extension note) doesn't need a schema migration.
    run_type: Mapped[str] = mapped_column(String, index=True)
    # Set only for run_type="paper_trade" -- backtest/walkforward runs
    # aren't tied to any one strategy.
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategies.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String, default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Set only for walkforward runs (a grid sweep is the only thing slow
    # enough to need this) -- backtest runs leave these null throughout.
    # `progress_total` is fixed once the run starts (windows * (grid
    # combos + 1)); `progress_current` is updated as run_walk_forward's
    # progress_callback fires, so the frontend can poll and render a bar
    # instead of the run just looking "stuck" on status=running.
    progress_current: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress_total: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Opt-in, not opt-out -- a run is only ever exposed on the unauthenticated
    # public dashboard (app/routes/public.py) if this is explicitly set True
    # via PATCH /api/runs/{id}. Default False so nothing is public by
    # accident. The public routes additionally never expose config_json,
    # universe_json, tearsheet_path, or walkforward_windows (winning
    # overrides reveal tuned signal weights), and redact equity/benchmark
    # points more recent than PUBLIC_DASHBOARD_DELAY_DAYS -- see public.py.
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    config_json: Mapped[str] = mapped_column(Text)  # full strategy config overrides used, as JSON
    universe_json: Mapped[str] = mapped_column(Text)  # symbol list used, as JSON
    start_date: Mapped[str] = mapped_column(String)
    # Nullable: a paper_trade run's session has no defined end while it's
    # still active (status="active") -- every other run_type always sets
    # this at creation.
    end_date: Mapped[str | None] = mapped_column(String, nullable=True)

    # Summary metrics as direct columns rather than a joined table -- a
    # small fixed set queried on every run-history list view.
    cagr: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    sortino: Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    alpha_vs_primary_benchmark: Mapped[float | None] = mapped_column(Float, nullable=True)

    tearsheet_path: Mapped[str | None] = mapped_column(String, nullable=True)

    equity_points: Mapped[list["EquityPoint"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="EquityPoint.date"
    )
    benchmark_points: Mapped[list["BenchmarkPoint"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="BenchmarkPoint.date"
    )
    walkforward_windows: Mapped[list["WalkForwardWindow"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="WalkForwardWindow.window_index"
    )
    live_orders: Mapped[list["LiveOrder"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="LiveOrder.id"
    )
    strategy: Mapped[Strategy | None] = relationship(back_populates="runs")


class EquityPoint(Base):
    __tablename__ = "equity_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    date: Mapped[str] = mapped_column(String)
    equity: Mapped[float] = mapped_column(Float)

    run: Mapped[Run] = relationship(back_populates="equity_points")


class BenchmarkPoint(Base):
    __tablename__ = "benchmark_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    symbol: Mapped[str] = mapped_column(String, index=True)
    date: Mapped[str] = mapped_column(String)
    equity: Mapped[float] = mapped_column(Float)

    run: Mapped[Run] = relationship(back_populates="benchmark_points")


class WalkForwardWindow(Base):
    __tablename__ = "walkforward_windows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    window_index: Mapped[int] = mapped_column(Integer)
    train_start: Mapped[str] = mapped_column(String)
    train_end: Mapped[str] = mapped_column(String)
    test_start: Mapped[str] = mapped_column(String)
    test_end: Mapped[str] = mapped_column(String)
    winning_overrides_json: Mapped[str] = mapped_column(Text)

    cagr: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    sortino: Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped[Run] = relationship(back_populates="walkforward_windows")


class LiveOrder(Base):
    """A submission-time audit trail for orders alpha_engine.broker.alpaca
    actually submitted during a paper-trading rebalance -- deliberately NOT
    a live fill-state tracker (status is whatever Alpaca returned at
    submission, never updated afterward). Cross-check the Alpaca dashboard
    (or GET /api/live/positions, which is a live proxy) for current fill
    state; this table is a record of what was submitted and when.
    """

    __tablename__ = "live_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    date: Mapped[str] = mapped_column(String)
    symbol: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)
    notional: Mapped[float] = mapped_column(Float)
    alpaca_order_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String)
    # "rebalance" (the weekly cycle) or "anomaly" (live/anomaly_guard.py's
    # daily per-position outlier check, forced closure outside the
    # rebalance cadence entirely) -- lets the dashboard tell the two apart.
    trigger: Mapped[str] = mapped_column(String, default="rebalance", server_default="rebalance")

    run: Mapped[Run] = relationship(back_populates="live_orders")


class AiUsageEvent(Base):
    """One row per real (non-cache-hit) call to signals/news_ai.py's
    Anthropic API -- raw token/search counts only, not a pre-computed
    dollar figure (see app/ai_cost.py for why). Exists because this
    specific signal has a real cost history worth watching (an earlier
    per-ticker design cost ~$27/rebalance before being redesigned -- see
    CLAUDE.md), and there was previously no visibility into it beyond
    checking the Anthropic console by hand.
    """

    __tablename__ = "ai_usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Nullable for backward compatibility with rows recorded before
    # multi-strategy support (a single global cost ledger, unscoped) --
    # every row recorded from here on always sets it.
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True, index=True)
    date: Mapped[str] = mapped_column(String, index=True)
    source: Mapped[str] = mapped_column(String)  # e.g. "news_ai_sentiment"
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cache_creation_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    web_search_requests: Mapped[int] = mapped_column(Integer, default=0)
