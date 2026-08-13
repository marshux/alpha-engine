"""SQLAlchemy ORM models: run history, the equity/benchmark time series
backing the dashboard's charts, and per-window walk-forward results.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Plain string, not a hardcoded enum -- "backtest" | "walkforward" today,
    # so a future "live_session" type (see CLAUDE.md's live-trading
    # extension note) doesn't need a schema migration.
    run_type: Mapped[str] = mapped_column(String, index=True)
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

    run: Mapped[Run] = relationship(back_populates="live_orders")
