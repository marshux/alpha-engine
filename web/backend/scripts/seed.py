"""Seeds the local dev database with a couple of fake runs, so the
frontend and API can be developed against realistic-shaped data before
app/runner.py (Phase 2) wires up real backtest execution.

Usage (from web/backend/): .venv/bin/python3 scripts/seed.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from app.database import SessionLocal, init_db
from app.models import BenchmarkPoint, EquityPoint, Run, WalkForwardWindow


def seed() -> None:
    init_db()
    db = SessionLocal()
    try:
        backtest_run = Run(
            run_type="backtest",
            status="completed",
            created_at=datetime.utcnow() - timedelta(days=1),
            completed_at=datetime.utcnow() - timedelta(days=1, minutes=-2),
            config_json=json.dumps({}),
            universe_json=json.dumps(["AAPL", "MSFT", "NVDA"]),
            start_date="2019-01-01",
            end_date="2024-12-31",
            cagr=0.1479,
            sharpe=0.95,
            sortino=1.22,
            calmar=0.62,
            max_drawdown=-0.2378,
            win_rate=0.5577,
            final_equity=228181.0,
            alpha_vs_primary_benchmark=-0.0247,
        )
        db.add(backtest_run)
        db.flush()

        equity = 100_000.0
        bench = 100_000.0
        start = datetime(2019, 1, 4)
        for week in range(52 * 3):
            date = (start + timedelta(weeks=week)).date().isoformat()
            equity *= 1.0015
            bench *= 1.0017
            db.add(EquityPoint(run_id=backtest_run.id, date=date, equity=equity))
            db.add(BenchmarkPoint(run_id=backtest_run.id, symbol="SPY", date=date, equity=bench))

        walkforward_run = Run(
            run_type="walkforward",
            status="completed",
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            config_json=json.dumps({}),
            universe_json=json.dumps(["AAPL", "MSFT", "NVDA"]),
            start_date="2008-01-01",
            end_date="2024-12-31",
            cagr=0.09,
            sharpe=0.8,
        )
        db.add(walkforward_run)
        db.flush()
        db.add(
            WalkForwardWindow(
                run_id=walkforward_run.id,
                window_index=0,
                train_start="2008-01-02",
                train_end="2011-01-02",
                test_start="2011-01-02",
                test_end="2012-01-02",
                winning_overrides_json=json.dumps({"portfolio.sizing": "min_variance"}),
                cagr=0.05,
                sharpe=0.6,
                sortino=0.7,
                calmar=0.4,
                max_drawdown=-0.12,
                win_rate=0.52,
            )
        )

        db.commit()
        print(f"Seeded backtest run {backtest_run.id} and walkforward run {walkforward_run.id}")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
