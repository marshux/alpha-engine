"""Persisted state for live rebalancing that has no equivalent on Alpaca's
side: the circuit breaker's running peak/cooldown (decision/risk.py has no
concept of "state across separate invocations" -- backtest/engine.py just
carries it in a loop variable), and which day this last ran (idempotency
guard).

Lives inside alpha_engine itself, not the web backend's SQLite DB, so the
CLI and a future scheduler job (web/backend/app/jobs/) share state
naturally without either needing to depend on the other -- see
live/runner.py's docstring for why that dependency direction matters.

Stored as JSON at data/live_state.json, REPO_ROOT-relative like
data/cache.py's CACHE_DIR -- lands on the same persistent volume the web
deployment already mounts at /app/data, so it survives redeploys for free.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = REPO_ROOT / "data" / "live_state.json"


@dataclass
class LiveState:
    breaker_peak: float | None = None  # None until the first rebalance sets it
    cooldown_remaining: int = 0
    last_rebalance_date: str | None = None  # ISO date string, e.g. "2026-08-13"


def load_state(path: Path = STATE_PATH) -> LiveState:
    if not path.exists():
        return LiveState()
    return LiveState(**json.loads(path.read_text()))


def save_state(state: LiveState, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state)))
