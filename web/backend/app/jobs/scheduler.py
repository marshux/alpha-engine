"""APScheduler setup: nightly cache refresh + weekly auto-backtest.

Uses BackgroundScheduler (its own thread pool), not AsyncIOScheduler --
the job functions (data fetch, run_backtest) are synchronous, CPU/IO-bound
work, and running them directly on the FastAPI event loop would block
request handling for the job's whole duration (a walk-forward-scale run
can take tens of minutes; the weekly job is a single backtest, cheaper,
but still not something to run on the event loop).

Trigger times are server-local (UTC in production on Fly.io) -- no
explicit timezone is set.
"""
from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.jobs.tasks import nightly_cache_refresh, weekly_auto_backtest

scheduler = BackgroundScheduler()


def start_scheduler() -> None:
    if scheduler.running:
        return
    scheduler.add_job(
        nightly_cache_refresh,
        CronTrigger(hour=2, minute=0),
        id="nightly_cache_refresh",
        replace_existing=True,
    )
    scheduler.add_job(
        weekly_auto_backtest,
        CronTrigger(day_of_week="sun", hour=3, minute=0),
        id="weekly_auto_backtest",
        replace_existing=True,
    )
    scheduler.start()


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
