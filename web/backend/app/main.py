"""FastAPI app assembly. Run locally with `uvicorn app.main:app --reload`
from this directory (web/backend/) -- internal imports are `app.xxx`,
relying on that as the working directory, the standard FastAPI-project
convention. `alpha_engine` itself is imported normally since it's
pip-installed (editable) into the same environment.
"""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.database import init_db
from app.jobs.scheduler import start_scheduler, stop_scheduler
from app.routes import auth as auth_routes
from app.routes import live as live_routes
from app.routes import public as public_routes
from app.routes import runs as runs_routes
from app.routes import strategies as strategies_routes

# No hardcoded fallback: this repo is public, so a default key in source
# would let anyone forge a logged-in session cookie on a deployment that
# forgot to set the variable. Unset just means sessions reset on restart.
SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY") or secrets.token_hex(32)
# The Dockerfile builds web/frontend and copies its `dist/` output here.
# Doesn't exist in local dev (frontend runs on its own via `npm run dev`,
# proxying API calls to this server -- see web/frontend/vite.config.ts) --
# the static-serving/SPA-fallback routes below are skipped entirely when
# it's absent, so `uvicorn app.main:app` alone still works for API dev.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
# Off by default -- tests and quick local dev runs shouldn't spin up
# background cron threads against whatever DB happens to be configured.
# Set to "true" in production (and in a docker-compose/fly.toml env block).
ENABLE_SCHEDULER = os.environ.get("ENABLE_SCHEDULER", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if ENABLE_SCHEDULER:
        start_scheduler()
    yield
    if ENABLE_SCHEDULER:
        stop_scheduler()


app = FastAPI(title="alpha-engine dashboard", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, same_site="lax")

app.include_router(auth_routes.router)
app.include_router(runs_routes.router)
app.include_router(public_routes.router)
app.include_router(strategies_routes.router)
app.include_router(live_routes.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Catch-all registered last, so it only ever matches paths the routers
    # above (and /health, /assets) didn't already claim -- FastAPI/Starlette
    # tries routes in registration order and stops at the first match.
    # Serves a real file if the built frontend has one at that path
    # (favicon.svg, etc.), otherwise falls back to index.html so
    # client-side routes (e.g. /runs/5, a React Router path with no
    # matching file) still load the SPA instead of 404ing.
    @app.get("/{full_path:path}")
    def spa(full_path: str) -> FileResponse:
        # Resolve before serving: the path arrives percent-decoded, so
        # "..%2f" segments would otherwise walk out of STATIC_DIR and
        # serve any file the process can read (the SQLite DB, source).
        candidate = (STATIC_DIR / full_path).resolve()
        if full_path and candidate.is_relative_to(STATIC_DIR) and candidate.is_file():
            return FileResponse(candidate)
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404)
        return FileResponse(index)
