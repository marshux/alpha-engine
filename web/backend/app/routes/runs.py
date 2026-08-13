"""Run history + detail + trigger endpoints. All require auth."""
from __future__ import annotations

import json

import yaml
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from alpha_engine.config import CONFIG_DIR

from app.auth import require_auth
from app.database import get_db
from app.models import Run
from app.runner import execute_run
from app.schemas import RunCreateRequest, RunDetail, RunSummary, RunVisibilityUpdate

router = APIRouter(prefix="/api/runs", tags=["runs"], dependencies=[Depends(require_auth)])


@router.get("/tuning-grid/default")
def get_default_tuning_grid() -> dict:
    """The dotted-config-path -> candidate-values grid used by `alpha-engine
    walkforward --param-grid config/tuning_grid.yaml` -- exposed so the
    "New run" form can offer it as a one-click grid-search option without
    the frontend having to reconstruct or duplicate its content.
    """
    path = CONFIG_DIR / "tuning_grid.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No default tuning grid configured")
    with open(path) as f:
        return yaml.safe_load(f)


@router.get("", response_model=list[RunSummary])
def list_runs(db: Session = Depends(get_db)) -> list[Run]:
    return list(db.scalars(select(Run).order_by(Run.created_at.desc())))


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: int, db: Session = Depends(get_db)) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/tearsheet")
def get_tearsheet(run_id: int, db: Session = Depends(get_db)) -> FileResponse:
    run = db.get(Run, run_id)
    if run is None or not run.tearsheet_path:
        raise HTTPException(status_code=404, detail="Tear sheet not found")
    return FileResponse(run.tearsheet_path, media_type="image/png")


@router.post("", response_model=RunSummary, status_code=201)
def create_run(
    payload: RunCreateRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> Run:
    if payload.run_type not in ("backtest", "walkforward"):
        raise HTTPException(status_code=422, detail=f"unknown run_type: {payload.run_type!r}")

    run = Run(
        run_type=payload.run_type,
        status="pending",
        config_json=json.dumps(payload.config_overrides or {}),
        universe_json=json.dumps(payload.symbols or []),
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Runs its own DB session (see app/runner.py) -- the request-scoped
    # `db` above is closed long before this actually executes, since a
    # walk-forward sweep can take tens of minutes.
    background_tasks.add_task(execute_run, run.id, payload)
    return run


@router.patch("/{run_id}", response_model=RunSummary)
def update_run_visibility(run_id: int, payload: RunVisibilityUpdate, db: Session = Depends(get_db)) -> Run:
    # Only field that's mutable post-creation right now -- this is how a
    # run gets listed on the unauthenticated public dashboard (app/routes/
    # public.py). Auth is already enforced at the router level.
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    run.is_public = payload.is_public
    db.commit()
    db.refresh(run)
    return run
