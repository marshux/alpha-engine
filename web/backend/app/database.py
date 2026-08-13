"""SQLAlchemy engine/session setup. SQLite file lives on the persistent
Fly.io volume in production (DATABASE_PATH env var); defaults to a local
file under the repo's data/ directory for development -- same directory
the CLI already uses for its price/fundamentals cache.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "web.db"


def resolve_database_path(env_value: str | None, default: Path = DEFAULT_DB_PATH) -> Path:
    """`or`, not a dict-get default -- an empty string is falsy, so
    `DATABASE_PATH=` (set but blank, e.g. from an unfilled .env template)
    falls through to `default` instead of resolving to `Path("") == "."`,
    which `sqlite3` then fails to open with a confusing "unable to open
    database file" (it's a directory, not a file).
    """
    return Path(env_value or default)


DATABASE_PATH = resolve_database_path(os.environ.get("DATABASE_PATH"))
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DATABASE_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    # Imported here (not at module top) so importing `database` alone
    # doesn't require `models` to already be defined -- avoids a circular
    # import between the two modules.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
