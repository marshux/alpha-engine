from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.auth import require_auth
from app.database import Base, get_db
from app.main import app


@pytest.fixture
def db_session(tmp_path):
    """Fresh SQLite file per test, tables created from the same Base
    metadata the app uses -- isolated from the dev database."""
    engine = create_engine(f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    def _get_db_override():
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override
    yield TestClient(app)
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def authed_client(client):
    app.dependency_overrides[require_auth] = lambda: "marshux"
    yield client
    app.dependency_overrides.pop(require_auth, None)
