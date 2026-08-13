from __future__ import annotations

from pathlib import Path

from app.database import DEFAULT_DB_PATH, resolve_database_path


def test_blank_env_value_falls_back_to_default():
    # Regression: os.environ.get(key, default) only applies `default` when
    # the key is absent -- an unfilled `.env` template's `DATABASE_PATH=`
    # sets the key to "", which resolved to Path("") == "." and made
    # sqlite3 fail with "unable to open database file" (a directory, not a
    # file) rather than actually falling back.
    assert resolve_database_path("") == DEFAULT_DB_PATH


def test_missing_env_value_falls_back_to_default():
    assert resolve_database_path(None) == DEFAULT_DB_PATH


def test_real_env_value_is_used():
    assert resolve_database_path("/tmp/custom.db") == Path("/tmp/custom.db")
