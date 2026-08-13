from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import auth as auth_module
from app.auth import require_auth


def _fake_request(session: dict) -> Request:
    request = Request(scope={"type": "http", "headers": []})
    request.scope["session"] = session
    return request


def test_require_auth_rejects_no_session(monkeypatch):
    monkeypatch.setattr(auth_module, "ALLOWED_GITHUB_USERNAME", "marshux")
    with pytest.raises(HTTPException) as exc_info:
        require_auth(_fake_request({}))
    assert exc_info.value.status_code == 401


def test_require_auth_rejects_non_allowlisted_user(monkeypatch):
    monkeypatch.setattr(auth_module, "ALLOWED_GITHUB_USERNAME", "marshux")
    with pytest.raises(HTTPException) as exc_info:
        require_auth(_fake_request({"github_username": "someone_else"}))
    assert exc_info.value.status_code == 401


def test_require_auth_accepts_allowlisted_user(monkeypatch):
    monkeypatch.setattr(auth_module, "ALLOWED_GITHUB_USERNAME", "marshux")
    username = require_auth(_fake_request({"github_username": "marshux"}))
    assert username == "marshux"


def test_me_endpoint_reflects_unauthenticated_session(client):
    response = client.get("/auth/me")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False, "username": None}
