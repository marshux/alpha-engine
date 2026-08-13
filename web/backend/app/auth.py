"""GitHub OAuth2 login -- single-user allowlist (only
ALLOWED_GITHUB_USERNAME can authenticate; this is a personal dashboard,
not a multi-user product). Session state is a signed cookie via
Starlette's SessionMiddleware (added in main.py); this module provides
the OAuth client and a FastAPI dependency that protects routes.
"""
from __future__ import annotations

import os

from authlib.integrations.starlette_client import OAuth
from fastapi import HTTPException, Request, status

GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
ALLOWED_GITHUB_USERNAME = os.environ.get("ALLOWED_GITHUB_USERNAME", "")

oauth = OAuth()
oauth.register(
    name="github",
    client_id=GITHUB_CLIENT_ID,
    client_secret=GITHUB_CLIENT_SECRET,
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "read:user"},
)


def require_auth(request: Request) -> str:
    """FastAPI dependency: raises 401 unless the session has an
    authenticated, allowlisted GitHub username. Returns that username.
    """
    username = request.session.get("github_username")
    if not username or username != ALLOWED_GITHUB_USERNAME:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return username
