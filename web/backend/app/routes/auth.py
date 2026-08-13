"""GitHub OAuth login/callback/logout routes."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from app.auth import ALLOWED_GITHUB_USERNAME, oauth

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.github.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback")
async def auth_callback(request: Request):
    token = await oauth.github.authorize_access_token(request)
    resp = await oauth.github.get("user", token=token)
    profile = resp.json()
    username = profile.get("login")

    if username != ALLOWED_GITHUB_USERNAME:
        return RedirectResponse(url="/login?error=unauthorized")

    request.session["github_username"] = username
    return RedirectResponse(url="/")


@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


@router.get("/me")
async def me(request: Request):
    username = request.session.get("github_username")
    return {"authenticated": username == ALLOWED_GITHUB_USERNAME and bool(username), "username": username}
