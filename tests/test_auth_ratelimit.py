"""Per-org rate limiting (Phase 1.4 hardening)."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from mmm_framework.auth import ratelimit, store
from mmm_framework.auth.config import AuthSettings, get_auth_settings
from mmm_framework.auth.ratelimit import RateLimitSettings, require_org_rate_limit
from mmm_framework.auth.routes import create_auth_router


@pytest.fixture()
def app_ctx(tmp_path, monkeypatch):
    dbp = tmp_path / "sessions.db"
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", dbp)
    store.init_auth_schema(dbp)
    ratelimit._BUCKETS.reset()

    auth_settings = AuthSettings(enabled=True, secret="r" * 32)
    monkeypatch.setattr(
        ratelimit,
        "get_ratelimit_settings",
        lambda: RateLimitSettings(enabled=True, chat_per_window=3, window_seconds=60),
    )

    app = FastAPI()
    app.dependency_overrides[get_auth_settings] = lambda: auth_settings
    app.include_router(create_auth_router())

    @app.get("/chatish", dependencies=[Depends(require_org_rate_limit("chat"))])
    def chatish():
        return {"ok": True}

    return TestClient(app)


def _token(c, org, email):
    r = c.post(
        "/auth/signup",
        json={"organization": org, "email": email, "password": "a-strong-password"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_429_after_limit(app_ctx):
    H = _token(app_ctx, "Acme", "a@acme.com")
    for _ in range(3):
        assert app_ctx.get("/chatish", headers=H).status_code == 200
    r = app_ctx.get("/chatish", headers=H)
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_limit_is_per_org(app_ctx):
    Ha = _token(app_ctx, "Acme", "a@acme.com")
    Hb = _token(app_ctx, "Beta", "b@beta.com")
    for _ in range(3):
        app_ctx.get("/chatish", headers=Ha)
    assert app_ctx.get("/chatish", headers=Ha).status_code == 429  # A exhausted
    assert app_ctx.get("/chatish", headers=Hb).status_code == 200  # B unaffected


def test_disabled_means_no_limit(app_ctx, monkeypatch):
    monkeypatch.setattr(
        ratelimit, "get_ratelimit_settings", lambda: RateLimitSettings(enabled=False)
    )
    H = _token(app_ctx, "Acme", "a@acme.com")
    for _ in range(10):
        assert app_ctx.get("/chatish", headers=H).status_code == 200


def test_dev_principal_not_limited(app_ctx, monkeypatch):
    # Auth disabled -> dev principal -> rate limiting is bypassed even when enabled.
    monkeypatch.setitem(
        app_ctx.app.dependency_overrides,
        get_auth_settings,
        lambda: AuthSettings(enabled=False),
    )
    for _ in range(10):
        assert app_ctx.get("/chatish").status_code == 200
