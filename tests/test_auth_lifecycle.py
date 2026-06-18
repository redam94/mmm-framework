"""Phase 1.4 auth lifecycle: token revocation/rotation, invites, password reset,
deactivation. Service-level tests against a temp DB + a route-level smoke."""

from __future__ import annotations

import pytest

from mmm_framework.auth import service, store
from mmm_framework.auth.config import AuthSettings, get_auth_settings
from mmm_framework.auth.routes import create_auth_router
from mmm_framework.auth.tokens import decode_jwt


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "sessions.db"
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", p)
    store.init_auth_schema(p)
    return p


@pytest.fixture()
def settings():
    return AuthSettings(enabled=True, secret="k" * 32)


def _signup(settings, org="Acme", email="a@acme.com"):
    return service.signup_organization(
        organization=org, email=email, password="a-strong-password", settings=settings
    )


# ----- token rotation + revocation --------------------------------------------


def test_refresh_rotation_single_use(db, settings):
    _u, toks = _signup(settings)
    new = service.refresh_tokens(toks.refresh_token, settings=settings)
    assert new.access_token
    # the rotated (original) refresh token can't be replayed
    with pytest.raises(service.AuthServiceError):
        service.refresh_tokens(toks.refresh_token, settings=settings)


def test_logout_revokes_refresh(db, settings):
    _u, toks = _signup(settings)
    service.logout(toks.refresh_token, settings=settings)
    with pytest.raises(service.AuthServiceError):
        service.refresh_tokens(toks.refresh_token, settings=settings)
    # logout is idempotent / tolerant of garbage
    service.logout("not-a-token", settings=settings)


# ----- invites ----------------------------------------------------------------


def test_invite_accept_flow(db, settings):
    owner, _ = _signup(settings)
    inv = service.create_invite(
        org_id=owner["org_id"],
        email="b@acme.com",
        role="analyst",
        invited_by=owner["user_id"],
        settings=settings,
    )
    user, toks = service.accept_invite(
        token=inv["token"], password="another-strong-pw", settings=settings
    )
    assert user["email"] == "b@acme.com"
    claims = decode_jwt(
        toks.access_token,
        settings.secret,
        audience=settings.audience,
        issuer=settings.issuer,
    )
    assert claims["org"] == owner["org_id"]
    assert claims["role"] == "analyst"
    # single-use
    with pytest.raises(service.AuthServiceError):
        service.accept_invite(
            token=inv["token"], password="another-strong-pw", settings=settings
        )


def test_invite_rejects_existing_email(db, settings):
    owner, _ = _signup(settings)
    with pytest.raises(service.AuthServiceError):
        service.create_invite(
            org_id=owner["org_id"],
            email="a@acme.com",
            role="analyst",
            invited_by=owner["user_id"],
            settings=settings,
        )


# ----- password reset ---------------------------------------------------------


def test_password_reset_flow(db, settings):
    _u, _ = _signup(settings)
    token = service.request_password_reset("a@acme.com", settings=settings)
    assert token
    service.confirm_password_reset(
        token=token, new_password="brand-new-password", settings=settings
    )
    with pytest.raises(service.AuthServiceError):
        service.authenticate("a@acme.com", "a-strong-password")
    assert (
        service.authenticate("a@acme.com", "brand-new-password")["email"]
        == "a@acme.com"
    )
    # token is single-use
    with pytest.raises(service.AuthServiceError):
        service.confirm_password_reset(
            token=token, new_password="yet-another-pw", settings=settings
        )


def test_password_reset_no_enumeration(db, settings):
    # unknown email returns None (caller must not reveal existence)
    assert service.request_password_reset("ghost@acme.com", settings=settings) is None


# ----- deactivation -----------------------------------------------------------


def test_deactivate_blocks_login_and_refresh(db, settings):
    owner, toks = _signup(settings)
    service.deactivate_user(owner["user_id"])
    with pytest.raises(service.AuthServiceError):
        service.authenticate("a@acme.com", "a-strong-password")
    with pytest.raises(service.AuthServiceError):
        service.refresh_tokens(toks.refresh_token, settings=settings)


# ----- route-level smoke ------------------------------------------------------


def _auth_app(settings):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.dependency_overrides[get_auth_settings] = lambda: settings
    app.include_router(create_auth_router())
    return TestClient(app)


def test_token_version_instant_kill(db, settings):
    """Bumping the token version rejects an ALREADY-ISSUED access token at once."""
    c = _auth_app(settings)
    owner = c.post(
        "/auth/signup",
        json={
            "organization": "Acme",
            "email": "a@acme.com",
            "password": "a-strong-password",
        },
    ).json()
    H = {"Authorization": f"Bearer {owner['access_token']}"}
    assert c.get("/auth/me", headers=H).status_code == 200

    # "sign out everywhere" bumps the version → the SAME live token now 401s
    assert c.post("/auth/logout-all", headers=H).status_code == 204
    assert c.get("/auth/me", headers=H).status_code == 401
    # the (single-use) refresh token from before the bump is also dead
    assert (
        c.post(
            "/auth/refresh", json={"refresh_token": owner["refresh_token"]}
        ).status_code
        == 401
    )
    # a fresh login mints a token at the new version and works again
    relog = c.post(
        "/auth/login", json={"email": "a@acme.com", "password": "a-strong-password"}
    ).json()
    assert (
        c.get(
            "/auth/me", headers={"Authorization": f"Bearer {relog['access_token']}"}
        ).status_code
        == 200
    )


def test_deactivate_instant_access_kill(db, settings):
    """Admin deactivation kills the target's LIVE access token immediately."""
    c = _auth_app(settings)
    owner = c.post(
        "/auth/signup",
        json={
            "organization": "Acme",
            "email": "a@acme.com",
            "password": "a-strong-password",
        },
    ).json()
    Hs = {"Authorization": f"Bearer {owner['access_token']}"}
    inv = c.post(
        "/auth/invite", headers=Hs, json={"email": "b@acme.com", "role": "analyst"}
    ).json()
    acc = c.post(
        "/auth/accept-invite",
        json={"token": inv["token"], "password": "another-strong-pw"},
    ).json()
    Hb = {"Authorization": f"Bearer {acc['access_token']}"}
    me = c.get("/auth/me", headers=Hb)
    assert me.status_code == 200
    bu = me.json()["user_id"]

    assert c.post(f"/auth/users/{bu}/deactivate", headers=Hs).status_code == 204
    # b's already-issued access token is dead instantly (not just on next refresh)
    assert c.get("/auth/me", headers=Hb).status_code == 401


def test_lifecycle_routes(db, settings):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.dependency_overrides[get_auth_settings] = lambda: settings
    app.include_router(create_auth_router())
    c = TestClient(app)

    owner = c.post(
        "/auth/signup",
        json={
            "organization": "Acme",
            "email": "a@acme.com",
            "password": "a-strong-password",
        },
    ).json()
    H = {"Authorization": f"Bearer {owner['access_token']}"}

    inv = c.post(
        "/auth/invite", headers=H, json={"email": "b@acme.com", "role": "analyst"}
    )
    assert inv.status_code == 200, inv.text
    assert (
        c.post(
            "/auth/accept-invite",
            json={"token": inv.json()["token"], "password": "another-strong-pw"},
        ).status_code
        == 200
    )

    # logout revokes the owner's refresh token
    assert (
        c.post(
            "/auth/logout", json={"refresh_token": owner["refresh_token"]}
        ).status_code
        == 204
    )
    assert (
        c.post(
            "/auth/refresh", json={"refresh_token": owner["refresh_token"]}
        ).status_code
        == 401
    )

    # reset request always 202 (no enumeration)
    assert (
        c.post("/auth/password-reset/request", json={"email": "a@acme.com"}).status_code
        == 202
    )
    assert (
        c.post(
            "/auth/password-reset/request", json={"email": "ghost@x.com"}
        ).status_code
        == 202
    )

    # a viewer-less analyst cannot invite (needs admin)
    analyst = c.post(
        "/auth/login", json={"email": "b@acme.com", "password": "another-strong-pw"}
    ).json()
    HA = {"Authorization": f"Bearer {analyst['access_token']}"}
    assert (
        c.post("/auth/invite", headers=HA, json={"email": "c@acme.com"}).status_code
        == 403
    )
