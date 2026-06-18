"""P0.4 — subscription tiers, entitlements, usage metering, plan limits."""

from __future__ import annotations

import sqlite3

import pytest

from mmm_framework.auth import plans, service, store
from mmm_framework.auth.config import AuthSettings, get_auth_settings
from mmm_framework.auth.routes import create_auth_router


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "sessions.db"
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", p)
    store.init_auth_schema(p)
    return p


@pytest.fixture()
def settings():
    return AuthSettings(enabled=True, secret="p" * 32)


def _signup(settings, org="Acme", email="a@acme.com"):
    user, _ = service.signup_organization(
        organization=org, email=email, password="a-strong-password", settings=settings
    )
    return user


# ----- plan definitions -------------------------------------------------------


def test_plan_defaults_and_features():
    assert plans.get_plan(None).key == "free"
    assert plans.get_plan("bogus").key == "free"  # unknown → default
    assert plans.get_plan("team").max_seats == 5
    assert plans.get_plan("enterprise").max_seats is None  # unlimited
    assert "sso" in plans.get_plan("business").features
    assert "sso" not in plans.get_plan("team").features
    assert plans.get_plan("enterprise").has("audit_export")


# ----- usage metering ---------------------------------------------------------


def test_usage_reflects_plan_and_counts(db, settings):
    owner = _signup(settings)
    org = owner["org_id"]
    u = plans.org_usage(org)
    assert u["plan"] == "free"
    assert u["seats"] == {"used": 1, "limit": 2, "remaining": 1, "over": False}
    assert u["projects"]["used"] == 0
    assert u["fits_this_month"]["used"] == 0
    # enterprise => unlimited (limit None, remaining None)
    store.set_org_plan(org, "enterprise")
    u2 = plans.org_usage(org)
    assert u2["seats"]["limit"] is None and u2["seats"]["remaining"] is None


# ----- seat limit -------------------------------------------------------------


def _invite(settings, org, owner_id, email):
    return service.create_invite(
        org_id=org, email=email, role="analyst", invited_by=owner_id, settings=settings
    )


def test_seat_limit_blocks_accept_until_upgrade(db, settings):
    owner = _signup(settings)  # 1 member; free allows 2 seats
    org = owner["org_id"]
    # fill the 2nd seat
    inv_b = _invite(settings, org, owner["user_id"], "b@acme.com")
    service.accept_invite(
        token=inv_b["token"], password="another-strong-pw", settings=settings
    )
    # a 3rd seat is over the free cap — the invite can't be redeemed
    inv_c = _invite(settings, org, owner["user_id"], "c@acme.com")
    with pytest.raises(service.AuthServiceError):
        service.accept_invite(
            token=inv_c["token"], password="another-strong-pw", settings=settings
        )
    # upgrading frees a seat — the SAME invite now redeems
    store.set_org_plan(org, "team")  # 5 seats
    user_c, _ = service.accept_invite(
        token=inv_c["token"], password="another-strong-pw", settings=settings
    )
    assert user_c["email"] == "c@acme.com"


# ----- project limit ----------------------------------------------------------


def test_project_limit(db, settings):
    owner = _signup(settings)
    org = owner["org_id"]
    with sqlite3.connect(db) as c:
        for i in range(3):  # free cap is 3
            c.execute(
                "INSERT INTO projects (project_id, name, org_id, created_at, updated_at)"
                " VALUES (?,?,?,0,0)",
                (f"p{i}", f"P{i}", org),
            )
    with pytest.raises(plans.PlanLimitError):
        plans.assert_within_project_limit(org)
    store.set_org_plan(org, "team")  # 10 projects
    plans.assert_within_project_limit(org)  # ok now


# ----- route: usage + feature gate -------------------------------------------


def test_usage_endpoint_and_feature_gate(db, settings):
    from fastapi import Depends, FastAPI
    from fastapi.testclient import TestClient

    from mmm_framework.auth.deps import require_plan_feature

    app = FastAPI()
    app.dependency_overrides[get_auth_settings] = lambda: settings
    app.include_router(create_auth_router())

    @app.get("/sso-only", dependencies=[Depends(require_plan_feature("sso"))])
    def sso_only():
        return {"ok": True}

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

    u = c.get("/auth/usage", headers=H).json()
    assert u["plan"] == "free"
    assert u["seats"]["limit"] == 2

    org = c.get("/auth/me", headers=H).json()["org_id"]
    # free plan lacks sso -> 402; the gate reads the LIVE plan (no re-login needed)
    assert c.get("/sso-only", headers=H).status_code == 402
    store.set_org_plan(org, "business")
    assert c.get("/sso-only", headers=H).status_code == 200
