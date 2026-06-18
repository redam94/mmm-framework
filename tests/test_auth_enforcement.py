"""Cross-tenant enforcement matrix for Phase 1.3.

Mounts the real auth router + ``require_project_access`` guards against a temp
DB and asserts org A cannot read/write org B's projects, plus role gating.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from mmm_framework.auth import store as auth_store
from mmm_framework.auth import service
from mmm_framework.auth.config import AuthSettings, get_auth_settings
from mmm_framework.auth.deps import get_current_principal, require_project_access
from mmm_framework.auth.models import Role
from mmm_framework.auth.passwords import hash_password
from mmm_framework.auth.routes import create_auth_router


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    dbp = tmp_path / "sessions.db"
    monkeypatch.setattr(auth_store, "DEFAULT_DB_PATH", dbp)
    from mmm_framework.api import sessions as ss

    monkeypatch.setattr(ss, "DB_PATH", dbp)
    ss.init_db()
    auth_store.init_auth_schema(dbp)

    settings = AuthSettings(enabled=True, secret="t" * 32)
    app = FastAPI()
    app.dependency_overrides[get_auth_settings] = lambda: settings
    app.include_router(create_auth_router())

    @app.get(
        "/projects/{project_id}/x",
        dependencies=[Depends(require_project_access(Role.VIEWER))],
    )
    def read_x(project_id: str):
        return {"ok": True}

    @app.delete(
        "/projects/{project_id}",
        dependencies=[Depends(require_project_access(Role.ADMIN))],
    )
    def del_p(project_id: str):
        return {"ok": True}

    client = TestClient(app)

    def signup(org, email):
        r = client.post(
            "/auth/signup",
            json={"organization": org, "email": email, "password": "a-strong-password"},
        )
        assert r.status_code == 200, r.text
        tok = r.json()["access_token"]
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {tok}"}).json()
        return tok, me["org_id"]

    return {
        "client": client,
        "ss": ss,
        "dbp": dbp,
        "settings": settings,
        "signup": signup,
    }


def _h(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_cross_tenant_read_write_isolation(ctx):
    client, ss, signup = ctx["client"], ctx["ss"], ctx["signup"]
    tok_a, org_a = signup("Acme", "a@acme.com")
    tok_b, org_b = signup("Beta", "b@beta.com")

    pa = ss.create_project("PA", org_id=org_a)["project_id"]
    pb = ss.create_project("PB", org_id=org_b)["project_id"]

    # Owner of A reads/deletes its own project
    assert client.get(f"/projects/{pa}/x", headers=_h(tok_a)).status_code == 200
    assert client.delete(f"/projects/{pa}", headers=_h(tok_a)).status_code == 200

    # A cannot see B's project — 404 (indistinguishable from "unknown")
    assert client.get(f"/projects/{pb}/x", headers=_h(tok_a)).status_code == 404
    assert client.delete(f"/projects/{pb}", headers=_h(tok_a)).status_code == 404
    # ...and vice-versa
    assert client.get(f"/projects/{pa}/x", headers=_h(tok_b)).status_code == 404

    # Unknown project id is also 404
    assert client.get("/projects/ghost/x", headers=_h(tok_a)).status_code == 404


def test_missing_token_is_401(ctx):
    client, ss, signup = ctx["client"], ctx["ss"], ctx["signup"]
    _tok_a, org_a = signup("Acme", "a@acme.com")
    pa = ss.create_project("PA", org_id=org_a)["project_id"]
    assert client.get(f"/projects/{pa}/x").status_code == 401


def test_role_gating(ctx):
    client, ss, signup = ctx["client"], ctx["ss"], ctx["signup"]
    dbp, settings = ctx["dbp"], ctx["settings"]
    _tok_owner, org_a = signup("Acme", "a@acme.com")
    pa = ss.create_project("PA", org_id=org_a)["project_id"]

    # A viewer in the same org: can read, cannot delete (admin-only)
    viewer = auth_store.create_user(
        email="v@acme.com",
        password_hash=hash_password("a-strong-password"),
        org_id=org_a,
        role="viewer",
        db_path=dbp,
    )
    auth_store.add_org_member(org_a, viewer["user_id"], "viewer", db_path=dbp)
    tok_v = service.issue_tokens(
        user_id=viewer["user_id"],
        org_id=org_a,
        email="v@acme.com",
        org_role="viewer",
        settings=settings,
    ).access_token

    assert client.get(f"/projects/{pa}/x", headers=_h(tok_v)).status_code == 200
    assert client.delete(f"/projects/{pa}", headers=_h(tok_v)).status_code == 403


def test_list_scoping_primitive(ctx):
    ss = ctx["ss"]
    _tok_a, org_a = ctx["signup"]("Acme", "a@acme.com")
    _tok_b, org_b = ctx["signup"]("Beta", "b@beta.com")
    pa = ss.create_project("PA", org_id=org_a)["project_id"]
    pb = ss.create_project("PB", org_id=org_b)["project_id"]

    assert auth_store.list_org_project_ids(org_a) == {pa}
    assert auth_store.list_org_project_ids(org_b) == {pb}


# ----- session/thread + report guards (the shared require_session_access dep) --


@pytest.fixture()
def sess_ctx(tmp_path, monkeypatch):
    """Two orgs, each with a project + a session, wired through the REAL agent-app
    guard factory against a temp DB with auth enabled."""
    dbp = tmp_path / "sessions.db"
    monkeypatch.setattr(auth_store, "DEFAULT_DB_PATH", dbp)
    from mmm_framework.api import sessions as ss
    from mmm_framework.api import main as agent_main

    monkeypatch.setattr(ss, "DB_PATH", dbp)
    ss.init_db()
    auth_store.init_auth_schema(dbp)

    settings = AuthSettings(enabled=True, secret="s" * 32)

    def signup(org, email):
        from mmm_framework.auth import service

        _u, toks = service.signup_organization(
            organization=org,
            email=email,
            password="a-strong-password",
            settings=settings,
            db_path=dbp,
        )
        from mmm_framework.auth.tokens import decode_jwt

        org_id = decode_jwt(
            toks.access_token,
            settings.secret,
            audience=settings.audience,
            issuer=settings.issuer,
        )["org"]
        return toks.access_token, org_id

    tok_a, org_a = signup("Acme", "a@acme.com")
    tok_b, org_b = signup("Beta", "b@beta.com")
    proj_a = ss.create_project("PA", org_id=org_a)["project_id"]
    proj_b = ss.create_project("PB", org_id=org_b)["project_id"]
    sess_a = ss.create_session("SA", project_id=proj_a)["thread_id"]
    sess_b = ss.create_session("SB", project_id=proj_b)["thread_id"]
    return {
        "main": agent_main,
        "ss": ss,
        "settings": settings,
        "tok_a": tok_a,
        "tok_b": tok_b,
        "proj_a": proj_a,
        "proj_b": proj_b,
        "sess_a": sess_a,
        "sess_b": sess_b,
    }


def test_require_session_access_cross_tenant(sess_ctx):
    AM = sess_ctx["main"]
    app = FastAPI()
    app.dependency_overrides[get_auth_settings] = lambda: sess_ctx["settings"]

    @app.get(
        "/state/{thread_id}",
        dependencies=[Depends(AM.require_session_access(Role.VIEWER))],
    )
    def fake_state(thread_id: str):
        return {"ok": True}

    @app.get(
        "/report",
        dependencies=[
            Depends(AM.require_session_access(Role.VIEWER, deny_missing=True))
        ],
    )
    def fake_report(thread_id: str | None = None):
        return {"ok": True}

    c = TestClient(app)
    a, b = sess_ctx["sess_a"], sess_ctx["sess_b"]
    # own session ok, other org's session 404, unknown id 404
    assert c.get(f"/state/{a}", headers=_h(sess_ctx["tok_a"])).status_code == 200
    assert c.get(f"/state/{b}", headers=_h(sess_ctx["tok_a"])).status_code == 404
    assert c.get("/state/ghost", headers=_h(sess_ctx["tok_a"])).status_code == 404
    # report: cross-tenant 404, own 200, missing thread under real auth 404
    assert (
        c.get(f"/report?thread_id={b}", headers=_h(sess_ctx["tok_a"])).status_code
        == 404
    )
    assert (
        c.get(f"/report?thread_id={a}", headers=_h(sess_ctx["tok_a"])).status_code
        == 200
    )
    assert c.get("/report", headers=_h(sess_ctx["tok_a"])).status_code == 404


def test_real_session_routes_enforced(sess_ctx, monkeypatch):
    """Exercise the REAL agent-app handlers (no lifespan) for the session routes."""
    AM = sess_ctx["main"]
    AM.app.dependency_overrides[get_auth_settings] = lambda: sess_ctx["settings"]
    try:
        c = TestClient(AM.app)
        ta, tb = sess_ctx["tok_a"], sess_ctx["tok_b"]
        a, b = sess_ctx["sess_a"], sess_ctx["sess_b"]
        # GET /sessions/{id}: own 200, cross 404
        assert c.get(f"/sessions/{a}", headers=_h(ta)).status_code == 200
        assert c.get(f"/sessions/{b}", headers=_h(ta)).status_code == 404
        # DELETE cross-tenant 404
        assert c.delete(f"/sessions/{b}", headers=_h(ta)).status_code == 404
        # POST /sessions cannot plant into another org's project
        r_bad = c.post(
            "/sessions",
            headers=_h(ta),
            json={"name": "x", "project_id": sess_ctx["proj_b"]},
        )
        assert r_bad.status_code == 404
        r_ok = c.post(
            "/sessions",
            headers=_h(ta),
            json={"name": "x", "project_id": sess_ctx["proj_a"]},
        )
        assert r_ok.status_code == 200
    finally:
        AM.app.dependency_overrides.pop(get_auth_settings, None)
