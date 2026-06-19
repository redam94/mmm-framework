"""Admin/org-management auth routes: members, roles, invites, change-password."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mmm_framework.auth import store as auth_store
from mmm_framework.auth.config import AuthSettings, get_auth_settings
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
    client = TestClient(app)

    def signup(org, email):
        r = client.post(
            "/auth/signup",
            json={"organization": org, "email": email, "password": "a-strong-password"},
        )
        assert r.status_code == 200, r.text
        tok = r.json()["access_token"]
        me = client.get("/auth/me", headers=_h(tok)).json()
        return tok, me

    def invite_and_accept(admin_tok, email, role="analyst"):
        r = client.post(
            "/auth/invite", json={"email": email, "role": role}, headers=_h(admin_tok)
        )
        assert r.status_code == 200, r.text
        token = r.json()["token"]
        r2 = client.post(
            "/auth/accept-invite",
            json={"token": token, "password": "another-strong-pw"},
        )
        assert r2.status_code == 200, r2.text
        tok = r2.json()["access_token"]
        me = client.get("/auth/me", headers=_h(tok)).json()
        return tok, me, token

    return {"client": client, "signup": signup, "invite_and_accept": invite_and_accept}


def _h(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_members_list_and_role_change(ctx):
    client, signup, invite_and_accept = (
        ctx["client"],
        ctx["signup"],
        ctx["invite_and_accept"],
    )
    owner_tok, owner = signup("Acme", "owner@acme.com")

    r = client.get("/auth/members", headers=_h(owner_tok))
    assert r.status_code == 200
    assert len(r.json()["members"]) == 1
    assert r.json()["members"][0]["role"] == "owner"

    _mtok, member, _itok = invite_and_accept(owner_tok, "ana@acme.com", "analyst")
    members = client.get("/auth/members", headers=_h(owner_tok)).json()["members"]
    assert len(members) == 2

    # Promote analyst -> admin
    r = client.patch(
        f"/auth/members/{member['user_id']}",
        json={"role": "admin"},
        headers=_h(owner_tok),
    )
    assert r.status_code == 200 and r.json()["role"] == "admin"


def test_cannot_demote_or_remove_last_owner(ctx):
    client, signup = ctx["client"], ctx["signup"]
    owner_tok, owner = signup("Acme", "owner@acme.com")
    oid = owner["user_id"]

    r = client.patch(
        f"/auth/members/{oid}", json={"role": "analyst"}, headers=_h(owner_tok)
    )
    assert r.status_code == 400 and "last owner" in r.json()["detail"]

    # Removing yourself is blocked before the owner check even fires.
    r = client.delete(f"/auth/members/{oid}", headers=_h(owner_tok))
    assert r.status_code == 400 and "yourself" in r.json()["detail"]


def test_remove_member(ctx):
    client, signup, invite_and_accept = (
        ctx["client"],
        ctx["signup"],
        ctx["invite_and_accept"],
    )
    owner_tok, _ = signup("Acme", "owner@acme.com")
    _t, member, _i = invite_and_accept(owner_tok, "ana@acme.com", "analyst")

    r = client.delete(f"/auth/members/{member['user_id']}", headers=_h(owner_tok))
    assert r.status_code == 204
    members = client.get("/auth/members", headers=_h(owner_tok)).json()["members"]
    assert len(members) == 1


def test_invites_list_and_revoke(ctx):
    client, signup = ctx["client"], ctx["signup"]
    owner_tok, _ = signup("Acme", "owner@acme.com")

    r = client.post(
        "/auth/invite",
        json={"email": "pending@acme.com", "role": "analyst"},
        headers=_h(owner_tok),
    )
    token = r.json()["token"]
    invites = client.get("/auth/invites", headers=_h(owner_tok)).json()["invites"]
    assert len(invites) == 1 and invites[0]["email"] == "pending@acme.com"

    assert (
        client.delete(f"/auth/invites/{token}", headers=_h(owner_tok)).status_code
        == 204
    )
    assert client.get("/auth/invites", headers=_h(owner_tok)).json()["invites"] == []
    assert (
        client.delete(f"/auth/invites/{token}", headers=_h(owner_tok)).status_code
        == 404
    )


def test_admin_routes_require_admin_role(ctx):
    client, signup, invite_and_accept = (
        ctx["client"],
        ctx["signup"],
        ctx["invite_and_accept"],
    )
    owner_tok, _ = signup("Acme", "owner@acme.com")
    analyst_tok, _, _ = invite_and_accept(owner_tok, "ana@acme.com", "analyst")

    assert client.get("/auth/members", headers=_h(analyst_tok)).status_code == 403
    assert client.get("/auth/invites", headers=_h(analyst_tok)).status_code == 403


def test_change_password_reissues_and_revokes_old(ctx):
    client, signup = ctx["client"], ctx["signup"]
    owner_tok, _ = signup("Acme", "owner@acme.com")

    # Wrong current password -> 400.
    r = client.post(
        "/auth/change-password",
        json={"current_password": "nope", "new_password": "brand-new-password"},
        headers=_h(owner_tok),
    )
    assert r.status_code == 400

    r = client.post(
        "/auth/change-password",
        json={
            "current_password": "a-strong-password",
            "new_password": "brand-new-password",
        },
        headers=_h(owner_tok),
    )
    assert r.status_code == 200
    new_tok = r.json()["access_token"]

    # Old token is revoked (token_version bumped); new one works.
    assert client.get("/auth/me", headers=_h(owner_tok)).status_code == 401
    assert client.get("/auth/me", headers=_h(new_tok)).status_code == 200
