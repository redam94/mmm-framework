"""Unit tests for the dependency-free auth foundation (mmm_framework.auth)."""

from __future__ import annotations

import time

import pytest

from mmm_framework.auth import (
    AuthSettings,
    Role,
    decode_jwt,
    encode_jwt,
    hash_password,
    role_satisfies,
    verify_password,
)
from mmm_framework.auth import service, store
from mmm_framework.auth.tokens import ExpiredToken, InvalidToken, make_claims


# ----- passwords --------------------------------------------------------------


def test_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert h.startswith("scrypt$")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong password", h)


def test_password_salts_differ():
    assert hash_password("same") != hash_password("same")


def test_verify_handles_garbage():
    assert not verify_password("x", None)
    assert not verify_password("x", "")
    assert not verify_password("x", "not-a-hash")
    assert not verify_password("", hash_password("x"))


# ----- tokens -----------------------------------------------------------------


def test_jwt_roundtrip_and_tamper():
    token = encode_jwt({"sub": "u1", "exp": time.time() + 60}, "secret")
    claims = decode_jwt(token, "secret")
    assert claims["sub"] == "u1"
    with pytest.raises(InvalidToken):
        decode_jwt(token, "wrong-secret")
    # tamper with the payload segment
    h, p, s = token.split(".")
    with pytest.raises(InvalidToken):
        decode_jwt(f"{h}.{p}x.{s}", "secret")


def test_jwt_expiry_and_claims():
    expired = encode_jwt({"sub": "u", "exp": time.time() - 1}, "secret")
    with pytest.raises(ExpiredToken):
        decode_jwt(expired, "secret")
    claims = make_claims(
        subject="u1",
        org_id="o1",
        org_role="owner",
        email="a@b.com",
        token_type="access",
        ttl_seconds=60,
        issuer="iss",
        audience="aud",
    )
    tok = encode_jwt(claims, "secret")
    assert decode_jwt(tok, "secret", audience="aud", issuer="iss")["org"] == "o1"
    with pytest.raises(InvalidToken):
        decode_jwt(tok, "secret", audience="other")


# ----- roles ------------------------------------------------------------------


def test_role_ordering():
    assert role_satisfies(Role.OWNER, Role.VIEWER)
    assert role_satisfies(Role.ANALYST, Role.ANALYST)
    assert not role_satisfies(Role.VIEWER, Role.ADMIN)
    assert not role_satisfies("bogus", Role.VIEWER)


# ----- store + service (against a temp DB) ------------------------------------


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "sessions.db"
    store.init_auth_schema(path)
    return path


@pytest.fixture()
def settings():
    return AuthSettings(enabled=True, secret="test-secret-xxxxxxxxxxxxxxxxxxxx")


def test_signup_creates_org_user_membership(db, settings, monkeypatch):
    # Point the store's default DB at the temp file for service calls.
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", db)
    user, tokens = service.signup_organization(
        organization="Acme",
        email="Admin@Acme.com",
        password="a-strong-password",
        settings=settings,
    )
    assert user["email"] == "admin@acme.com"  # normalized
    claims = decode_jwt(
        tokens.access_token,
        settings.secret,
        audience=settings.audience,
        issuer=settings.issuer,
    )
    assert claims["role"] == "owner"
    assert store.get_org_role(claims["org"], claims["sub"]) == "owner"


def test_duplicate_signup_rejected(db, settings, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", db)
    service.signup_organization(
        organization="Acme",
        email="a@acme.com",
        password="a-strong-password",
        settings=settings,
    )
    with pytest.raises(service.AuthServiceError):
        service.signup_organization(
            organization="Acme2",
            email="a@acme.com",
            password="a-strong-password",
            settings=settings,
        )


def test_weak_password_rejected(db, settings, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", db)
    with pytest.raises(service.AuthServiceError):
        service.signup_organization(
            organization="Acme",
            email="a@acme.com",
            password="short",
            settings=settings,
        )


def test_authenticate_roundtrip(db, settings, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", db)
    service.signup_organization(
        organization="Acme",
        email="a@acme.com",
        password="a-strong-password",
        settings=settings,
    )
    user = service.authenticate("a@acme.com", "a-strong-password")
    assert user["email"] == "a@acme.com"
    with pytest.raises(service.AuthServiceError):
        service.authenticate("a@acme.com", "nope")
    with pytest.raises(service.AuthServiceError):
        service.authenticate("ghost@acme.com", "whatever")


def test_project_tenant_attachment(db, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", db)
    org = store.create_organization("Acme", db_path=db)
    # seed a project row directly
    import sqlite3

    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO projects (project_id, name, created_at, updated_at)"
            " VALUES ('p1','P',0,0)"
        )
    assert store.get_project_org("p1", db_path=db) is None
    store.attach_project_to_org("p1", org["org_id"], db_path=db)
    assert store.get_project_org("p1", db_path=db) == org["org_id"]


def test_initialize_auth_backfills_orphans(db, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", db)
    import sqlite3

    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO projects (project_id, name, created_at, updated_at)"
            " VALUES ('p1','P',0,0)"
        )
        c.execute(
            "INSERT INTO users (user_id, name, email, role, created_at, updated_at)"
            " VALUES ('u1','U','u@x.com','analyst',0,0)"
        )
    s = AuthSettings(enabled=True, secret="secret-aaaaaaaaaaaaaaaaaaaaaaaa")
    org = service.initialize_auth(s, db_path=db)
    assert org == store.DEFAULT_ORG_ID
    assert store.get_project_org("p1", db_path=db) == org
    assert store.get_org_role(org, "u1", db_path=db) == "analyst"
    # idempotent
    assert service.initialize_auth(s, db_path=db) == org


def test_initialize_auth_bootstrap_owner(db, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", db)
    s = AuthSettings(
        enabled=True,
        secret="secret-bbbbbbbbbbbbbbbbbbbbbbbb",
        bootstrap_org="Acme",
        bootstrap_email="boss@acme.com",
        bootstrap_password="a-strong-password",
    )
    org = service.initialize_auth(s, db_path=db)
    user = store.get_user_by_email("boss@acme.com", db_path=db)
    assert user["org_id"] == org
    assert store.get_org_role(org, user["user_id"], db_path=db) == "owner"
    # second boot must not duplicate the owner or change the org
    assert service.initialize_auth(s, db_path=db) == org


def test_sessions_project_org_scoping(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as ss

    dbp = tmp_path / "sessions.db"
    monkeypatch.setattr(ss, "DB_PATH", dbp)
    ss.init_db()
    store.init_auth_schema(dbp)

    a = ss.create_project("A", org_id="org-a")
    b = ss.create_project("B", org_id="org-b")

    scoped = {p["project_id"] for p in ss.list_projects(org_id="org-a")}
    assert a["project_id"] in scoped
    assert b["project_id"] not in scoped
    # unscoped (dev) sees everything, incl. the default project
    assert len(ss.list_projects()) >= 2


def test_refresh_token_flow(db, settings, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_DB_PATH", db)
    _user, tokens = service.signup_organization(
        organization="Acme",
        email="a@acme.com",
        password="a-strong-password",
        settings=settings,
    )
    new = service.refresh_tokens(tokens.refresh_token, settings=settings)
    assert new.access_token
    # an access token must not be accepted as a refresh token
    with pytest.raises(service.AuthServiceError):
        service.refresh_tokens(tokens.access_token, settings=settings)
