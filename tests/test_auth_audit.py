"""P1 observability — structured audit emission + org-scoped, plan-gated export."""

from __future__ import annotations

import json
import logging

import pytest

from mmm_framework.auth import audit, service, store
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
    return AuthSettings(enabled=True, secret="a" * 32)


def _install_sink(path):
    from mmm_framework.agents import audit_sink

    lg = logging.getLogger("mmm_audit")
    for h in list(lg.handlers):
        if isinstance(h, audit_sink.HashChainAuditHandler):
            lg.removeHandler(h)
    return audit_sink.install_audit_sink(str(path))


@pytest.fixture()
def sink(tmp_path):
    path = tmp_path / "audit.jsonl"
    h = _install_sink(path)
    yield str(path)
    logging.getLogger("mmm_audit").removeHandler(h)


# ----- structured emission ----------------------------------------------------


def test_emission_is_structured_and_queryable(sink):
    audit.audit_event("auth.login", user_id="u1", org_id="o1")
    audit.audit_event("auth.logout", user_id="u2", org_id="o2")
    recs = [json.loads(l) for l in open(sink)]
    assert [r["event"] for r in recs] == ["auth.login", "auth.logout"]
    assert recs[0]["fields"]["org_id"] == "o1"
    assert recs[0]["fields"]["user_id"] == "u1"
    # hash chain is intact
    from mmm_framework.agents import audit_sink

    ok, err = audit_sink.verify(sink)
    assert ok, err


def test_read_events_filters_by_org(sink):
    audit.audit_event("auth.login", user_id="ua", org_id="A")
    audit.audit_event("auth.login", user_id="ub", org_id="B")
    audit.audit_event("auth.logout", user_id="ua", org_id="A")
    a = audit.read_audit_events("A")
    assert len(a) == 2
    assert all(r["fields"]["org_id"] == "A" for r in a)
    assert audit.read_audit_events("ghost") == []


def test_read_events_no_log_is_empty(tmp_path):
    assert audit.read_audit_events("A", path=str(tmp_path / "nope.jsonl")) == []


# ----- plan-gated export route ------------------------------------------------


def test_audit_export_route_gated_by_plan(db, settings, sink):
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
    org = c.get("/auth/me", headers=H).json()["org_id"]
    # a login emits an org-tagged auth.login event into the sink
    c.post("/auth/login", json={"email": "a@acme.com", "password": "a-strong-password"})

    # free plan lacks audit_export -> 402
    assert c.get("/auth/audit-export", headers=H).status_code == 402

    # business plan includes it -> 200 with the org's events
    store.set_org_plan(org, "business")
    r = c.get("/auth/audit-export", headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["org_id"] == org
    assert body["count"] >= 1
    assert all(e["fields"]["org_id"] == org for e in body["events"])
