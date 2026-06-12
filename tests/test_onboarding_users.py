"""Tests for project onboarding (meta + KB project brief), the per-project
guide session, and the users/team registry."""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException


def _body(resp) -> dict:
    return json.loads(resp.body)


@pytest.fixture()
def api(tmp_path, monkeypatch):
    from mmm_framework.api import main as M
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    S.init_db()
    return M


class TestUsers:
    def test_crud_and_roles(self, api):
        from mmm_framework.api import sessions as S

        u = S.create_user("Ada Lovelace", "ada@example.com", "owner")
        assert u["role"] == "owner"
        with pytest.raises(ValueError, match="already exists"):
            S.create_user("Imposter", "ada@example.com")
        with pytest.raises(ValueError, match="Invalid role"):
            S.create_user("Bob", role="wizard")

        u2 = S.update_user(u["user_id"], role="viewer", name="Ada L.")
        assert u2["role"] == "viewer" and u2["name"] == "Ada L."
        with pytest.raises(ValueError, match="Unknown user"):
            S.update_user("nope", name="x")

        assert S.delete_user(u["user_id"])
        assert S.list_users() == []

    @pytest.mark.asyncio
    async def test_user_endpoints(self, api):
        u = _body(
            await api.create_user_endpoint(
                api.UserCreateRequest(name="Grace", email="g@x.com", role="analyst")
            )
        )
        out = _body(await api.list_users_endpoint())
        assert out["total"] == 1
        upd = _body(
            await api.update_user_endpoint(
                u["user_id"], api.UserUpdateRequest(role="owner")
            )
        )
        assert upd["role"] == "owner"
        with pytest.raises(HTTPException) as exc:
            await api.update_user_endpoint("missing", api.UserUpdateRequest(name="x"))
        assert exc.value.status_code == 404
        await api.delete_user_endpoint(u["user_id"])
        assert _body(await api.list_users_endpoint())["total"] == 0


class TestMembers:
    @pytest.mark.asyncio
    async def test_membership_roundtrip(self, api):
        from mmm_framework.api import sessions as S

        pid = S.create_project("P")["project_id"]
        u1 = S.create_user("A", "a@x.com")
        u2 = S.create_user("B", "b@x.com")
        out = _body(
            await api.set_members_endpoint(
                pid,
                api.MembersRequest(
                    members=[
                        {"user_id": u1["user_id"], "role": "owner"},
                        {"user_id": u2["user_id"]},
                    ]
                ),
            )
        )
        assert out["total"] == 2
        roles = {m["name"]: m["role"] for m in out["members"]}
        assert roles == {"A": "owner", "B": "analyst"}
        # replace-all semantics
        out = _body(
            await api.set_members_endpoint(
                pid, api.MembersRequest(members=[{"user_id": u2["user_id"]}])
            )
        )
        assert out["total"] == 1
        with pytest.raises(HTTPException) as exc:
            await api.set_members_endpoint(
                pid, api.MembersRequest(members=[{"user_id": "ghost"}])
            )
        assert exc.value.status_code == 400
        # deleting a user removes their memberships
        S.delete_user(u2["user_id"])
        assert _body(await api.list_members_endpoint(pid))["total"] == 0


class TestOnboarding:
    @pytest.mark.asyncio
    async def test_onboarding_saves_meta_and_brief(self, api):
        from mmm_framework.api import sessions as S

        pid = S.create_project("Acme MMM", description="FY27 measurement")["project_id"]
        u = S.create_user("Lead", "lead@x.com", "owner")
        out = _body(
            await api.project_onboarding_endpoint(
                pid,
                api.OnboardingRequest(
                    client_name="Acme Beverages",
                    industry="CPG — soft drinks",
                    goals="Grow share in the sparkling category; defend vs PrivateLabel.",
                    kpis="Sales = scanner $ sales, weekly, national.",
                    channels="TV, Search, Social; TV under long-term contract.",
                    constraints="No dark periods on TV before Q4.",
                    members=[{"user_id": u["user_id"], "role": "owner"}],
                ),
            )
        )
        meta = out["project"]["meta"]
        assert meta["client_name"] == "Acme Beverages" and meta["onboarded"] is True
        assert out["members"][0]["role"] == "owner"
        # the brief landed in the KB registry (ingestion may error without an
        # embedding backend — the doc row must exist either way)
        docs = S.list_kb_documents(pid)
        assert any(d["name"] == "project_brief.md" for d in docs)
        brief = open(
            next(d["path"] for d in docs if d["name"] == "project_brief.md")
        ).read()
        assert "Acme Beverages" in brief and "No dark periods" in brief

        # re-onboarding merges meta and REPLACES the brief (no duplicates)
        await api.project_onboarding_endpoint(
            pid, api.OnboardingRequest(industry="CPG — beverages")
        )
        docs = S.list_kb_documents(pid)
        assert sum(1 for d in docs if d["name"] == "project_brief.md") == 1
        proj = S.get_project(pid)
        assert proj["meta"]["industry"] == "CPG — beverages"
        assert proj["meta"]["client_name"] == "Acme Beverages"  # merge kept it


class TestGuideSession:
    @pytest.mark.asyncio
    async def test_guide_session_created_once(self, api):
        from mmm_framework.api import sessions as S

        pid = S.create_project("P")["project_id"]
        a = _body(await api.project_guide_session_endpoint(pid))
        assert a["created"] is True
        b = _body(await api.project_guide_session_endpoint(pid))
        assert b["created"] is False and b["thread_id"] == a["thread_id"]
        names = [s["name"] for s in S.list_sessions(project_id=pid)]
        assert names.count(api.GUIDE_SESSION_NAME) == 1

        with pytest.raises(HTTPException) as exc:
            await api.project_guide_session_endpoint("ghost")
        assert exc.value.status_code == 404
