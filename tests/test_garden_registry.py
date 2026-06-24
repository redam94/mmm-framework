"""Tests for the Model Garden registry (api/sessions.py garden_models): versioned
org-scoped CRUD, the lifecycle state machine (draft→tested→published→deprecated),
the compat-pass gate on draft→tested, published immutability, and org isolation."""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _create(store, org="orgA", name="my-model", **kw):
    return store.upsert_garden_model(
        org_id=org,
        name=name,
        manifest={"class_name": "MyMMM"},
        source_path="/tmp/m.py",
        **kw,
    )


class TestLifecycle:
    def test_create_starts_as_draft_v1(self, store):
        m = _create(store)
        assert m["status"] == "draft"
        assert m["version"] == 1
        assert m["status_history"][0]["status"] == "draft"

    def test_draft_to_tested_requires_passing_report(self, store):
        m = _create(store)
        with pytest.raises(ValueError, match="compatibility"):
            store.transition_garden_model(m["id"], "tested")
        # store a failing report -> still blocked
        store.set_garden_compat_report(m["id"], {"blocking_passed": False})
        with pytest.raises(ValueError):
            store.transition_garden_model(m["id"], "tested")
        # passing report -> allowed
        store.set_garden_compat_report(m["id"], {"blocking_passed": True, "score": 0.8})
        t = store.transition_garden_model(m["id"], "tested")
        assert t["status"] == "tested"

    def test_full_lifecycle_and_audit_trail(self, store):
        m = _create(store)
        store.set_garden_compat_report(m["id"], {"blocking_passed": True})
        store.transition_garden_model(m["id"], "tested")
        p = store.transition_garden_model(m["id"], "published", note="signed off")
        assert p["status"] == "published"
        statuses = [h["status"] for h in p["status_history"]]
        assert statuses == ["draft", "tested", "published"]
        assert p["status_history"][-1]["note"] == "signed off"

    def test_illegal_transition_raises(self, store):
        m = _create(store)
        store.set_garden_compat_report(m["id"], {"blocking_passed": True})
        store.transition_garden_model(m["id"], "tested")
        store.transition_garden_model(m["id"], "published")
        with pytest.raises(ValueError, match="Illegal transition"):
            store.transition_garden_model(m["id"], "draft")

    def test_invalid_status_rejected(self, store):
        m = _create(store)
        with pytest.raises(ValueError):
            store.transition_garden_model(m["id"], "bogus")


class TestImmutabilityAndVersioning:
    def test_published_is_immutable(self, store):
        m = _create(store)
        store.set_garden_compat_report(m["id"], {"blocking_passed": True})
        store.transition_garden_model(m["id"], "tested")
        store.transition_garden_model(m["id"], "published")
        with pytest.raises(ValueError, match="immutable"):
            store.upsert_garden_model(
                org_id="orgA", name="my-model", model_id=m["id"], docs="edit"
            )

    def test_auto_version_increment(self, store):
        a = _create(store)
        b = _create(store)
        assert a["version"] == 1 and b["version"] == 2
        assert store.next_garden_version("orgA", "my-model") == 3

    def test_duplicate_version_rejected(self, store):
        _create(store, version=5)
        with pytest.raises(ValueError, match="already exists"):
            _create(store, version=5)

    def test_draft_partial_update(self, store):
        m = _create(store)
        u = store.upsert_garden_model(
            org_id="orgA", name="my-model", model_id=m["id"], docs="now documented"
        )
        assert u["docs"] == "now documented"
        assert u["version"] == m["version"]


class TestListingAndIsolation:
    def test_org_isolation(self, store):
        _create(store, org="orgA", name="shared")
        _create(store, org="orgB", name="shared")
        a = store.list_garden_models("orgA")
        b = store.list_garden_models("orgB")
        assert {r["org_id"] for r in a} == {"orgA"}
        assert {r["org_id"] for r in b} == {"orgB"}

    def test_latest_only_collapses_versions(self, store):
        _create(store, name="m")
        _create(store, name="m")
        latest = store.list_garden_models("orgA", name=None, latest_only=True)
        ms = [r for r in latest if r["name"] == "m"]
        assert len(ms) == 1 and ms[0]["version"] == 2

    def test_get_latest_published(self, store):
        m1 = _create(store, name="m")  # v1
        store.set_garden_compat_report(m1["id"], {"blocking_passed": True})
        store.transition_garden_model(m1["id"], "tested")
        store.transition_garden_model(m1["id"], "published")
        _create(store, name="m")  # v2 draft
        pub = store.get_latest_garden_model("orgA", "m", status="published")
        assert pub["version"] == 1  # v2 isn't published yet

    def test_list_versions(self, store):
        _create(store, name="m")
        _create(store, name="m")
        vers = store.list_garden_versions("orgA", "m")
        assert [v["version"] for v in vers] == [2, 1]


class TestDeleteRules:
    def test_delete_draft_ok(self, store):
        m = _create(store)
        assert store.delete_garden_model(m["id"]) is True
        assert store.get_garden_model(model_id=m["id"]) is None

    def test_cannot_delete_published(self, store):
        m = _create(store)
        store.set_garden_compat_report(m["id"], {"blocking_passed": True})
        store.transition_garden_model(m["id"], "tested")
        store.transition_garden_model(m["id"], "published")
        with pytest.raises(ValueError, match="cannot delete"):
            store.delete_garden_model(m["id"])

    def test_deprecate_then_delete(self, store):
        m = _create(store)
        store.transition_garden_model(m["id"], "deprecated")
        assert store.delete_garden_model(m["id"]) is True


def test_resolve_org_id_fallback(store):
    # Unknown project -> the dev-posture default org (matches the dev principal).
    assert store.resolve_org_id(None) == store.DEFAULT_ORG_ID
    assert store.resolve_org_id("no-such-project") == store.DEFAULT_ORG_ID


class TestRegisterCore:
    """The shared registration core (agents/garden_registry) used by both the
    agent tool and the REST POST /model-garden endpoint."""

    @pytest.fixture()
    def reg(self, store, tmp_path, monkeypatch):
        monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
        from mmm_framework.agents import garden_registry

        return garden_registry

    def test_register_writes_source_and_row(self, store, reg):
        src = "from mmm_framework.garden import CustomMMM\nclass M(CustomMMM): pass\nGARDEN_MODEL = M\n"
        row = reg.register_garden_model_core(
            org_id="orgA", source_code=src, name="cool-model", docs="does X"
        )
        assert row["version"] == 1 and row["status"] == "draft"
        assert row["manifest"]["class_name"] == "M"
        # source round-trips via read_garden_source
        assert "class M(CustomMMM)" in reg.read_garden_source(row)

    def test_register_rejects_bad_source(self, store, reg):
        with pytest.raises(ValueError, match="multiple classes"):
            reg.register_garden_model_core(
                org_id="orgA", source_code="class A: pass\nclass B: pass", name="x"
            )

    def test_register_autoincrements(self, store, reg):
        src = "class M: pass\nGARDEN_MODEL = M\n"
        r1 = reg.register_garden_model_core(org_id="orgA", source_code=src, name="m")
        r2 = reg.register_garden_model_core(org_id="orgA", source_code=src, name="m")
        assert (r1["version"], r2["version"]) == (1, 2)


class TestDocsUpdateEndpoint:
    """PATCH /model-garden/{name}/{version}: edit docs in place (no new version),
    published immutability (409), missing version (404). Calls the handler
    directly with the dev principal, like the other endpoint unit tests."""

    @pytest.fixture()
    def main(self, store, tmp_path, monkeypatch):
        monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
        from mmm_framework.api import main as M

        return M

    def _draft(self, store, main, name="docs-edit"):
        org = main._garden_org(main._DEV_PRINCIPAL)
        row = store.upsert_garden_model(
            org_id=org,
            name=name,
            manifest={"class_name": "X"},
            source_path="/tmp/x.py",
            docs="old docs",
        )
        return org, row

    @pytest.mark.asyncio
    async def test_edit_docs_in_place(self, store, main):
        import json

        org, row = self._draft(store, main)
        resp = await main.update_garden_docs_endpoint(
            row["name"],
            row["version"],
            main.GardenDocsRequest(docs="# new docs"),
            main._DEV_PRINCIPAL,
        )
        body = json.loads(resp.body)
        assert body["docs"] == "# new docs"
        assert body["status"] == "draft"  # status untouched
        assert body["version"] == row["version"]  # no new version minted
        # persisted to the store
        got = store.get_garden_model(
            org_id=org, name=row["name"], version=row["version"]
        )
        assert got["docs"] == "# new docs"

    @pytest.mark.asyncio
    async def test_published_is_immutable_409(self, store, main):
        from fastapi import HTTPException

        _org, row = self._draft(store, main, name="docs-pub")
        store.set_garden_compat_report(row["id"], {"blocking_passed": True})
        store.transition_garden_model(row["id"], "tested")
        store.transition_garden_model(row["id"], "published")
        with pytest.raises(HTTPException) as ei:
            await main.update_garden_docs_endpoint(
                row["name"],
                row["version"],
                main.GardenDocsRequest(docs="x"),
                main._DEV_PRINCIPAL,
            )
        assert ei.value.status_code == 409

    @pytest.mark.asyncio
    async def test_missing_version_404(self, store, main):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as ei:
            await main.update_garden_docs_endpoint(
                "no-such-model",
                9,
                main.GardenDocsRequest(docs="x"),
                main._DEV_PRINCIPAL,
            )
        assert ei.value.status_code == 404
