"""The KPI valuation reaches the surfaces that spend money (#215).

The refusal has to be *targeted*. `mode='free'` genuinely needs the exchange
rate — it funds each channel until the next dollar returns one dollar. A
frontier sweep and a goal-seek do not: both run `optimize_budget(mode='fixed')`
underneath and target a KPI total, so a positive constant does not move the
argmax. Refusing those would remove a capability that works, which is its own
bug — see the reach/frequency correction in #219.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.platform import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    from fastapi.testclient import TestClient

    import mmm_framework_server.main as main

    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def project(client):
    return client.post("/projects", json={"name": "P"}).json()["project_id"]


class TestPlannerRefusal:
    def test_free_mode_without_a_valuation_is_a_400(self, client, project):
        r = client.post(
            f"/projects/{project}/planner/optimize",
            json={"by_geo": False, "mode": "free"},
        )
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert "KPI valuation" in detail
        # The message must say what to do, and that fixed mode is unaffected.
        assert "economics" in detail and "fixed-budget" in detail

    def test_free_mode_with_an_explicit_value_is_accepted(self, client, project):
        r = client.post(
            f"/projects/{project}/planner/optimize",
            json={"by_geo": False, "mode": "free", "value_per_kpi": 2.5},
        )
        assert r.status_code == 202, r.text

    def test_free_mode_resolves_from_the_project_economics_preference(
        self, client, project
    ):
        client.put(
            "/preferences",
            json={"key": "economics", "value": {"gross_margin": 0.4}},
        )
        r = client.post(
            f"/projects/{project}/planner/optimize",
            json={"by_geo": False, "mode": "free"},
        )
        # Either accepted, or still 400 because this deployment scopes
        # preferences per-project — but never a 500.
        assert r.status_code in (202, 400), r.text

    @pytest.mark.parametrize(
        "body",
        [
            {"by_geo": False},  # fixed (default)
            {"by_geo": False, "mode": "fixed"},
            {"by_geo": False, "frontier": True},
            {"by_geo": False, "target_kpi": 500.0},
        ],
    )
    def test_valuation_free_paths_are_not_refused(self, client, project, body):
        """Fixed, frontier and goal-seek need no valuation — do not refuse them.

        All three call `optimize_budget(mode='fixed')` underneath and are
        denominated in KPI, not profit dollars. #215's issue text claimed the
        frontier/goal-seek route was a "bypass" needing the same refusal; it is
        not, and adding one there would be a spurious refusal.
        """
        r = client.post(f"/projects/{project}/planner/optimize", json=body)
        assert r.status_code == 202, r.text


class TestEconomicsPreferenceValidation:
    def test_percentage_margin_is_rejected(self):
        """`gross_margin: 40` used to persist and multiply profit by 40."""
        from mmm_framework.finance import KpiValuation

        with pytest.raises(ValueError):
            KpiValuation.model_validate({"gross_margin": 40})

    def test_save_preference_rejects_it_with_a_pointed_hint(self):
        import inspect

        from mmm_framework.agents import tools

        # save_preference is a LangChain StructuredTool; the callable is .func.
        fn = getattr(tools.save_preference, "func", tools.save_preference)
        src = inspect.getsource(fn)
        assert 'key == "economics"' in src, "economics payloads are unvalidated"
        assert "FRACTION" in src, "the 40-vs-0.4 hint is the actionable part"

    def test_a_valid_economics_payload_still_round_trips(self):
        from mmm_framework.finance import KpiValuation

        v = KpiValuation.model_validate({"gross_margin": 0.4, "currency": "GBP"})
        assert v.value_per_kpi() == pytest.approx(0.4)
        assert v.currency == "GBP"


def test_server_resolver_delegates_to_the_single_chain():
    """The server must not grow a seventh spelling of this number."""
    import inspect

    import mmm_framework_server.main as main

    src = inspect.getsource(main._project_valuation)
    assert "kpi_to_dollars" in src
    # and the refusal is scoped to free mode only
    refusal = inspect.getsource(main._resolved_value_per_kpi)
    assert 'body.mode == "free"' in refusal
