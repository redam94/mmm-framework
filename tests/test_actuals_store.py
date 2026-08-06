"""Realized-KPI actuals store + reconciliation (issue #227, first deliverable).

The blocker the issue names: there was no realized-KPI record anywhere, so a
"variance to plan" could only ever restate a forecast. What these pin:

* re-stating a period keeps BOTH rows (as-of-dated, never an overwrite) and
  `latest_actuals_for_project` returns the newer while the older stays
  readable — the acceptance criterion verbatim;
* `reconcile_against_panel` returns zero disagreement when actuals equal the
  panel's aggregation and the SIGNED gap otherwise, never silently preferring
  one source; an out-of-vocabulary period is reported as unmatched and shifts
  nothing;
* the ingest endpoint round-trips CSV and JSON.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.platform import sessions as S
from mmm_framework.platform.actuals import (
    parse_actuals_records,
    reconcile_against_panel,
)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


class TestAsOfDatedStore:
    def test_restatement_keeps_both_rows(self, store):
        store.record_actuals(
            "p", [{"period": "2025-01-06", "kpi_value": 100.0}], as_of="2025-02-01"
        )
        store.record_actuals(
            "p", [{"period": "2025-01-06", "kpi_value": 105.0}], as_of="2025-03-01"
        )
        rows = store.list_actuals("p")
        assert len(rows) == 2
        assert {r["as_of"] for r in rows} == {"2025-02-01", "2025-03-01"}

    def test_latest_returns_newer_and_older_stays_readable(self, store):
        store.record_actuals(
            "p", [{"period": "2025-01-06", "kpi_value": 100.0}], as_of="2025-02-01"
        )
        store.record_actuals(
            "p", [{"period": "2025-01-06", "kpi_value": 105.0}], as_of="2025-03-01"
        )
        latest = store.latest_actuals_for_project("p")
        assert len(latest) == 1
        assert latest[0]["kpi_value"] == 105.0
        old = store.list_actuals("p", as_of="2025-02-01")
        assert old[0]["kpi_value"] == 100.0

    def test_same_as_of_replaces_that_statement_only(self, store):
        store.record_actuals("p", [{"period": "w1", "kpi_value": 1.0}], as_of="a")
        store.record_actuals("p", [{"period": "w1", "kpi_value": 2.0}], as_of="a")
        rows = store.list_actuals("p")
        assert len(rows) == 1 and rows[0]["kpi_value"] == 2.0

    def test_bad_rows_are_dropped_not_stored(self, store):
        out = store.record_actuals(
            "p",
            [
                {"period": "", "kpi_value": 1.0},
                {"period": "w1", "kpi_value": "not-a-number"},
                {"period": "w2", "kpi_value": float("nan")},
                {"period": "w3", "kpi_value": 3.0},
            ],
            as_of="a",
        )
        assert [r["period"] for r in out] == ["w3"]


class TestParser:
    def test_csv_long(self):
        raw = b"period,kpi_value\n2025-01-06,100.5\n2025-01-13,98.0\n"
        recs = parse_actuals_records(raw, "actuals.csv")
        assert recs[0] == {"period": "2025-01-06", "kpi_value": "100.5"}
        assert len(recs) == 2

    def test_csv_alias_columns(self):
        raw = b"date,actual\n2025-01-06,42\n"
        assert parse_actuals_records(raw, "x.csv")[0]["period"] == "2025-01-06"

    def test_json_mapping_and_list(self):
        assert parse_actuals_records(b'{"2025-01-06": 7}')[0]["kpi_value"] == 7
        recs = parse_actuals_records(
            b'[{"period": "w1", "kpi_value": 1.5, "source": "erp"}]'
        )
        assert recs[0]["source"] == "erp"


class _PanelStub:
    """National weekly stub: y_raw is per-period, time_idx identity."""

    def __init__(self, values, labels):
        self.y_raw = np.asarray(values, dtype=float)
        self.time_idx = np.arange(len(values))
        self._labels = labels
        self.panel = None
        self.index = None
        self.n_periods = len(values)

    @property
    def n_obs(self):
        return len(self.y_raw)


class TestReconciliation:
    def _stub(self):
        import types

        m = _PanelStub([100.0, 110.0, 120.0], ["w1", "w2", "w3"])
        coords = types.SimpleNamespace(periods=["w1", "w2", "w3"])
        m.panel = types.SimpleNamespace(coords=coords, index=None)
        return m

    def test_zero_disagreement_reads_agrees(self):
        m = self._stub()
        rec = reconcile_against_panel(
            m,
            [
                {"period": "w1", "kpi_value": 100.0},
                {"period": "w2", "kpi_value": 110.0},
            ],
        )
        assert rec["agrees"] is True
        assert rec["max_abs_gap"] == 0.0

    def test_signed_gap_reported_never_resolved(self):
        m = self._stub()
        rec = reconcile_against_panel(m, [{"period": "w2", "kpi_value": 104.0}])
        assert rec["agrees"] is False
        row = rec["periods"][0]
        # Both sources in hand, signed gap, no silent winner.
        assert row["actual"] == 104.0 and row["panel"] == 110.0
        assert row["gap"] == pytest.approx(-6.0)

    def test_unmatched_period_shifts_nothing(self):
        m = self._stub()
        rec = reconcile_against_panel(
            m,
            [
                {"period": "w1", "kpi_value": 100.0},
                {"period": "2099-12-31", "kpi_value": 5.0},
            ],
        )
        assert rec["unmatched"] == ["2099-12-31"]
        assert rec["n_matched"] == 1
        assert rec["agrees"] is False  # an unmatched period is not agreement

    def test_geo_panel_periods_are_summed(self):
        import types

        m = _PanelStub([50.0, 60.0, 50.0, 50.0], ["w1", "w2"])
        m.time_idx = np.array([0, 1, 0, 1])  # two geos x two periods
        m.n_periods = 2
        coords = types.SimpleNamespace(periods=["w1", "w2"])
        m.panel = types.SimpleNamespace(coords=coords, index=None)
        rec = reconcile_against_panel(m, [{"period": "w1", "kpi_value": 100.0}])
        assert rec["periods"][0]["panel"] == pytest.approx(100.0)
        assert rec["agrees"] is True


class TestLeanImport:
    def test_no_web_or_llm_import_at_module_scope(self):
        """The issue's lean-core criterion, checked on the AST (a docstring
        may legitimately NAME fastapi while never importing it)."""
        import ast

        import mmm_framework.platform.actuals as A

        tree = ast.parse(open(A.__file__).read())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"fastapi", "langchain", "langgraph", "httpx"}


class TestEndpoints:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
        monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
        S.init_db()
        import mmm_framework_server.main as main
        from fastapi.testclient import TestClient

        with TestClient(main.app) as c:
            yield c

    @pytest.fixture()
    def project(self, client):
        return client.post("/projects", json={"name": "P"}).json()["project_id"]

    def test_unknown_project_is_404(self, client):
        r = client.get("/projects/nope/actuals")
        assert r.status_code == 404

    def test_csv_roundtrip_with_as_of(self, client, project):
        csv_body = b"period,kpi_value\n2025-01-06,100\n2025-01-13,110\n"
        r = client.post(
            f"/projects/{project}/actuals?as_of=2025-02-01&kpi_name=Sales",
            files={"file": ("act.csv", csv_body, "text/csv")},
        )
        assert r.status_code == 200, r.text
        assert r.json()["ingested"] == 2
        # Restate one period under a newer as_of.
        r2 = client.post(
            f"/projects/{project}/actuals?as_of=2025-03-01",
            files={
                "file": ("act2.csv", b"period,kpi_value\n2025-01-06,104\n", "text/csv")
            },
        )
        assert r2.status_code == 200
        body = client.get(f"/projects/{project}/actuals").json()
        assert len(body["actuals"]) == 3  # both statements + the untouched period
        latest = {r_["period"]: r_["kpi_value"] for r_ in body["latest"]}
        assert latest["2025-01-06"] == 104.0
        assert latest["2025-01-13"] == 110.0

    def test_unparseable_upload_is_400(self, client, project):
        r = client.post(
            f"/projects/{project}/actuals",
            files={"file": ("x.csv", b"nonsense-without-columns\n", "text/csv")},
        )
        assert r.status_code == 400
