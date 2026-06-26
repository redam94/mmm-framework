"""Project estimand aggregation for the Performance page (api/estimands.py).

Covers the pure (estimand × KPI) comparability grouping, the evidence/reference
logic mirrored from the report, the shared estimand-row helper + compute_estimands
parity, and the store-backed endpoint.
"""

from __future__ import annotations

import json

import pytest

from mmm_framework.api import estimands as E

# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_is_ratio_kind_and_labels():
    assert E.is_ratio_kind("roi", "ROI")
    assert E.is_ratio_kind("marginal_roas", "mROAS")
    assert E.is_ratio_kind("anything", "multiple")
    assert not E.is_ratio_kind("contribution", "KPI")
    assert not E.is_ratio_kind("lift", "KPI/period")
    assert E.estimand_label("contribution_roi") == "Contribution ROI"
    assert E.estimand_label("marginal_roas") == "Marginal ROAS"
    # unknown -> title-cased fallback
    assert E.estimand_label("brand_health_index") == "Brand Health Index"


def test_classify_evidence_ratio_reference_one():
    # ratio kind => reference 1.0
    assert (
        E.classify_evidence(status="ok", mean=2.0, lower=1.2, upper=3.0, reference=1.0)
        == "strong"
    )
    assert (
        E.classify_evidence(status="ok", mean=0.5, lower=0.2, upper=0.8, reference=1.0)
        == "below"
    )
    assert (
        E.classify_evidence(status="ok", mean=1.1, lower=0.6, upper=1.6, reference=1.0)
        == "uncertain"
    )


def test_classify_evidence_na_paths():
    assert (
        E.classify_evidence(
            status="unsupported", mean=None, lower=None, upper=None, reference=0.0
        )
        == "na"
    )
    assert (
        E.classify_evidence(status="ok", mean=1.0, lower=None, upper=2.0, reference=0.0)
        == "na"
    )


# ---------------------------------------------------------------------------
# group_estimands: comparability clustering
# ---------------------------------------------------------------------------


def _row(estimand, channel, kind, units, mean, lo, hi, status="ok", **extra):
    return {
        "estimand": estimand,
        "channel": channel,
        "kind": kind,
        "units": units,
        "mean": mean,
        "hdi_low": lo,
        "hdi_high": hi,
        "status": status,
        **extra,
    }


def _run(run_id, kpi, channels, rows, created_at, model_kind="mmm"):
    return {
        "run_id": run_id,
        "label": run_id,
        "model_kind": model_kind,
        "model_key": E._model_key(model_kind, kpi, channels),
        "kpi": kpi,
        "created_at": created_at,
        "estimands": rows,
    }


def test_same_estimand_same_kpi_groups_together():
    """Two models on the same estimand + KPI form ONE comparable cluster."""
    rows_a = [_row("contribution_roi", "TV", "roi", "ROI", 2.0, 1.5, 2.6)]
    rows_b = [_row("contribution_roi", "TV", "roi", "ROI", 1.8, 1.2, 2.4)]
    out = E.group_estimands(
        [
            _run("a", "revenue", ["TV"], rows_a, 100.0),
            _run("b", "revenue", ["TV"], rows_b, 200.0),
        ]
    )
    groups = out["groups"]
    assert len(groups) == 1
    g = groups[0]
    assert g["estimand"] == "contribution_roi" and g["kpi"] == "revenue"
    assert g["n_models"] == 2 and g["n_models_with_data"] == 2
    assert g["reference"] == 1.0 and g["is_ratio"] is True
    # latest first
    assert [m["run_id"] for m in g["models"]] == ["b", "a"]


def test_different_kpi_never_compared():
    """Same estimand NAME on different KPIs => two separate clusters."""
    rows_rev = [_row("contribution_roi", "TV", "roi", "ROI", 2.0, 1.5, 2.6)]
    rows_aware = [_row("contribution_roi", "TV", "roi", "ROI", 1.4, 0.9, 1.9)]
    out = E.group_estimands(
        [
            _run("rev", "revenue", ["TV"], rows_rev, 100.0),
            _run("awr", "awareness", ["TV"], rows_aware, 200.0),
        ]
    )
    assert len(out["groups"]) == 2
    kpis = {g["kpi"] for g in out["groups"]}
    assert kpis == {"revenue", "awareness"}
    for g in out["groups"]:
        assert g["n_models"] == 1
    assert out["kpis"] == ["awareness", "revenue"]


def test_two_roi_methodologies_stay_distinct():
    """contribution_roi and counterfactual_roi share kind 'roi' but are different
    numbers — they must NOT merge into one cluster."""
    rows = [
        _row("contribution_roi", "TV", "roi", "ROI", 2.0, 1.5, 2.6),
        _row("counterfactual_roi", "TV", "roi", "ROI", 1.6, 1.0, 2.2),
    ]
    out = E.group_estimands([_run("a", "revenue", ["TV"], rows, 100.0)])
    names = sorted(g["estimand"] for g in out["groups"])
    assert names == ["contribution_roi", "counterfactual_roi"]


def test_channel_union_and_per_channel_rows():
    rows_a = [
        _row("contribution_roi", "TV", "roi", "ROI", 2.0, 1.5, 2.6),
        _row("contribution_roi", "Search", "roi", "ROI", 3.0, 2.0, 4.0),
    ]
    rows_b = [
        _row("contribution_roi", "TV", "roi", "ROI", 1.8, 1.2, 2.4),
        _row("contribution_roi", "Social", "roi", "ROI", 0.6, 0.3, 0.9),
    ]
    out = E.group_estimands(
        [
            _run("a", "revenue", ["TV", "Search"], rows_a, 100.0),
            _run("b", "revenue", ["TV", "Social"], rows_b, 200.0),
        ]
    )
    g = out["groups"][0]
    assert g["channels"] == ["TV", "Search", "Social"]
    by_run = {m["run_id"]: m for m in g["models"]}
    # each model only carries the channels it has, ordered by the union
    assert [r["channel"] for r in by_run["a"]["rows"]] == ["TV", "Search"]
    assert [r["channel"] for r in by_run["b"]["rows"]] == ["TV", "Social"]
    # Search in model a is strongly positive vs ref 1.0
    search = next(r for r in by_run["a"]["rows"] if r["channel"] == "Search")
    assert search["evidence"] == "strong"
    # Social in model b is below ref 1.0
    social = next(r for r in by_run["b"]["rows"] if r["channel"] == "Social")
    assert social["evidence"] == "below"


def test_contribution_kind_uses_zero_reference():
    rows = [_row("contribution", "TV", "contribution", "KPI", 1200.0, 800.0, 1600.0)]
    out = E.group_estimands([_run("a", "revenue", ["TV"], rows, 100.0)])
    g = out["groups"][0]
    assert g["reference"] == 0.0 and g["is_ratio"] is False
    assert g["models"][0]["rows"][0]["evidence"] == "strong"  # lower 800 > 0


def test_unsupported_rows_are_na_and_not_counted_as_data():
    rows = [
        _row(
            "contribution_roi",
            "TV",
            "roi",
            "ROI",
            None,
            None,
            None,
            status="unsupported",
        )
    ]
    out = E.group_estimands([_run("a", "revenue", ["TV"], rows, 100.0)])
    g = out["groups"][0]
    assert g["n_models"] == 1 and g["n_models_with_data"] == 0
    assert g["models"][0]["rows"][0]["evidence"] == "na"


def test_is_latest_for_model_flag():
    """Two runs of the SAME structural model => only the newest is flagged latest;
    a different model is independently latest."""
    rows = [_row("contribution_roi", "TV", "roi", "ROI", 2.0, 1.5, 2.6)]
    out = E.group_estimands(
        [
            _run("old", "revenue", ["TV"], rows, 100.0),
            _run("new", "revenue", ["TV"], rows, 300.0),
            _run("other", "awareness", ["TV"], rows, 50.0),
        ]
    )
    latest = {r["run_id"]: r["is_latest_for_model"] for r in out["runs"]}
    assert latest == {"new": True, "old": False, "other": True}


# ---------------------------------------------------------------------------
# store-backed: build_project_estimands + endpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _seed_run(store, pid, run_id, kpi, channels, rows, model_kind="mmm"):
    tid = store.create_session("s", project_id=pid)["thread_id"]
    store.add_artifact(
        tid,
        "model_run",
        {
            "run_id": run_id,
            "run_name": run_id,
            "kpi": kpi,
            "channels": channels,
            "model_kind": model_kind,
            "estimands": rows,
            "spec": {"kpi": kpi},
        },
    )


def test_build_project_estimands_reads_artifacts(store):
    pid = store.create_project("P")["project_id"]
    _seed_run(
        store,
        pid,
        "a",
        "revenue",
        ["TV"],
        [_row("contribution_roi", "TV", "roi", "ROI", 2.0, 1.5, 2.6)],
    )
    _seed_run(
        store,
        pid,
        "b",
        "revenue",
        ["TV"],
        [_row("contribution_roi", "TV", "roi", "ROI", 1.8, 1.2, 2.4)],
    )
    out = E.build_project_estimands(pid)
    assert len(out["groups"]) == 1
    assert out["groups"][0]["n_models"] == 2
    assert {r["run_id"] for r in out["runs"]} == {"a", "b"}


def test_build_project_estimands_skips_runs_without_estimands(store):
    pid = store.create_project("P")["project_id"]
    tid = store.create_session("s", project_id=pid)["thread_id"]
    # a model_run with NO estimands key is skipped (pre-persistence run)
    store.add_artifact(tid, "model_run", {"run_id": "x", "kpi": "revenue"})
    out = E.build_project_estimands(pid)
    assert out["groups"] == [] and out["runs"] == []


@pytest.mark.asyncio
async def test_endpoint_returns_groups_and_404(store):
    from mmm_framework.api import main as M

    pid = store.create_project("P")["project_id"]
    _seed_run(
        store,
        pid,
        "a",
        "revenue",
        ["TV"],
        [_row("contribution_roi", "TV", "roi", "ROI", 2.0, 1.5, 2.6)],
    )
    resp = await M.project_estimands_endpoint(pid)
    body = json.loads(resp.body)
    assert body["groups"][0]["estimand"] == "contribution_roi"

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        await M.project_estimands_endpoint("does-not-exist")
    assert ei.value.status_code == 404


# ---------------------------------------------------------------------------
# shared estimand-row helper + compute_estimands parity
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, kind, status, mean, lo, hi, units, extra=None):
        self.kind = kind
        self.status = status
        self.mean = mean
        self.hdi_low = lo
        self.hdi_high = hi
        self.hdi_prob = 0.94
        self.units = units
        self.extra = extra or {}


class _FakeModel:
    def evaluate_estimands(self, estimands=None, random_seed=None):
        return {
            "contribution_roi:TV": _FakeResult(
                "roi", "ok", 2.0, 1.5, 2.6, "ROI", {"prob_positive": 0.99}
            ),
            "contribution": _FakeResult(
                "contribution", "ok", 1200.0, 800.0, 1600.0, "KPI"
            ),
        }


def test_evaluate_estimand_rows_shape():
    from mmm_framework.agents.estimand_rows import evaluate_estimand_rows

    rows = evaluate_estimand_rows(_FakeModel())
    by = {(r["estimand"], r["channel"]): r for r in rows}
    assert by[("contribution_roi", "TV")]["mean"] == 2.0
    assert by[("contribution_roi", "TV")]["prob_positive"] == 0.99
    # scalar estimand -> channel "—"
    assert ("contribution", "—") in by


def test_compute_estimands_op_still_works():
    from mmm_framework.agents.model_ops import compute_estimands

    res = compute_estimands(_FakeModel())
    assert "estimands" in res["dashboard"]
    rows = res["dashboard"]["estimands"]
    assert any(r["estimand"] == "contribution_roi" for r in rows)
    assert res["tables"]  # a table is attached
