"""Tests for MLflow-style run tracking (api/runs.py): dataset fingerprints,
spec hashes, and the lineage timeline (spec diff + assumptions delta vs the
previous run), plus the direct load-model endpoint's error contract."""

from __future__ import annotations

import json

import pytest

from mmm_framework.api import runs as R


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def test_data_fingerprint(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("a,b\n1,2\n3,4\n")
    fp = R.data_fingerprint(str(f))
    assert fp["n_rows"] == 2 and fp["size_bytes"] == len("a,b\n1,2\n3,4\n")
    # content-identity: same bytes -> same hash; different bytes -> different
    f2 = tmp_path / "d2.csv"
    f2.write_text("a,b\n1,2\n3,4\n")
    assert R.data_fingerprint(str(f2))["md5"] == fp["md5"]
    f2.write_text("a,b\n1,2\n3,5\n")
    assert R.data_fingerprint(str(f2))["md5"] != fp["md5"]
    assert R.data_fingerprint(str(tmp_path / "missing.csv")) is None


def test_spec_hash_stable_and_order_independent():
    h1 = R.spec_hash({"kpi": "Sales", "inference": {"draws": 1000, "chains": 4}})
    h2 = R.spec_hash({"inference": {"chains": 4, "draws": 1000}, "kpi": "Sales"})
    assert h1 == h2
    assert R.spec_hash({"kpi": "Revenue"}) != h1
    assert R.spec_hash(None) is None


class TestTimeline:
    def _seed_run(
        self, store, tid, run_id, spec, fingerprint, assumptions, parent=None
    ):
        store.add_artifact(
            tid,
            "model_run",
            {
                "run_id": run_id,
                "run_name": run_id,
                "kpi": spec.get("kpi"),
                "channels": [c["name"] for c in spec.get("media_channels", [])],
                "spec": spec,
                "data_fingerprint": fingerprint,
                "spec_hash": R.spec_hash(spec),
                "parent_run_id": parent,
                "assumptions": assumptions,
                "summary": f"run {run_id}",
            },
        )

    def test_lineage_diffs(self, store):
        tid = store.create_session(name="s", project_id="p1")["thread_id"]
        spec_v1 = {
            "kpi": "Sales",
            "media_channels": [{"name": "TV", "adstock": {"l_max": 8}}],
            "trend": {"type": "linear"},
        }
        spec_v2 = json.loads(json.dumps(spec_v1))
        spec_v2["media_channels"][0]["adstock"]["l_max"] = 12
        spec_v2["trend"] = {"type": "piecewise"}

        self._seed_run(
            store,
            tid,
            "run_1",
            spec_v1,
            {"md5": "aaa", "n_rows": 100},
            [
                {
                    "key": "tv_carryover",
                    "version": 1,
                    "category": "prior",
                    "rationale": "industry norm",
                }
            ],
        )
        self._seed_run(
            store,
            tid,
            "run_2",
            spec_v2,
            {"md5": "bbb", "n_rows": 120},
            [
                {
                    "key": "tv_carryover",
                    "version": 2,
                    "category": "prior",
                    "rationale": "longer window",
                },
                {
                    "key": "trend_break",
                    "version": 1,
                    "category": "functional_form",
                    "rationale": "PPC misfit",
                },
            ],
            parent="run_1",
        )

        runs = R.build_run_timeline("p1")
        assert [r["run_id"] for r in runs] == ["run_2", "run_1"]  # newest first

        first = runs[1]
        assert first["changes"]["baseline"] is True
        assert first["changes"]["assumptions_delta"][0]["key"] == "tv_carryover"

        second = runs[0]
        ch = second["changes"]
        assert ch["baseline"] is False
        assert ch["data_changed"] is True
        changed_paths = {c["path"] for c in ch["spec_changes"]}
        assert "media_channels.TV.adstock.l_max" in changed_paths
        assert "trend.type" in changed_paths
        deltas = {(a["key"], a["change"]) for a in ch["assumptions_delta"]}
        assert ("tv_carryover", "revised") in deltas
        assert ("trend_break", "added") in deltas
        # heavy full specs are not shipped to the client
        assert "_spec" not in second and "spec" not in second

    def test_markdown_report(self, store):
        tid = store.create_session(name="s", project_id="p2")["thread_id"]
        self._seed_run(
            store,
            tid,
            "run_1",
            {"kpi": "Sales", "media_channels": [{"name": "TV"}]},
            {"md5": "abc", "n_rows": 50},
            [{"key": "k1", "version": 1, "category": "prior", "rationale": "why"}],
        )
        md = R.run_timeline_markdown("p2")
        assert "run_1" in md and "Baseline run" in md and "`abc`" in md
        assert R.run_timeline_markdown("empty-project") == "No model runs recorded yet."


@pytest.mark.asyncio
async def test_load_model_endpoint_rejects_unknown_model(store, monkeypatch, tmp_path):
    from langgraph.checkpoint.memory import MemorySaver
    from fastapi import HTTPException

    from mmm_framework.api import main as M

    monkeypatch.setattr(M, "memory", MemorySaver())
    monkeypatch.chdir(tmp_path)  # no mmm_models dir here
    with pytest.raises(HTTPException) as exc:
        await M.load_model_endpoint("t-load", M.LoadModelRequest(name="nope"))
    assert exc.value.status_code == 400
    assert "not found" in exc.value.detail
