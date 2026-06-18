"""Tests for per-run history metrics: the kernel-side snapshot
(planning/history.py), host-side enrichment + persistence (api/history.py),
the trajectory/coverage/priorities assembly, the project endpoints, and the
backfill module's skip behavior."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


class _StubModel:
    """Just enough surface for compute_response_curves: additive per-channel
    sqrt response with per-draw coefficients (decision uncertainty in ch0)."""

    def __init__(self, n_obs=52, seed=5):
        rng = np.random.default_rng(seed)
        self.channel_names = ["TV", "Digital"]
        self.X_media_raw = np.abs(rng.normal(10.0, 2.0, size=(n_obs, 2)))
        self._coefs = np.stack(
            [np.where(rng.random(60) < 0.5, 0.5, 2.0), np.full(60, 1.0)], axis=1
        )  # (D, C)

    def sample_channel_contributions(self, X_media, max_draws=200, random_seed=None):
        D = min(max_draws, self._coefs.shape[0])
        contrib = (
            np.sqrt(np.maximum(X_media, 0.0))[None, :, :] * self._coefs[:D, None, :]
        )
        return contrib  # (D, obs, C)


@pytest.fixture()
def metrics():
    from mmm_framework.planning.history import compute_run_metrics

    return compute_run_metrics(_StubModel(), max_draws=60, random_seed=0)


class TestComputeRunMetrics:
    def test_snapshot_shape_and_json_safety(self, metrics):
        import json

        from mmm_framework.planning.history import RUN_METRICS_SCHEMA_VERSION

        assert metrics["schema_version"] == RUN_METRICS_SCHEMA_VERSION
        assert set(metrics["channels"]) == {"TV", "Digital"}
        tv = metrics["channels"]["TV"]
        for key in (
            "spend",
            "spend_share",
            "roi_mean",
            "roi_sd",
            "ci_width",
            "marginal_roi",
            "share_gap",
            "eig",
            "evoi",
            "quadrant",
        ):
            assert key in tv
        p = metrics["portfolio"]
        assert p["total_spend"] > 0
        assert p["mean_ci_width"] > 0
        assert p["evpi"] >= 0
        json.dumps(metrics)  # must round-trip without numpy leakage

    def test_response_curves_block(self, metrics):
        rc = metrics["response_curves"]
        mults = rc["multipliers"]
        assert mults[rc["current_index"]] == pytest.approx(1.0)
        assert set(rc["channels"]) == {"TV", "Digital"}
        tv = rc["channels"]["TV"]
        G = len(mults)
        assert (
            len(tv["spend"]) == len(tv["mean"]) == len(tv["p5"]) == len(tv["p95"]) == G
        )
        # monotone spend grid, band brackets the mean, sqrt response is increasing
        assert tv["spend"] == sorted(tv["spend"])
        assert all(lo <= m <= hi for lo, m, hi in zip(tv["p5"], tv["mean"], tv["p95"]))
        assert tv["mean"][-1] > tv["mean"][0]

    def test_uncertain_channel_has_wider_ci_and_higher_priority(self, metrics):
        tv, dig = metrics["channels"]["TV"], metrics["channels"]["Digital"]
        assert tv["ci_width"] > dig["ci_width"]
        assert tv["priority"] > dig["priority"]


class TestPersistAndAssembly:
    def _seed_run(self, store, metrics, run_id, tid, created_at):
        from mmm_framework.api.history import persist_run_metrics

        import copy

        model_run = {"run_id": run_id, "metrics": copy.deepcopy(metrics)}
        persist_run_metrics(model_run, tid, created_at=created_at)
        return model_run

    def test_enrichment_and_series(self, store, metrics):
        from mmm_framework.api.history import build_history_series

        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s1", project_id=pid)["thread_id"]
        # calibrated evidence for TV only
        e = store.upsert_experiment(
            channel="TV", project_id=pid, status="completed", end_date="2026-05-01"
        )
        store.transition_experiment(e["id"], "calibrated", calibrated_run_id="r1")

        self._seed_run(store, metrics, "r1", tid, created_at=1000.0)
        self._seed_run(store, metrics, "r2", tid, created_at=2000.0)

        out = build_history_series(pid)
        assert [r["run_id"] for r in out["runs"]] == ["r1", "r2"]  # oldest first
        assert set(out["channels"]) == {"TV", "Digital"}
        assert len(out["series"]["roi"]["TV"]) == 2
        assert out["series"]["calibration"]["TV"][0]["status"] == "experiment_backed"
        assert out["series"]["calibration"]["Digital"][0]["status"] == "model_only"
        assert len(out["portfolio"]) == 2
        assert out["portfolio"][0]["expected_uplift"] is not None

    def test_coverage_tiers_and_decay(self, store, metrics):
        from mmm_framework.api.history import build_calibration_coverage

        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s1", project_id=pid)["thread_id"]
        e = store.upsert_experiment(
            channel="TV", project_id=pid, status="completed", end_date="2024-01-01"
        )
        store.transition_experiment(e["id"], "calibrated")
        self._seed_run(store, metrics, "r1", tid, created_at=1000.0)

        cov = build_calibration_coverage(pid, as_of="2026-06-10")
        by = {c["channel"]: c for c in cov["channels"]}
        # TV has evidence but it's 2.5 years old -> decayed -> stale
        assert by["TV"]["tier"] == "stale" and by["TV"]["retest_due"]
        assert by["Digital"]["tier"] == "model_only"
        assert cov["coverage_pct"] == pytest.approx(50.0)
        assert 0 < cov["spend_weighted_coverage_pct"] < 100

    def test_fresh_evidence_is_not_stale(self, store, metrics):
        """Freshness floor: evidence a month old stays 'calibrated' even when
        the posterior is wide enough that the decayed EIG clears the
        threshold (a perpetual-stale dashboard is a broken dashboard)."""
        from mmm_framework.api.history import build_calibration_coverage

        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s1", project_id=pid)["thread_id"]
        e = store.upsert_experiment(
            channel="TV", project_id=pid, status="completed", end_date="2026-05-10"
        )
        store.transition_experiment(e["id"], "calibrated")
        self._seed_run(store, metrics, "r1", tid, created_at=1000.0)

        cov = build_calibration_coverage(pid, as_of="2026-06-10")
        by = {c["channel"]: c for c in cov["channels"]}
        assert by["TV"]["tier"] == "calibrated"
        assert not by["TV"]["retest_due"]
        # the decayed EIG is still reported so the UI can show the trajectory
        assert by["TV"]["eig_decayed"] is not None

    def test_in_flight_experiment_wins_the_tier(self, store, metrics):
        """A running test (or a readout awaiting calibration) shows the
        channel as 'running' — the live edge of the program — and an
        evidence-backed running channel still counts as covered."""
        from mmm_framework.api.history import build_calibration_coverage

        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s1", project_id=pid)["thread_id"]
        # TV: old calibration + a fresh running re-test
        e1 = store.upsert_experiment(
            channel="TV", project_id=pid, status="completed", end_date="2024-01-01"
        )
        store.transition_experiment(e1["id"], "calibrated")
        store.upsert_experiment(
            channel="TV", project_id=pid, status="running", start_date="2026-06-01"
        )
        # Digital: readout in, awaiting calibration (no prior evidence)
        store.upsert_experiment(
            channel="Digital",
            project_id=pid,
            status="completed",
            value=1.2,
            se=0.2,
            end_date="2026-06-08",
        )
        self._seed_run(store, metrics, "r1", tid, created_at=1000.0)

        cov = build_calibration_coverage(pid, as_of="2026-06-10")
        by = {c["channel"]: c for c in cov["channels"]}
        assert by["TV"]["tier"] == "running"
        assert by["TV"]["in_flight_status"] == "running"
        assert by["TV"]["in_flight_started"] == "2026-06-01"
        assert by["Digital"]["tier"] == "running"
        assert by["Digital"]["in_flight_status"] == "completed"
        # TV (has evidence) counts as covered; Digital (no evidence yet) doesn't
        assert cov["coverage_pct"] == pytest.approx(50.0)

    def test_priorities_payload_decay_and_stale_flag(self, store, metrics):
        from mmm_framework.api.history import build_priorities_payload

        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s1", project_id=pid)["thread_id"]
        assert build_priorities_payload(pid) is None  # no metrics yet

        e = store.upsert_experiment(
            channel="TV", project_id=pid, status="completed", end_date="2024-01-01"
        )
        store.transition_experiment(e["id"], "calibrated")
        self._seed_run(store, metrics, "r1", tid, created_at=1000.0)

        out = build_priorities_payload(pid, as_of="2026-06-10")
        assert out["run_id"] == "r1"
        by = {c["channel"]: c for c in out["channels"]}
        assert by["TV"]["retest_due"] and by["TV"]["eig_decayed"] > by["TV"]["eig"]
        assert by["Digital"]["eig_decayed"] is None
        assert set(out["matrix"]) <= {
            "test_now",
            "learn_cheaply",
            "monitor",
            "deprioritize",
        }
        assert out["stale"] is False  # no newer model_run artifact

        # a newer fit without metrics flags the payload as stale
        store.add_artifact(tid, "model_run", {"run_id": "r2"})
        out2 = build_priorities_payload(pid, as_of="2026-06-10")
        assert out2["stale"] is True


@pytest.mark.asyncio
class TestEndpoints:
    async def test_history_and_coverage_and_priorities(self, store, metrics):
        import json

        from mmm_framework.api import main as M
        from mmm_framework.api.history import persist_run_metrics

        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s1", project_id=pid)["thread_id"]

        with pytest.raises(HTTPException) as exc:
            await M.project_history_endpoint("nope")
        assert exc.value.status_code == 404

        # no metrics yet -> priorities endpoint 404s with an actionable hint
        with pytest.raises(HTTPException) as exc:
            await M.experiment_priorities_endpoint(pid)
        assert exc.value.status_code == 404
        assert "backfill" in exc.value.detail

        persist_run_metrics({"run_id": "r1", "metrics": metrics}, tid)
        hist = json.loads((await M.project_history_endpoint(pid)).body)
        assert hist["runs"][0]["run_id"] == "r1"
        cov = json.loads((await M.calibration_coverage_endpoint(pid)).body)
        assert {c["channel"] for c in cov["channels"]} == {"TV", "Digital"}
        pri = json.loads((await M.experiment_priorities_endpoint(pid)).body)
        assert pri["run_id"] == "r1" and len(pri["channels"]) == 2


class TestBackfill:
    def test_skips_unrecoverable_runs_and_respects_existing(self, store, tmp_path):
        from mmm_framework.api.backfill import backfill_run_metrics

        pid = store.create_project("P")["project_id"]
        tid = store.create_session("s1", project_id=pid)["thread_id"]
        # run with an empty model dir -> skipped, with the reason named
        empty = tmp_path / "mmm_models" / "run_a"
        empty.mkdir(parents=True)
        store.add_artifact(
            tid,
            "model_run",
            {
                "run_id": "run_a",
                "model_path": str(empty),
                "dataset_path": str(tmp_path / "missing.csv"),
                "spec": {"kpi": "Sales"},
            },
        )
        # run that already has metrics -> exists
        store.record_run_metrics(
            "run_b", {"schema_version": 1, "channels": {}}, project_id=pid
        )
        store.add_artifact(tid, "model_run", {"run_id": "run_b"})

        report = {r["run_id"]: r for r in backfill_run_metrics(pid)}
        assert report["run_a"]["status"] == "skipped"
        assert "model dir missing/empty" in report["run_a"]["detail"]
        assert report["run_b"]["status"] == "exists"
