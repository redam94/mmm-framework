"""Tests for the experiment lifecycle registry (api/sessions.py): validated
status transitions with an append-only audit trail, pre-registration stamping,
readout/calibration merges, the per-channel evidence map, and the schema
migration of pre-lifecycle installs."""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def _create(store, **kw):
    defaults = dict(channel="TV", project_id="p1", status="draft")
    defaults.update(kw)
    return store.upsert_experiment(**defaults)


class TestTransitions:
    def test_happy_path_full_lifecycle(self, store):
        exp = _create(store)
        assert exp["status"] == "draft"
        assert exp["status_history"][0]["status"] == "draft"

        exp = store.transition_experiment(exp["id"], "planned", note="pre-registered")
        assert exp["status"] == "planned"
        assert exp["preregistered_at"] is not None

        exp = store.transition_experiment(exp["id"], "running")
        exp = store.transition_experiment(
            exp["id"],
            "completed",
            value=2.1,
            se=0.3,
            estimand="roas",
            start_date="2026-01-05",
            end_date="2026-03-01",
            readout={"method": "geo_holdout_did", "lift": 2.1},
        )
        assert exp["value"] == 2.1 and exp["se"] == 0.3
        assert exp["readout"]["method"] == "geo_holdout_did"

        exp = store.transition_experiment(
            exp["id"], "calibrated", calibrated_run_id="run_42"
        )
        assert exp["status"] == "calibrated"
        assert exp["calibrated_run_id"] == "run_42"
        # audit trail covers every move, in order
        assert [h["status"] for h in exp["status_history"]] == [
            "draft",
            "planned",
            "running",
            "completed",
            "calibrated",
        ]
        assert exp["status_history"][1]["note"] == "pre-registered"

    def test_illegal_transitions_raise(self, store):
        exp = _create(store)
        with pytest.raises(ValueError, match="Illegal transition draft->running"):
            store.transition_experiment(exp["id"], "running")
        with pytest.raises(ValueError, match="Illegal transition draft->calibrated"):
            store.transition_experiment(exp["id"], "calibrated")
        # terminal states allow nothing
        store.transition_experiment(exp["id"], "abandoned")
        with pytest.raises(ValueError, match="Illegal transition abandoned->"):
            store.transition_experiment(exp["id"], "planned")

    def test_unknown_id_and_bad_status(self, store):
        with pytest.raises(ValueError, match="Unknown experiment id"):
            store.transition_experiment("nope", "planned")
        exp = _create(store)
        with pytest.raises(ValueError, match="Invalid status"):
            store.transition_experiment(exp["id"], "exploded")

    def test_preregistered_at_only_on_draft_to_planned(self, store):
        exp = _create(store, status="planned")
        exp = store.transition_experiment(exp["id"], "running")
        assert exp["preregistered_at"] is None  # never went through draft->planned


class TestUpsertExtensions:
    def test_create_with_lifecycle_payloads(self, store):
        exp = _create(
            store,
            recommending_run_id="run_1",
            design={"design_type": "geo_holdout", "duration_weeks": 8},
            priority={"eig": 0.4, "evoi": 1200.0, "quadrant": "test_now"},
        )
        assert exp["recommending_run_id"] == "run_1"
        assert exp["design"]["duration_weeks"] == 8
        assert exp["priority"]["quadrant"] == "test_now"

    def test_channel_filter(self, store):
        _create(store, channel="TV")
        _create(store, channel="Digital")
        assert [e["channel"] for e in store.list_experiments(channel="TV")] == ["TV"]

    def test_legacy_statuses_still_accepted(self, store):
        exp = _create(store, status="cancelled")
        assert exp["status"] == "cancelled"


class TestEvidenceMap:
    def test_latest_calibrated_per_channel(self, store):
        def calibrated(channel, end_date, run_id):
            e = _create(store, channel=channel, status="completed")
            return store.transition_experiment(
                e["id"], "calibrated", calibrated_run_id=run_id, note=end_date
            )

        # two calibrated TV experiments; newer end_date must win
        e_old = _create(store, channel="TV", status="completed", end_date="2025-06-01")
        store.transition_experiment(e_old["id"], "calibrated", calibrated_run_id="r1")
        e_new = _create(store, channel="TV", status="completed", end_date="2026-01-01")
        store.transition_experiment(e_new["id"], "calibrated", calibrated_run_id="r2")
        # a non-calibrated Digital experiment must not appear
        _create(store, channel="Digital", status="running")

        ev = store.latest_calibrated_evidence("p1")
        assert set(ev) == {"TV"}
        assert ev["TV"]["experiment_id"] == e_new["id"]
        assert ev["TV"]["calibrated_run_id"] == "r2"


class TestMigration:
    def test_pre_lifecycle_db_migrates_and_reads(self, tmp_path, monkeypatch):
        """A DB created before the lifecycle columns must gain them on init_db
        and its legacy rows must read back with None/[] lifecycle fields."""
        from mmm_framework.api import sessions as S

        db = tmp_path / "sessions.db"
        with sqlite3.connect(db) as c:
            c.execute(
                """
                CREATE TABLE experiments (
                    id TEXT PRIMARY KEY, project_id TEXT, thread_id TEXT,
                    channel TEXT NOT NULL, design_type TEXT,
                    status TEXT NOT NULL DEFAULT 'planned',
                    start_date TEXT, end_date TEXT, estimand TEXT,
                    value REAL, se REAL, notes TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                )
                """
            )
            c.execute(
                "INSERT INTO experiments (id, project_id, channel, status,"
                " created_at, updated_at) VALUES ('e1', 'p1', 'TV', 'completed',"
                " 1.0, 1.0)"
            )
        monkeypatch.setattr(S, "DB_PATH", db)
        S.init_db()

        exp = S.get_experiment("e1")
        assert exp["channel"] == "TV"
        assert exp["design"] is None and exp["status_history"] == []
        # lifecycle now works on the legacy row
        exp = S.transition_experiment("e1", "calibrated", calibrated_run_id="r9")
        assert exp["calibrated_run_id"] == "r9"
        # the synthesized history starts from the legacy status
        assert [h["status"] for h in exp["status_history"]] == [
            "completed",
            "calibrated",
        ]
