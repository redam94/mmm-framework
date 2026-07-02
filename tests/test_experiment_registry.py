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


class TestUpsertStateMachine:
    def test_create_with_calibrated_rejected(self, store):
        with pytest.raises(ValueError, match="Illegal status for create"):
            _create(store, status="calibrated")

    def test_create_with_completed_backfills_history_chain(self, store):
        exp = _create(store, status="completed", value=1.2, se=0.3)
        assert exp["status"] == "completed"
        assert [h["status"] for h in exp["status_history"]] == [
            "draft",
            "planned",
            "running",
            "completed",
        ]
        assert all(h["note"] == "backfilled on create" for h in exp["status_history"])

    def test_update_illegal_status_jump_raises(self, store):
        exp = _create(store, status="planned")
        with pytest.raises(ValueError, match="Illegal transition planned->calibrated"):
            store.upsert_experiment(experiment_id=exp["id"], status="calibrated")
        assert store.get_experiment(exp["id"])["status"] == "planned"

    def test_update_to_calibrated_always_raises_even_from_completed(self, store):
        # completed->calibrated is a legal transition_experiment edge, but
        # calibration is fit-close-out-only: upsert must never mint it.
        exp = _create(store, status="completed")
        with pytest.raises(
            ValueError, match="Illegal transition completed->calibrated"
        ):
            store.upsert_experiment(experiment_id=exp["id"], status="calibrated")
        assert store.get_experiment(exp["id"])["status"] == "completed"

    def test_update_backward_status_unreachable_raises(self, store):
        exp = _create(store, status="completed")
        exp = store.transition_experiment(exp["id"], "calibrated")
        with pytest.raises(ValueError, match="Illegal transition calibrated->draft"):
            store.upsert_experiment(experiment_id=exp["id"], status="draft")
        assert store.get_experiment(exp["id"])["status"] == "calibrated"

    def test_update_multihop_forward_status_backfills_history(self, store):
        # log_experiment's documented one-call results flow: an existing
        # planned row updated straight to completed walks the legal path and
        # records the skipped 'running' hop in the audit trail.
        exp = _create(store, status="planned")
        exp = store.upsert_experiment(
            experiment_id=exp["id"], status="completed", value=1.2, se=0.3
        )
        assert exp["status"] == "completed"
        assert exp["value"] == 1.2 and exp["se"] == 0.3
        assert [h["status"] for h in exp["status_history"]] == [
            "draft",
            "planned",
            "running",
            "completed",
        ]
        hop = exp["status_history"][-2]
        assert hop["status"] == "running" and hop["note"] == "backfilled via upsert"
        assert exp["status_history"][-1]["note"] == "via upsert"

    def test_update_legal_status_change_appends_history(self, store):
        exp = _create(store, status="planned")
        exp = store.upsert_experiment(experiment_id=exp["id"], status="running")
        assert exp["status"] == "running"
        last = exp["status_history"][-1]
        assert last["status"] == "running" and last["note"] == "via upsert"

    def test_update_same_status_is_silent_noop(self, store):
        exp = _create(store, status="planned")
        n = len(exp["status_history"])
        exp = store.upsert_experiment(
            experiment_id=exp["id"], status="planned", notes="tweak"
        )
        assert exp["notes"] == "tweak"
        assert len(exp["status_history"]) == n


class TestCalibratedRowGuard:
    """A calibrated experiment's measurement fields are locked behind
    allow_calibrated_edit (the record_experiment_readout sanctioned path) —
    raw upserts must not silently mutate likelihood-feeding evidence."""

    def _calibrated(self, store):
        exp = _create(
            store,
            status="completed",
            value=2.0,
            se=0.2,
            estimand="roas",
            readout={"value": 2.0, "se": 0.2, "spend_per_period": 5000.0},
        )
        return store.transition_experiment(exp["id"], "calibrated")

    def test_measurement_edit_without_flag_raises(self, store):
        exp = self._calibrated(store)
        with pytest.raises(ValueError, match="Illegal update"):
            store.upsert_experiment(experiment_id=exp["id"], value=9.9)
        after = store.get_experiment(exp["id"])
        assert after["value"] == 2.0 and after["status"] == "calibrated"

    def test_readout_edit_without_flag_raises(self, store):
        exp = self._calibrated(store)
        with pytest.raises(ValueError, match="Illegal update"):
            store.upsert_experiment(
                experiment_id=exp["id"],
                readout={"value": 2.0, "se": 0.2, "spend_per_period": 8000.0},
            )
        after = store.get_experiment(exp["id"])
        assert after["readout"]["spend_per_period"] == 5000.0

    def test_flag_allows_sanctioned_edit(self, store):
        exp = self._calibrated(store)
        exp = store.upsert_experiment(
            experiment_id=exp["id"], value=3.0, allow_calibrated_edit=True
        )
        assert exp["value"] == 3.0 and exp["status"] == "calibrated"

    def test_unchanged_resend_is_not_an_edit(self, store):
        exp = self._calibrated(store)
        out = store.upsert_experiment(
            experiment_id=exp["id"], value=2.0, se=0.2, estimand="roas"
        )
        assert out["value"] == 2.0

    def test_non_measurement_fields_stay_editable(self, store):
        exp = self._calibrated(store)
        out = store.upsert_experiment(
            experiment_id=exp["id"],
            notes="post-hoc note",
            priority={"eig": 0.1},
            calibrated_run_id="r2",
        )
        assert out["notes"] == "post-hoc note"
        assert out["calibrated_run_id"] == "r2"


class TestAppendExperimentEvent:
    def test_appends_event_with_changed_diff(self, store):
        exp = _create(store, status="completed", value=1.0, se=0.2)
        store.append_experiment_event(
            exp["id"], note="readout edited", changed={"value": [1.0, 1.4]}
        )
        exp = store.get_experiment(exp["id"])
        last = exp["status_history"][-1]
        assert last["status"] == "completed"  # status unchanged by an event
        assert last["note"] == "readout edited"
        assert last["changed"] == {"value": [1.0, 1.4]}

    def test_unknown_id_raises(self, store):
        with pytest.raises(ValueError, match="Unknown experiment id"):
            store.append_experiment_event("nope", note="x")

    def test_tolerates_null_history(self, tmp_path, monkeypatch):
        """A pre-lifecycle row (NULL status_history_json) gets its history
        synthesized from the current status, like transition_experiment."""
        from mmm_framework.api import sessions as S

        db = tmp_path / "sessions.db"
        with sqlite3.connect(db) as c:
            c.execute("""
                CREATE TABLE experiments (
                    id TEXT PRIMARY KEY, project_id TEXT, thread_id TEXT,
                    channel TEXT NOT NULL, design_type TEXT,
                    status TEXT NOT NULL DEFAULT 'planned',
                    start_date TEXT, end_date TEXT, estimand TEXT,
                    value REAL, se REAL, notes TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                )
                """)
            c.execute(
                "INSERT INTO experiments (id, project_id, channel, status,"
                " created_at, updated_at) VALUES ('e1', 'p1', 'TV', 'completed',"
                " 1.0, 1.0)"
            )
        monkeypatch.setattr(S, "DB_PATH", db)
        S.init_db()

        S.append_experiment_event("e1", note="edited", changed={"se": [None, 0.3]})
        exp = S.get_experiment("e1")
        assert [h["status"] for h in exp["status_history"]] == [
            "completed",
            "completed",
        ]
        assert exp["status_history"][-1]["changed"] == {"se": [None, 0.3]}


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
            c.execute("""
                CREATE TABLE experiments (
                    id TEXT PRIMARY KEY, project_id TEXT, thread_id TEXT,
                    channel TEXT NOT NULL, design_type TEXT,
                    status TEXT NOT NULL DEFAULT 'planned',
                    start_date TEXT, end_date TEXT, estimand TEXT,
                    value REAL, se REAL, notes TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                )
                """)
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
