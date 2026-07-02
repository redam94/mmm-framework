"""Sessions-store tests for the continuous-learning tables (wiring §3.2/§3.3):
learning_programs / learning_waves CRUD, status validation, cascade delete,
and the experiments registry's new nullable ``subchannel`` column."""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


# ── learning_programs ─────────────────────────────────────────────────────────


def test_program_create_get_list_update(store):
    prog = store.create_learning_program(
        project_id="p1",
        thread_id="t1",
        name="Q3 geo program",
        channels=["Chatter", "Pulse"],
        config={"budget": 280000, "channels": ["Chatter", "Pulse"]},
    )
    assert prog["status"] == "active"
    assert prog["channels"] == ["Chatter", "Pulse"]
    assert prog["config"]["budget"] == 280000
    assert prog["summary"] is None

    got = store.get_learning_program(prog["id"])
    assert got["name"] == "Q3 geo program" and got["project_id"] == "p1"

    # list is project-scoped + status-filterable
    store.create_learning_program(
        project_id="p2", channels=["X"], config={"channels": ["X"]}
    )
    assert [p["id"] for p in store.list_learning_programs("p1")] == [prog["id"]]
    assert store.list_learning_programs("p1", status="stopped") == []

    upd = store.update_learning_program(
        prog["id"],
        status="stopped",
        state_path="/tmp/state.npz",
        summary={"schema_version": 1, "regret": {"stop": True}},
    )
    assert upd["status"] == "stopped"
    assert upd["state_path"] == "/tmp/state.npz"
    assert upd["summary"]["regret"]["stop"] is True
    # untouched fields survive a partial update
    assert upd["channels"] == ["Chatter", "Pulse"]


def test_program_status_validation(store):
    with pytest.raises(ValueError, match="Invalid status"):
        store.create_learning_program(
            project_id="p1", channels=["A"], config={}, status="exploded"
        )
    prog = store.create_learning_program(project_id="p1", channels=["A"], config={})
    with pytest.raises(ValueError, match="Invalid status"):
        store.update_learning_program(prog["id"], status="paused")
    with pytest.raises(ValueError, match="Unknown learning program"):
        store.update_learning_program("nope", status="stopped")
    with pytest.raises(ValueError, match="channels is required"):
        store.create_learning_program(project_id="p1", channels=[], config={})


def test_delete_program_cascades_waves(store):
    prog = store.create_learning_program(project_id="p1", channels=["A"], config={})
    w1 = store.add_learning_wave(prog["id"], project_id="p1", design={"n_cells": 4})
    store.add_learning_wave(prog["id"], project_id="p1", status="ingested")
    assert len(store.list_learning_waves(prog["id"])) == 2

    assert store.delete_learning_program(prog["id"]) is True
    assert store.get_learning_program(prog["id"]) is None
    assert store.list_learning_waves(prog["id"]) == []
    assert store.get_learning_wave(w1["id"]) is None
    # deleting again reports absence
    assert store.delete_learning_program(prog["id"]) is False


def test_wave_auto_index_json_round_trip_and_update(store):
    prog = store.create_learning_program(project_id="p1", channels=["A"], config={})
    w0 = store.add_learning_wave(
        prog["id"],
        project_id="p1",
        design={"cells_dollars": [[1.0]], "n_cells": 1},
    )
    w1 = store.add_learning_wave(
        prog["id"],
        project_id="p1",
        status="ingested",
        source="experiment_import",
        observations={"imported": 2},
        snapshot={"schema_version": 1},
        experiment_ids=["e1", "e2"],
    )
    assert (w0["wave_index"], w1["wave_index"]) == (0, 1)
    assert w1["experiment_ids"] == ["e1", "e2"]
    assert w1["snapshot"] == {"schema_version": 1}
    assert w1["observations"] == {"imported": 2}
    # ordered oldest-first by wave_index
    assert [w["id"] for w in store.list_learning_waves(prog["id"])] == [
        w0["id"],
        w1["id"],
    ]

    upd = store.update_learning_wave(
        w0["id"], status="ingested", snapshot={"schema_version": 1, "ok": True}
    )
    assert upd["status"] == "ingested" and upd["snapshot"]["ok"] is True
    with pytest.raises(ValueError, match="Invalid wave status"):
        store.update_learning_wave(w0["id"], status="finished")
    with pytest.raises(ValueError, match="Unknown learning wave"):
        store.update_learning_wave("nope", status="ingested")
    with pytest.raises(ValueError, match="Unknown learning program"):
        store.add_learning_wave("nope")
    with pytest.raises(ValueError, match="Invalid wave status"):
        store.add_learning_wave(prog["id"], status="finished")


def test_record_ingested_wave_resolves_designed_row(store):
    prog = store.create_learning_program(project_id="p1", channels=["A"], config={})

    # no open design -> a rows-ingest appends a fresh 'ingested' row
    w0 = store.record_ingested_wave(
        prog["id"], project_id="p1", source="wave", observations={"n_rows": 2}
    )
    assert w0["status"] == "ingested" and w0["wave_index"] == 0

    # design -> ingest RESOLVES the same row (one board row per real wave)
    d = store.add_learning_wave(
        prog["id"],
        project_id="p1",
        status="designed",
        source="wave",
        design={"n_cells": 4},
    )
    w1 = store.record_ingested_wave(
        prog["id"],
        project_id="p1",
        source="wave",
        observations={"n_rows": 4},
        snapshot={"schema_version": 1},
    )
    assert w1["id"] == d["id"] and w1["status"] == "ingested"
    assert w1["design"] == {"n_cells": 4}  # the design stays joined to results
    assert w1["observations"] == {"n_rows": 4}
    assert len(store.list_learning_waves(prog["id"])) == 2

    # experiment imports ALWAYS append — an open designed row stays open
    d2 = store.add_learning_wave(
        prog["id"], project_id="p1", status="designed", source="wave"
    )
    w2 = store.record_ingested_wave(
        prog["id"],
        project_id="p1",
        source="experiment_import",
        experiment_ids=["e1"],
    )
    assert w2["id"] != d2["id"] and w2["source"] == "experiment_import"
    assert store.get_learning_wave(d2["id"])["status"] == "designed"
    assert w2["experiment_ids"] == ["e1"]


# ── experiments.subchannel ────────────────────────────────────────────────────


def test_experiment_subchannel_round_trip_and_filter(store):
    exp = store.upsert_experiment(
        project_id="p1", channel="Search", subchannel="Brand", status="draft"
    )
    assert exp["subchannel"] == "Brand"
    assert store.get_experiment(exp["id"])["subchannel"] == "Brand"

    # a channel-level row has NULL subchannel
    plain = store.upsert_experiment(project_id="p1", channel="Search")
    assert plain["subchannel"] is None

    # update-in-place through the upsert path
    upd = store.upsert_experiment(experiment_id=plain["id"], subchannel="NonBrand")
    assert upd["subchannel"] == "NonBrand"

    # list filter
    brand = store.list_experiments(project_id="p1", subchannel="Brand")
    assert [e["id"] for e in brand] == [exp["id"]]
    assert (
        store.list_experiments(project_id="p1", channel="Search", subchannel="Nope")
        == []
    )
