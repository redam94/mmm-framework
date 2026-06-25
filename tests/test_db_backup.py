"""Online backup/restore of the sessions DB (Phase 3 / P1c)."""

from __future__ import annotations

import sqlite3

import pytest

from mmm_framework.api import backup as backup_mod


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from mmm_framework.api import sessions as S

    monkeypatch.setattr(S, "DB_PATH", tmp_path / "sessions.db")
    S.init_db()
    return S


def test_backup_then_restore_roundtrip(store, tmp_path):
    store.record_run_metrics(
        "r1", {"schema_version": 1, "channels": {}}, project_id="p"
    )
    dest = tmp_path / "backup.db"
    backup_mod.backup_db(dest)
    assert dest.exists()

    # Simulate data loss on the live DB.
    with sqlite3.connect(str(store.DB_PATH)) as c:
        c.execute("DELETE FROM run_metrics")
    assert store.get_run_metrics("r1") is None

    # Restore brings it back.
    backup_mod.restore_db(dest)
    assert store.get_run_metrics("r1") is not None


def test_backup_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        backup_mod.backup_db(tmp_path / "out.db", db_path=tmp_path / "missing.db")


def test_restore_missing_backup_raises(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        backup_mod.restore_db(tmp_path / "no-such-backup.db")
