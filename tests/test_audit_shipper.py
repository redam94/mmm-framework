"""P1 reliability — off-host audit shipper + observability snapshot."""

from __future__ import annotations

import json

import pytest

from mmm_framework.agents import audit_shipper


def _write_log(path, n, start=0):
    with open(path, "w") as f:
        for i in range(start, start + n):
            f.write(
                json.dumps({"seq": i, "event": "auth.login", "fields": {"org_id": "o"}})
                + "\n"
            )


def test_ship_pending_advances_cursor_and_idempotent(tmp_path):
    log = str(tmp_path / "audit.jsonl")
    _write_log(log, 3)
    sent: list[dict] = []
    res = audit_shipper.ship_pending(log, lambda recs: sent.extend(recs))
    assert res == {"shipped": 3, "cursor": 2}
    assert [r["seq"] for r in sent] == [0, 1, 2]

    # second flush ships nothing (cursor persisted)
    sent.clear()
    assert audit_shipper.ship_pending(log, lambda recs: sent.extend(recs)) == {
        "shipped": 0,
        "cursor": 2,
    }
    assert sent == []

    # appended records ship next time
    with open(log, "a") as f:
        f.write(json.dumps({"seq": 3, "event": "auth.logout", "fields": {}}) + "\n")
    assert audit_shipper.ship_pending(log, lambda recs: sent.extend(recs)) == {
        "shipped": 1,
        "cursor": 3,
    }


def test_ship_failure_does_not_advance_cursor(tmp_path, monkeypatch):
    monkeypatch.delenv("MMM_AUDIT_SHIP_URL", raising=False)
    log = str(tmp_path / "audit.jsonl")
    _write_log(log, 2)

    def boom(_recs):
        raise RuntimeError("sink down")

    with pytest.raises(RuntimeError):
        audit_shipper.ship_pending(log, boom)
    # cursor un-advanced -> batch re-ships next flush (at-least-once)
    st = audit_shipper.ship_status(log)
    assert st["pending"] == 2 and st["cursor"] == -1


def test_ship_status_and_flush_noop_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("MMM_AUDIT_SHIP_URL", raising=False)
    log = str(tmp_path / "audit.jsonl")
    _write_log(log, 2)
    assert audit_shipper.ship_status(log) == {
        "configured": False,
        "cursor": -1,
        "pending": 2,
        "total": 2,
    }
    assert audit_shipper.flush_audit_to_remote(log) == {
        "configured": False,
        "shipped": 0,
    }


def test_system_health_shape(tmp_path, monkeypatch):
    from mmm_framework.api import observability
    from mmm_framework.api import sessions as ss

    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "s.db")
    ss.init_db()
    h = observability.system_health()
    assert h["ok"] is True
    assert "audit" in h
    assert h["fits"]["total"] == 0  # no fits in a fresh db
    assert h["fits"]["window_hours"] == 24.0
