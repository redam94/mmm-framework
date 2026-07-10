"""Tests for question + ts provenance stamped on persisted session artifacts.

The ``/chat`` SSE generator wraps every artifact payload it persists in
``_provenanced(payload, question)`` so the Library/Artifacts panel can label
and group artifacts by the question they answer. The fields are purely
additive: ``created_at`` remains the fallback ordering key and nothing may
assume ``question``/``ts`` exist.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the session store + workspace at a temp location."""
    monkeypatch.setenv("MMM_AGENT_WORKSPACE", str(tmp_path / "ws"))
    from mmm_framework.api import sessions as ss

    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.db")
    ss.init_db()
    return ss


# ── _provenanced unit tests ──────────────────────────────────────────────────


def test_provenanced_stamps_question_and_ts():
    from mmm_framework.api.main import _provenanced

    payload = {"path": "report.html"}
    out = _provenanced(payload, "What drove sales in Q3?")

    assert out["question"] == "What drove sales in Q3?"
    assert isinstance(out["ts"], float)
    # Existing keys are preserved.
    assert out["path"] == "report.html"


def test_provenanced_does_not_mutate_original_payload():
    from mmm_framework.api.main import _provenanced

    payload = {"run_id": "abc123"}
    _provenanced(payload, "How should I allocate budget?")

    assert payload == {"run_id": "abc123"}


def test_provenanced_truncates_question_to_200_chars():
    from mmm_framework.api.main import _provenanced

    out = _provenanced({}, "x" * 500)

    assert len(out["question"]) == 200
    assert out["question"] == "x" * 200


def test_provenanced_none_question_omits_key_but_keeps_ts():
    from mmm_framework.api.main import _provenanced

    out = _provenanced({"a": 1}, None)

    assert "question" not in out
    assert isinstance(out["ts"], float)
    assert out["a"] == 1


def test_provenanced_empty_question_omits_key():
    from mmm_framework.api.main import _provenanced

    out = _provenanced({}, "")

    assert "question" not in out


def test_helpers_pass_non_dict_payloads_through():
    """Defensive tolerance: corrupted/legacy non-dict payloads never raise.

    The hydrate loop runs _strip_provenance over every stored artifact on
    every /chat request BEFORE the stream's error handling — a raise there
    would brick the session's chat.
    """
    from mmm_framework.api.main import _provenanced, _strip_provenance

    assert _provenanced(None, "q") is None
    assert _provenanced(["not", "a", "dict"], "q") == ["not", "a", "dict"]
    assert _strip_provenance(None) is None
    assert _strip_provenance([1, 2]) == [1, 2]


# ── dedup contract (plan-kind content hashes) ────────────────────────────────


def test_strip_provenance_restores_content_hash_equality():
    """A stamped stored payload must hash like the raw streamed payload.

    The /chat generator dedupes experiment_design/budget_optimization
    artifacts by content hash; both hash sites (persist + hydrate) apply
    ``_strip_provenance`` first, so for a clean plan the stored (stamped)
    payload must strip back to hash-equality with the raw plan.
    """
    from mmm_framework.api.main import _payload_hash, _provenanced, _strip_provenance

    plan = {"channel": "TV", "budget": 50_000, "weeks": 8}
    stamped = _provenanced(plan, "Design a holdout for TV")

    assert _payload_hash(_strip_provenance(stamped)) == _payload_hash(plan)


def test_strip_provenance_is_noop_on_unstamped_payload():
    from mmm_framework.api.main import _strip_provenance

    payload = {"path": "old_report.html"}

    assert _strip_provenance(payload) == payload


def test_plan_dedup_hash_symmetric_even_with_native_provenance_keys():
    """Both dedup-hash sites strip, so a plan natively carrying a reserved
    key still yields matching persist-side and hydrate-side dedup keys."""
    from mmm_framework.api.main import _payload_hash, _provenanced, _strip_provenance

    plan = {"ranking": ["TV"], "question": "native", "ts": 123.0}
    stored = _provenanced(plan, "Design a holdout for TV")

    # Persist-side key hashes _strip_provenance(raw plan); hydrate-side key
    # hashes _strip_provenance(stored payload). They must agree.
    assert _payload_hash(_strip_provenance(plan)) == _payload_hash(
        _strip_provenance(stored)
    )


# ── store round-trip ─────────────────────────────────────────────────────────


def test_add_artifact_roundtrip_carries_question_and_ts(store):
    from mmm_framework.api.main import _provenanced

    tid = "thread-question-stamp"
    payload = _provenanced(
        {"path": "agent_report.html"}, "Which channel has the best ROI?"
    )
    store.add_artifact(tid, "report", payload)

    rows = store.list_artifacts(tid)
    assert len(rows) == 1
    stored = rows[0]["payload"]
    assert stored["path"] == "agent_report.html"
    assert stored["question"] == "Which channel has the best ROI?"
    assert isinstance(stored["ts"], float)
    # created_at (the fallback ordering key) is untouched.
    assert rows[0]["created_at"]


def test_add_artifact_roundtrip_without_provenance_still_works(store):
    """Nothing assumes question/ts exist — legacy payloads stay valid."""
    tid = "thread-legacy"
    store.add_artifact(tid, "report", {"path": "legacy.html"})

    rows = store.list_artifacts(tid)
    assert rows[0]["payload"] == {"path": "legacy.html"}
