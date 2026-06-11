"""Integration tests for the server-authoritative model-spec endpoints.

Exercises ``PATCH /spec/{thread_id}`` (manual edit + diff-based auto-lock) and
``POST /spec/{thread_id}/resolve`` (confirm/decline an LLM-proposed change to a
locked field), including the decline-memory note written into the thread so the
LLM doesn't re-propose a rejected change.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver


def _body(resp) -> dict:
    return json.loads(resp.body)


@pytest.fixture()
def app_main(monkeypatch):
    """main.py with its module-level checkpointer swapped for an in-memory one."""
    from mmm_framework.api import main as M

    monkeypatch.setattr(M, "memory", MemorySaver())
    return M


async def _seed(M, tid, spec, **extra):
    # Seed a *terminal* thread (a completed turn ends with an AIMessage that has
    # no tool_calls → routes to END), mirroring how the spec endpoints are only
    # ever reached after the agent has already configured a model. as_node="agent"
    # makes the bare aupdate_state in the endpoints unambiguous, as in production.
    g = M._admin_graph()
    cfg = {"configurable": {"thread_id": tid}}
    await g.aupdate_state(
        cfg,
        {
            "messages": [AIMessage(content="seeded")],
            "model_spec": spec,
            "dashboard_data": {},
            **extra,
        },
        as_node="agent",
    )


@pytest.mark.asyncio
async def test_patch_locks_only_changed_fields(app_main):
    M = app_main
    tid = "t-patch"
    await _seed(M, tid, {"kpi": "Sales", "inference": {"draws": 1000, "chains": 4}})

    resp = await M.update_spec(
        tid,
        M.SpecUpdateRequest(
            model_spec={"kpi": "Sales", "inference": {"draws": 2000, "chains": 4}}
        ),
    )
    body = _body(resp)
    assert body["locked_fields"] == ["inference.draws"]  # chains unchanged → not locked

    # the write is server-authoritative: it lands in agent state + dashboard
    snap = await M._admin_graph().aget_state({"configurable": {"thread_id": tid}})
    assert snap.values["model_spec"]["inference"]["draws"] == 2000
    assert snap.values["locked_fields"] == ["inference.draws"]
    assert snap.values["dashboard_data"]["locked_fields"] == ["inference.draws"]


@pytest.mark.asyncio
async def test_patch_honors_explicit_lock_paths(app_main):
    # The editor sends the precise leaves the user touched; the server must lock
    # exactly those and NOT over-lock the rest of a fully-materialized spec.
    M = app_main
    tid = "t-explicit"
    await _seed(M, tid, {"kpi": "Sales", "inference": {"draws": 1000}})

    full = {
        "kpi": "Sales",
        "inference": {"draws": 2000, "chains": 4, "tune": 1000, "target_accept": 0.85},
        "trend": {"type": "linear"},
        "seasonality": {"yearly": 0},
    }
    resp = await M.update_spec(
        tid, M.SpecUpdateRequest(model_spec=full, lock_paths=["inference.draws"])
    )
    # only the one touched leaf is locked, despite many materialized defaults
    assert _body(resp)["locked_fields"] == ["inference.draws"]


@pytest.mark.asyncio
async def test_unlock_hands_field_back(app_main):
    M = app_main
    tid = "t-unlock"
    await _seed(
        M,
        tid,
        {"kpi": "Sales", "inference": {"draws": 2000}},
        locked_fields=["inference.draws"],
    )
    resp = await M.update_spec(
        tid,
        M.SpecUpdateRequest(
            model_spec={"kpi": "Sales", "inference": {"draws": 2000}},
            unlock_paths=["inference.draws"],
        ),
    )
    assert _body(resp)["locked_fields"] == []


@pytest.mark.asyncio
async def test_resolve_reject_keeps_value_and_writes_decline_note(app_main):
    M = app_main
    tid = "t-reject"
    await _seed(
        M,
        tid,
        {"kpi": "Sales", "inference": {"draws": 2000}},
        locked_fields=["inference.draws"],
        pending_spec_changes=[
            {
                "path": "inference.draws",
                "current": 2000,
                "proposed": 4000,
                "reason": "more draws",
                "tool_call_id": "t1",
            }
        ],
    )

    resp = await M.resolve_spec_change(
        tid, M.SpecResolveRequest(path="inference.draws", action="reject")
    )
    body = _body(resp)
    assert body["model_spec"]["inference"]["draws"] == 2000  # user's value kept
    assert body["pending_spec_changes"] == []  # cleared
    assert body["locked_fields"] == ["inference.draws"]  # stays locked

    # decline-memory: a note is appended so the LLM won't re-propose
    snap = await M._admin_graph().aget_state({"configurable": {"thread_id": tid}})
    notes = [m.content for m in snap.values["messages"] if "DECLINED" in str(m.content)]
    assert notes and "inference.draws" in notes[0]


@pytest.mark.asyncio
async def test_resolve_approve_applies_and_keeps_locked(app_main):
    M = app_main
    tid = "t-approve"
    await _seed(
        M,
        tid,
        {"kpi": "Sales", "inference": {"draws": 2000}},
        locked_fields=["inference.draws"],
        pending_spec_changes=[
            {"path": "inference.draws", "current": 2000, "proposed": 4000}
        ],
    )

    resp = await M.resolve_spec_change(
        tid, M.SpecResolveRequest(path="inference.draws", action="approve")
    )
    body = _body(resp)
    assert body["model_spec"]["inference"]["draws"] == 4000  # proposal applied
    assert body["locked_fields"] == ["inference.draws"]  # stays locked at new value
    assert body["pending_spec_changes"] == []

    snap = await M._admin_graph().aget_state({"configurable": {"thread_id": tid}})
    notes = [m.content for m in snap.values["messages"] if "APPROVED" in str(m.content)]
    assert notes


@pytest.mark.asyncio
async def test_resolve_unknown_path_is_404(app_main):
    M = app_main
    tid = "t-404"
    await _seed(M, tid, {"kpi": "Sales"}, pending_spec_changes=[])
    resp = await M.resolve_spec_change(
        tid, M.SpecResolveRequest(path="nope.path", action="reject")
    )
    assert resp.status_code == 404
