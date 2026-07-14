"""Regression tests for the checkpoint-bloat fixes that resolved the recurring
``sqlite3.OperationalError: database is locked``.

Root cause (diagnosed 2026-07-13): the shared ``sessions.db`` had grown to
3.5 GB because LangGraph writes a full ``AgentState`` checkpoint on every graph
super-step and keeps every version forever. One session accumulated 400+
versions at ~7.5 MB each — dominated by multi-MB tool-message text — and on a
multi-GB WAL file each write + WAL fold-back grew slow enough that concurrent
writers blew past the 30 s ``busy_timeout`` and surfaced as "database is locked".

Two prevention layers, both exercised here:

  1. ``agents.graph._cap_tool_output`` — a single tool result can no longer push
     megabytes into the (per-checkpoint re-serialized) ``messages`` channel.
  2. ``api.main._prune_thread_checkpoints`` — each /chat turn bounds a thread's
     checkpoint history to ``MMM_CHECKPOINT_RETENTION`` (default 40) versions.
"""

import aiosqlite
import pytest
from langchain_core.messages import AIMessage, ToolMessage


# --------------------------------------------------------------------------- #
# Layer 1: oversized ToolMessage content is truncated; everything else is not.
# --------------------------------------------------------------------------- #
def test_cap_tool_output_truncates_only_oversized_tool_messages():
    from mmm_framework.agents.graph import _cap_tool_output, _TOOL_MESSAGE_MAX_CHARS

    big = "X" * (_TOOL_MESSAGE_MAX_CHARS + 50_000)
    small = "concise summary"
    result = {
        "messages": [
            ToolMessage(content=big, tool_call_id="a"),
            ToolMessage(content=small, tool_call_id="b"),
            # A non-ToolMessage of the same size must NOT be capped (only tool
            # results accumulate as the runaway growth; AI/human text is trimmed
            # for the LLM elsewhere and matters for conversation fidelity).
            AIMessage(content="Y" * (_TOOL_MESSAGE_MAX_CHARS + 1)),
        ]
    }
    out = _cap_tool_output(result)
    m_big, m_small, m_ai = out["messages"]

    assert len(m_big.content) < len(big)
    assert "truncated" in m_big.content
    assert m_big.content.startswith("X" * 100)  # head preserved
    assert m_small.content == small  # untouched
    assert len(m_ai.content) == _TOOL_MESSAGE_MAX_CHARS + 1  # untouched


def test_cap_tool_output_handles_list_and_empty_shapes():
    from mmm_framework.agents.graph import _cap_tool_output, _TOOL_MESSAGE_MAX_CHARS

    big = "Z" * (_TOOL_MESSAGE_MAX_CHARS + 50_000)
    # list shape (some ToolNode paths)
    capped = _cap_tool_output([ToolMessage(content=big, tool_call_id="c")])
    assert len(capped[0].content) < len(big)
    assert "truncated" in capped[0].content
    # None-safe / no messages
    assert _cap_tool_output({"messages": None}) == {"messages": None}
    assert _cap_tool_output({}) == {}


# --------------------------------------------------------------------------- #
# Layer 2: the per-thread checkpoint retention prune (exercises the real async
# function against a temp DB with the real LangGraph checkpoint/writes schema).
# --------------------------------------------------------------------------- #
_CK_SCHEMA = """
CREATE TABLE checkpoints (
    thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL, parent_checkpoint_id TEXT, type TEXT,
    checkpoint BLOB, metadata BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id));
CREATE TABLE writes (
    thread_id TEXT NOT NULL, checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL, task_id TEXT NOT NULL, idx INTEGER NOT NULL,
    channel TEXT NOT NULL, type TEXT, value BLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx));
"""


async def _seed(conn, thread, ns, n):
    """Insert n checkpoints (+ one write each) with lexically-ordered ids."""
    for i in range(n):
        ck = f"ck-{i:05d}"  # zero-padded => lexical DESC == newest-first
        await conn.execute(
            "INSERT INTO checkpoints(thread_id, checkpoint_ns, checkpoint_id, checkpoint)"
            " VALUES (?,?,?,?)",
            (thread, ns, ck, b"x"),
        )
        await conn.execute(
            "INSERT INTO writes(thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel)"
            " VALUES (?,?,?,?,?,?)",
            (thread, ns, ck, "t", 0, "messages"),
        )
    await conn.commit()


@pytest.mark.asyncio
async def test_prune_keeps_latest_n_and_drops_orphaned_writes(tmp_path, monkeypatch):
    from mmm_framework.api import main

    monkeypatch.setenv("MMM_CHECKPOINT_RETENTION", "5")
    db = tmp_path / "sessions.db"
    conn = await aiosqlite.connect(str(db))
    try:
        await conn.executescript(_CK_SCHEMA)
        await _seed(conn, "T", "", 50)  # 50 versions in the main namespace
        await _seed(conn, "T", "sub", 8)  # 8 in a sub-graph namespace
        await _seed(conn, "OTHER", "", 20)  # a different thread, must survive whole

        monkeypatch.setattr(main, "_aiosqlite_conn", conn)
        await main._prune_thread_checkpoints("T")

        async def one(sql, *a):
            async with conn.execute(sql, a) as cur:
                return (await cur.fetchone())[0]

        # Thread T bounded to 5 PER namespace; OTHER untouched.
        assert (
            await one(
                "SELECT count(*) FROM checkpoints WHERE thread_id='T' AND checkpoint_ns=''"
            )
            == 5
        )
        assert (
            await one(
                "SELECT count(*) FROM checkpoints WHERE thread_id='T' AND checkpoint_ns='sub'"
            )
            == 5
        )
        assert (
            await one("SELECT count(*) FROM checkpoints WHERE thread_id='OTHER'") == 20
        )
        # The survivors are the NEWEST ids (ck-00045..ck-00049), not the oldest.
        assert (
            await one(
                "SELECT min(checkpoint_id) FROM checkpoints WHERE thread_id='T' AND checkpoint_ns=''"
            )
            == "ck-00045"
        )
        # Orphaned writes for pruned checkpoints are gone; kept ones remain.
        assert (
            await one(
                "SELECT count(*) FROM writes WHERE thread_id='T' AND checkpoint_ns=''"
            )
            == 5
        )
        assert await one("SELECT count(*) FROM writes WHERE thread_id='OTHER'") == 20
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_prune_is_a_noop_below_the_cap_and_when_conn_is_none(
    tmp_path, monkeypatch
):
    from mmm_framework.api import main

    monkeypatch.setenv("MMM_CHECKPOINT_RETENTION", "40")
    db = tmp_path / "sessions.db"
    conn = await aiosqlite.connect(str(db))
    try:
        await conn.executescript(_CK_SCHEMA)
        await _seed(conn, "T", "", 10)  # fewer than the cap => nothing pruned
        monkeypatch.setattr(main, "_aiosqlite_conn", conn)
        await main._prune_thread_checkpoints("T")
        async with conn.execute("SELECT count(*) FROM checkpoints") as cur:
            assert (await cur.fetchone())[0] == 10
    finally:
        await conn.close()

    # No connection yet (startup race) => best-effort no-op, never raises.
    monkeypatch.setattr(main, "_aiosqlite_conn", None)
    await main._prune_thread_checkpoints("T")
