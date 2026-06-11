"""Tests for field-level model_spec locking (manual-edit priority + confirmation).

Covers the dot-path helpers, server-side diff-based auto-lock, and
``reconcile_with_locks`` — the shared core both ``configure_model`` and
``update_model_setting`` route through so an LLM can never silently overwrite a
field the user set manually.
"""

from __future__ import annotations

import copy

from mmm_framework.agents.spec_locks import (
    flatten_leaves,
    get_at,
    set_at,
    diff_locked,
    reconcile_with_locks,
    merge_pending,
)


def _spec():
    return {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [
            {"name": "TV", "adstock": {"type": "geometric", "l_max": 8}},
            {"name": "Digital"},
        ],
        "control_variables": [{"name": "Price"}],
        "inference": {"chains": 4, "draws": 1000},
    }


class TestPathHelpers:
    def test_flatten_uses_channel_name_as_segment(self):
        leaves = flatten_leaves(_spec())
        assert leaves["media_channels.TV.adstock.l_max"] == 8
        assert leaves["inference.draws"] == 1000
        # a name-only channel still registers via its name leaf
        assert leaves["media_channels.Digital.name"] == "Digital"

    def test_get_at_named_list_and_default(self):
        s = _spec()
        assert get_at(s, "media_channels.TV.adstock.l_max") == 8
        assert get_at(s, "inference.draws") == 1000
        assert get_at(s, "media_channels.Radio.adstock.l_max", "X") == "X"

    def test_set_at_autocreates_dicts_and_channels(self):
        s = _spec()
        set_at(s, "inference.target_accept", 0.95)
        assert s["inference"]["target_accept"] == 0.95
        # auto-create a brand-new channel + nested path
        set_at(s, "media_channels.Radio.adstock.l_max", 4)
        assert get_at(s, "media_channels.Radio.adstock.l_max") == 4


class TestDiffLocked:
    def test_identical_spec_locks_nothing(self):
        s = _spec()
        assert diff_locked(s, copy.deepcopy(s)) == []

    def test_only_touched_leaves_are_locked(self):
        old = _spec()
        new = copy.deepcopy(old)
        set_at(new, "inference.draws", 2000)
        set_at(new, "media_channels.TV.adstock.l_max", 13)
        set_at(new, "inference.target_accept", 0.9)  # newly added
        assert set(diff_locked(old, new)) == {
            "inference.draws",
            "media_channels.TV.adstock.l_max",
            "inference.target_accept",
        }


class TestReconcileWithLocks:
    def test_non_locked_change_applies(self):
        current = _spec()
        cand = copy.deepcopy(current)
        set_at(cand, "inference.chains", 6)
        merged, pending = reconcile_with_locks(cand, current, locked_fields=[])
        assert get_at(merged, "inference.chains") == 6
        assert pending == []

    def test_locked_conflict_is_reverted_and_deferred(self):
        current = _spec()
        cand = copy.deepcopy(current)
        set_at(cand, "inference.draws", 4000)
        merged, pending = reconcile_with_locks(
            cand,
            current,
            ["inference.draws"],
            reason="need more draws",
            tool_call_id="t1",
        )
        # user's value preserved
        assert get_at(merged, "inference.draws") == 1000
        assert len(pending) == 1
        p = pending[0]
        assert p["path"] == "inference.draws"
        assert p["current"] == 1000
        assert p["proposed"] == 4000
        assert p["reason"] == "need more draws"

    def test_mixed_batch_applies_free_defers_locked(self):
        current = _spec()
        cand = copy.deepcopy(current)
        set_at(cand, "inference.draws", 4000)  # locked
        set_at(cand, "inference.chains", 8)  # free
        merged, pending = reconcile_with_locks(cand, current, ["inference.draws"])
        assert get_at(merged, "inference.draws") == 1000  # reverted
        assert get_at(merged, "inference.chains") == 8  # applied
        assert [p["path"] for p in pending] == ["inference.draws"]

    def test_reconfigure_reasserts_locked_field_dropped_by_candidate(self):
        # configure_model rebuilds the whole spec and omits inference entirely.
        current = _spec()
        rebuilt = {
            "kpi": "Sales",
            "kpi_level": "national",
            "media_channels": [{"name": "TV"}],
            "control_variables": [],
            "time_granularity": "weekly",
            "model_type": "numpyro",
        }
        merged, pending = reconcile_with_locks(
            rebuilt, current, ["inference.draws", "media_channels.TV.adstock.l_max"]
        )
        # locked values survive the rebuild, no confirmation needed (just re-asserted)
        assert get_at(merged, "inference.draws") == 1000
        assert get_at(merged, "media_channels.TV.adstock.l_max") == 8
        assert pending == []

    def test_locked_field_referencing_removed_channel_is_ignored(self):
        current = _spec()
        # lock a field on a channel that the candidate no longer contains at all
        cand = {"kpi": "Sales", "media_channels": [{"name": "TV"}]}
        merged, pending = reconcile_with_locks(
            cand, current, ["control_variables.Price.name"]
        )
        # Price still exists in current, so it's re-asserted; no crash
        assert get_at(merged, "control_variables.Price.name") == "Price"
        assert pending == []


class TestMergePending:
    def test_dedup_by_path_newest_wins(self):
        merged = merge_pending(
            [{"path": "inference.draws", "proposed": 2000}],
            [{"path": "inference.draws", "proposed": 4000}],
        )
        assert len(merged) == 1
        assert merged[0]["proposed"] == 4000

    def test_distinct_paths_kept(self):
        merged = merge_pending(
            [{"path": "a", "proposed": 1}], [{"path": "b", "proposed": 2}]
        )
        assert {p["path"] for p in merged} == {"a", "b"}


class TestToolDeferral:
    """The actual modal-triggering path: tools route through _commit_spec, which
    reads locked_fields from injected state and defers conflicting changes."""

    def test_update_model_setting_free_field_applies(self):
        from mmm_framework.agents import tools as T
        from mmm_framework.agents.spec_locks import is_spec_patch
        from mmm_framework.agents.state import _merge_spec

        state = {"model_spec": _spec(), "locked_fields": [], "dashboard_data": {}}
        cmd = T.update_model_setting.func(
            state=state, setting_path="inference.chains", value=8, tool_call_id="c1"
        )
        # single-setting updates are written as patch envelopes so concurrent
        # tool calls in one step compose instead of clobbering each other
        assert is_spec_patch(cmd.update["model_spec"])
        folded = _merge_spec(state["model_spec"], cmd.update["model_spec"])
        assert folded["inference"]["chains"] == 8
        assert folded["inference"]["draws"] == 1000  # untouched field survives
        assert cmd.update["pending_spec_changes"] == []

    def test_update_model_setting_locked_field_is_deferred(self):
        from mmm_framework.agents import tools as T
        from mmm_framework.agents.state import _merge_spec

        state = {
            "model_spec": _spec(),
            "locked_fields": ["inference.draws"],
            "dashboard_data": {},
        }
        cmd = T.update_model_setting.func(
            state=state,
            setting_path="inference.draws",
            value=4000,
            reason="need more draws",
            tool_call_id="c2",
        )
        # the user's value is preserved (NOT silently overwritten)
        folded = _merge_spec(state["model_spec"], cmd.update["model_spec"])
        assert folded["inference"]["draws"] == 1000
        pending = cmd.update["pending_spec_changes"]
        assert [p["path"] for p in pending] == ["inference.draws"]
        assert (
            pending[0]["proposed"] == 4000 and pending[0]["reason"] == "need more draws"
        )
        # the ToolMessage tells the LLM not to retry
        msg = cmd.update["messages"][0].content
        assert "Do not retry" in msg
        # mirrored into dashboard_data for the frontend modal
        assert cmd.update["dashboard_data"]["pending_spec_changes"] == pending

    def test_configure_model_respects_locked_field(self):
        from mmm_framework.agents import tools as T

        # user locked granularity to daily; a reconfigure hardcodes "weekly"
        spec = _spec()
        spec["time_granularity"] = "daily"
        state = {
            "model_spec": spec,
            "locked_fields": ["time_granularity"],
            "dashboard_data": {},
        }
        cmd = T.configure_model.func(
            state=state,
            kpi="Sales",
            kpi_level="national",
            media_channels=["TV"],
            control_variables=[],
            tool_call_id="c3",
        )
        assert cmd.update["model_spec"]["time_granularity"] == "daily"  # kept
        assert [p["path"] for p in cmd.update["pending_spec_changes"]] == [
            "time_granularity"
        ]


class TestConcurrentSpecUpdates:
    """LangGraph's ToolNode runs all tool_calls of one AIMessage concurrently
    against the SAME state snapshot. update_model_setting therefore emits patch
    envelopes that the model_spec reducer applies against the latest folded
    value — full-spec writes from each tool would last-writer-wins clobber the
    others' changes (each carries stale values for fields it didn't touch)."""

    def test_parallel_setting_updates_compose(self):
        from mmm_framework.agents import tools as T
        from mmm_framework.agents.state import _merge_spec

        spec = _spec()
        state = {"model_spec": spec, "locked_fields": [], "dashboard_data": {}}
        # Two tools fired from the same snapshot, different channels
        cmd_tv = T.update_model_setting.func(
            state=state,
            setting_path="media_channels.TV.adstock.l_max",
            value=12,
            tool_call_id="c1",
        )
        cmd_dig = T.update_model_setting.func(
            state=state,
            setting_path="media_channels.Digital.adstock.l_max",
            value=6,
            tool_call_id="c2",
        )
        # LangGraph folds same-step writes through the reducer in order
        folded = _merge_spec(spec, cmd_tv.update["model_spec"])
        folded = _merge_spec(folded, cmd_dig.update["model_spec"])
        assert get_at(folded, "media_channels.TV.adstock.l_max") == 12
        assert get_at(folded, "media_channels.Digital.adstock.l_max") == 6

    def test_full_spec_write_still_replaces(self):
        from mmm_framework.agents.state import _merge_spec

        replacement = {"kpi": "Revenue", "media_channels": []}
        assert _merge_spec(_spec(), replacement) == replacement

    def test_dashboard_mirror_composes_patches(self):
        from mmm_framework.agents.spec_locks import is_spec_patch, make_spec_patch
        from mmm_framework.agents.state import _merge_dashboard

        d0 = {"model_spec": _spec()}
        p1 = make_spec_patch([{"path": "media_channels.TV.adstock.l_max", "value": 12}])
        p2 = make_spec_patch([{"path": "inference.chains", "value": 8}])
        d1 = _merge_dashboard(d0, {"model_spec": p1})
        d2 = _merge_dashboard(d1, {"model_spec": p2})
        # stored mirror is always a concrete spec with both changes applied
        assert not is_spec_patch(d2["model_spec"])
        assert get_at(d2["model_spec"], "media_channels.TV.adstock.l_max") == 12
        assert get_at(d2["model_spec"], "inference.chains") == 8
