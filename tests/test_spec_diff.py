"""Tests for the pre-spec lock + diff core (P2-3)."""

from __future__ import annotations

from mmm_framework.config import diff_spec, summarize_spec_diff


def _spec(channels, controls, draws=2000):
    return {
        "kpi": {"name": "Sales"},
        "media_channels": [{"name": c, "adstock": {"l_max": 8}} for c in channels],
        "controls": [{"name": c, "causal_role": "confounder"} for c in controls],
        "model": {"n_draws": draws, "n_chains": 4},
    }


class TestDiffSpec:
    def test_identical_specs_no_diff(self):
        s = _spec(["TV", "Radio"], ["Price"])
        assert diff_spec(s, s) == []
        assert "No divergence" in summarize_spec_diff(diff_spec(s, s))

    def test_reordering_channels_is_not_a_divergence(self):
        a = _spec(["TV", "Radio"], ["Price"])
        b = _spec(["Radio", "TV"], ["Price"])  # reordered
        assert diff_spec(a, b) == []

    def test_added_channel_detected(self):
        a = _spec(["TV"], ["Price"])
        b = _spec(["TV", "Radio"], ["Price"])
        changes = diff_spec(a, b)
        assert any(c.kind == "added" and "Radio" in c.path for c in changes)

    def test_removed_control_detected(self):
        a = _spec(["TV"], ["Price", "Weather"])
        b = _spec(["TV"], ["Price"])
        changes = diff_spec(a, b)
        assert any(c.kind == "removed" and "Weather" in c.path for c in changes)

    def test_changed_hyperparameter_detected(self):
        a = _spec(["TV"], ["Price"], draws=2000)
        b = _spec(["TV"], ["Price"], draws=500)
        changes = diff_spec(a, b)
        changed = [c for c in changes if c.kind == "changed"]
        assert len(changed) == 1
        assert changed[0].path.endswith("n_draws")
        assert (changed[0].old, changed[0].new) == (2000, 500)

    def test_changed_causal_role_detected(self):
        # The exact "researcher degrees of freedom" P2-3 targets: silently
        # re-labelling a confounder after pre-registration.
        a = _spec(["TV"], ["Price"])
        b = _spec(["TV"], ["Price"])
        b["controls"][0]["causal_role"] = "precision_control"
        changes = diff_spec(a, b)
        assert any("causal_role" in c.path and c.kind == "changed" for c in changes)

    def test_nested_change_detected(self):
        a = _spec(["TV"], ["Price"])
        b = _spec(["TV"], ["Price"])
        b["media_channels"][0]["adstock"]["l_max"] = 13
        changes = diff_spec(a, b)
        assert any("l_max" in c.path and c.new == 13 for c in changes)

    def test_summary_lists_every_change(self):
        a = _spec(["TV"], ["Price"])
        b = _spec(["TV", "Radio"], [])
        summary = summarize_spec_diff(diff_spec(a, b))
        assert "divergence" in summary.lower()
        assert "Radio" in summary and "Price" in summary

    def test_duplicate_names_disambiguated_cleanly(self):
        # Adding a SECOND channel named "TV" must surface as ONE clean addition,
        # not a re-indexing of the whole list. Both specs key by occurrence so
        # the shared first TV and Radio still match.
        a = {"media_channels": [{"name": "TV"}, {"name": "Radio"}]}
        b = {"media_channels": [{"name": "TV"}, {"name": "TV"}, {"name": "Radio"}]}
        changes = diff_spec(a, b)
        # Radio (unique on both sides) matches cleanly and is untouched -- the
        # key win vs. a whole-list re-index. Only the duplicated TV churns.
        assert all("Radio" not in c.path for c in changes)
        assert all("TV" in c.path for c in changes)
        assert len(changes) <= 3  # bounded, not the 165-diff blowup

    def test_unnamed_list_items_fall_back_to_index(self):
        a = {"weights": [1, 2, 3]}
        b = {"weights": [1, 9, 3]}
        changes = diff_spec(a, b)
        assert len(changes) == 1
        assert changes[0].kind == "changed"
        assert changes[0].old == 2 and changes[0].new == 9


class TestCheckSpecDivergenceTool:
    def _dag(self, channels):
        from mmm_framework.dag_model_builder.dag_spec import (
            DAGEdge,
            DAGNode,
            DAGSpec,
            NodeType,
        )

        nodes = [DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI)]
        edges = []
        for ch in channels:
            nodes.append(
                DAGNode(id=ch.lower(), variable_name=ch, node_type=NodeType.MEDIA)
            )
            edges.append(DAGEdge(source=ch.lower(), target="sales"))
        return DAGSpec(nodes=nodes, edges=edges)

    def _invoke(self, tid, state):
        from mmm_framework.agents.causal_tools import check_spec_divergence

        return check_spec_divergence.invoke(
            {
                "name": "check_spec_divergence",
                "type": "tool_call",
                "id": "tc",
                "args": {"state": state, "tool_call_id": "tc"},
            },
            config={"configurable": {"thread_id": tid}},
        ).update

    def test_reports_divergence_against_locked_plan(self):
        import uuid

        from mmm_framework.api import sessions as sessions_store

        sessions_store.init_db()
        tid = "test-" + uuid.uuid4().hex
        frozen = self._dag(["TV"])
        sessions_store.lock_analysis_plan(
            tid, "p", {"dag": {"spec": frozen.model_dump()}}
        )
        current = self._dag(["TV", "Radio"])  # a channel was added post-lock
        state = {"dashboard_data": {"dag": {"spec": current.model_dump()}}}
        update = self._invoke(tid, state)
        content = update["messages"][0].content
        assert "diverges" in content
        assert "Radio" in content
        assert update["dashboard_data"]["spec_divergences"]

    def test_no_divergence_when_unchanged(self):
        import uuid

        from mmm_framework.api import sessions as sessions_store

        sessions_store.init_db()
        tid = "test-" + uuid.uuid4().hex
        dag = self._dag(["TV", "Radio"])
        sessions_store.lock_analysis_plan(tid, "p", {"dag": {"spec": dag.model_dump()}})
        state = {"dashboard_data": {"dag": {"spec": dag.model_dump()}}}
        content = self._invoke(tid, state)["messages"][0].content
        assert "No divergence" in content

    def test_no_locked_plan_message(self):
        import uuid

        tid = "test-" + uuid.uuid4().hex  # never locked
        content = self._invoke(tid, {"dashboard_data": {}})["messages"][0].content
        assert "No locked analysis plan" in content

    def test_locked_plan_but_no_current_dag(self):
        # A plan is locked but the agent state has no current DAG -> everything in
        # the frozen spec reads as removed (a divergence), not a crash.
        import uuid

        from mmm_framework.api import sessions as sessions_store

        sessions_store.init_db()
        tid = "test-" + uuid.uuid4().hex
        dag = self._dag(["TV"])
        sessions_store.lock_analysis_plan(tid, "p", {"dag": {"spec": dag.model_dump()}})
        update = self._invoke(tid, {"dashboard_data": {}})
        content = update["messages"][0].content
        assert "diverges" in content
        assert update["dashboard_data"]["spec_divergences"]
        assert all(
            d["kind"] == "removed" for d in update["dashboard_data"]["spec_divergences"]
        )
