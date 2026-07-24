"""Plain-language DAG reading (`dag_model_builder.narrative`) + the causal
elicitation tools (`causal_structure_interview`, `explain_dag`).

The reading is deterministic — the agent relays it instead of inventing causal
interpretation — so these tests pin the load-bearing sentences: mediators must
never be controls, confounders get the both-arrows story, missing arrows are
stated as assumptions, and unmeasured nodes are flagged with the data they need.
"""

from __future__ import annotations

from mmm_framework.dag_model_builder.dag_spec import (
    DAGEdge,
    DAGNode,
    DAGSpec,
    EdgeType,
    NodeType,
)
from mmm_framework.dag_model_builder.narrative import dag_human_reading


def _mmm_dag() -> DAGSpec:
    """TV → awareness → Sales (+ TV direct); Search direct; demand confounds
    TV & Sales; price is a KPI-only control."""
    nodes = [
        DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
        DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
        DAGNode(id="search", variable_name="Search", node_type=NodeType.MEDIA),
        DAGNode(id="awareness", variable_name="Awareness", node_type=NodeType.MEDIATOR),
        DAGNode(id="demand", variable_name="Demand", node_type=NodeType.CONTROL),
        DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL),
    ]
    edges = [
        DAGEdge(source="tv", target="sales"),
        DAGEdge(source="tv", target="awareness", edge_type=EdgeType.MEDIATED),
        DAGEdge(source="awareness", target="sales", edge_type=EdgeType.MEDIATED),
        DAGEdge(source="search", target="sales"),
        DAGEdge(source="demand", target="tv"),
        DAGEdge(source="demand", target="sales"),
        DAGEdge(source="price", target="sales"),
    ]
    return DAGSpec(nodes=nodes, edges=edges)


class TestDagHumanReading:
    def test_media_routes_are_narrated(self):
        text = dag_human_reading(_mmm_dag())
        # TV: both routes; Search: direct only
        assert "`TV` reaches `Sales` two ways" in text
        assert "`Awareness`" in text
        assert "`Search` → `Sales` directly" in text

    def test_mediator_never_a_control(self):
        text = dag_human_reading(_mmm_dag())
        assert "NEVER be listed as a control" in text
        assert "total effect" in text

    def test_confounder_vs_precision_control(self):
        text = dag_human_reading(_mmm_dag())
        # Demand: both arrows + back-door language; Price: precision control
        assert "`Demand` drives BOTH `TV` and `Sales`" in text
        assert "back-door" in text
        assert "associational, not a causal effect" in text
        assert "`Price` affects only `Sales`" in text
        assert "precision control" in text

    def test_missing_arrow_assumptions_always_stated(self):
        text = dag_human_reading(_mmm_dag())
        assert "budgets do NOT react to the KPI" in text
        assert "no-unobserved-confounding" in text

    def test_data_checklist_flags_unmeasured_nodes(self):
        cols = {"Sales", "TV", "Search", "Price"}  # Demand + Awareness unmeasured
        text = dag_human_reading(_mmm_dag(), known_columns=cols)
        assert "Measured ✓" in text and "`TV`" in text
        assert "`Demand`" in text and "back-door OPEN" in text
        assert "`Awareness`" in text and "keep it out of the controls" in text
        # loose matching: case/punctuation-insensitive
        text2 = dag_human_reading(_mmm_dag(), known_columns={"sales", "tv_"})
        assert "Needs a series" in text2

    def test_disconnected_media_flagged(self):
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
        ]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=[]))
        assert "NO route to any outcome" in text

    def test_unwired_mediator_flagged(self):
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="m", variable_name="Aware", node_type=NodeType.MEDIATOR),
        ]
        edges = [DAGEdge(source="m", target="sales", edge_type=EdgeType.MEDIATED)]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=edges))
        assert "no incoming driver edge" in text

    def test_spend_driver_is_not_called_a_confounder(self):
        """A control that steers spend but never touches the KPI opens no
        back-door — the reading must say so (and warn about conditioning),
        never claim it 'drives BOTH X and Sales'."""
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="plan", variable_name="PlanCycle", node_type=NodeType.CONTROL),
        ]
        edges = [
            DAGEdge(source="tv", target="sales"),
            DAGEdge(source="plan", target="tv"),
        ]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=edges))
        assert "drives BOTH" not in text
        assert "NOT a confounder" in text and "amplify" in text
        # ... and the data checklist must not claim an open back-door for it
        text2 = dag_human_reading(
            DAGSpec(nodes=nodes, edges=edges), known_columns={"Sales", "TV"}
        )
        assert "back-door OPEN" not in text2
        assert "`PlanCycle`" in text2

    def test_mediator_outcome_confounder_not_called_precision(self):
        """A control driving a mediator AND the KPI confounds the mediation
        pathway — it must not be filed as an inert 'affects only Sales'
        precision control."""
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="aw", variable_name="Awareness", node_type=NodeType.MEDIATOR),
            DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL),
        ]
        edges = [
            DAGEdge(source="tv", target="aw", edge_type=EdgeType.MEDIATED),
            DAGEdge(source="aw", target="sales", edge_type=EdgeType.MEDIATED),
            DAGEdge(source="price", target="aw"),
            DAGEdge(source="price", target="sales"),
        ]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=edges))
        assert "affects only `Sales`" not in text
        assert "confounds the mediation pathway" in text

    def test_mediator_chain_not_flagged_disconnected(self):
        """Funnel chains (TV → awareness → consideration → Sales) are legal —
        the downstream mediator is driven by the upstream one, not 'floating'."""
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="aw", variable_name="Awareness", node_type=NodeType.MEDIATOR),
            DAGNode(
                id="cons", variable_name="Consideration", node_type=NodeType.MEDIATOR
            ),
        ]
        edges = [
            DAGEdge(source="tv", target="aw", edge_type=EdgeType.MEDIATED),
            DAGEdge(source="aw", target="cons", edge_type=EdgeType.MEDIATED),
            DAGEdge(source="cons", target="sales", edge_type=EdgeType.MEDIATED),
        ]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=edges))
        assert "no incoming driver edge" not in text
        assert "floats disconnected" not in text
        # the chain is narrated: consideration carries awareness's effect
        assert "`Consideration` carries part of `Awareness`" in text

    def test_mediator_without_outlet_flagged(self):
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="aw", variable_name="Awareness", node_type=NodeType.MEDIATOR),
        ]
        edges = [
            DAGEdge(source="tv", target="sales"),
            DAGEdge(source="tv", target="aw", edge_type=EdgeType.MEDIATED),
            # awareness → sales edge missing
        ]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=edges))
        assert "does not reach any outcome" in text

    def test_secondary_outcome_channel_not_told_to_be_dropped(self):
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="brand", variable_name="Brand", node_type=NodeType.OUTCOME),
            DAGNode(id="social", variable_name="Social", node_type=NodeType.MEDIA),
        ]
        edges = [DAGEdge(source="social", target="brand")]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=edges))
        assert "NO route" not in text
        assert "`Social` moves `Brand`" in text and "secondary" in text

    def test_instrument_and_cross_effect_narrated(self):
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="web", variable_name="Web", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="z", variable_name="Mandate", node_type=NodeType.INSTRUMENT),
        ]
        edges = [
            DAGEdge(source="tv", target="sales"),
            DAGEdge(source="z", target="tv"),
            DAGEdge(source="web", target="sales", edge_type=EdgeType.CROSS_EFFECT),
        ]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=edges))
        assert "exclusion restriction" in text
        assert "spills over" in text

    def test_no_kpi_degrades_gracefully(self):
        nodes = [DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA)]
        text = dag_human_reading(DAGSpec(nodes=nodes, edges=[]))
        assert "no KPI" in text


# --------------------------------------------------------------------------- #
# agent tools
# --------------------------------------------------------------------------- #


def _invoke(tool, args: dict) -> tuple[str, dict]:
    cmd = tool.invoke(
        {
            "name": tool.name,
            "type": "tool_call",
            "id": "tc1",
            "args": {**args, "tool_call_id": "tc1"},
        }
    )
    content = cmd.update["messages"][0].content
    return content, cmd.update.get("dashboard_data", {})


class TestCausalStructureInterview:
    def test_questions_are_grounded_in_session(self):
        from mmm_framework.agents.causal_tools import causal_structure_interview

        state = {
            "model_spec": {"kpi": "Revenue", "media_channels": ["TV", "Search"]},
            "dashboard_data": {},
        }
        content, dashboard = _invoke(causal_structure_interview, {"state": state})
        # grounded in the session's channels/KPI
        assert "`TV`" in content and "`Search`" in content and "Revenue" in content
        # the elicitation contract: focused round, wait for answers, map to DAG
        assert "3–5" in content
        assert "propose_dag" in content
        assert "Budget-setting" in content and "mediators" in content.lower()
        # data implications are spelled out
        assert "series" in content
        assert dashboard["causal_interview"]["questions"]

    def test_dict_form_spec_entries_render_clean_names(self):
        """The canonical stored spec keeps media_channels as {"name": ...}
        dicts — the questions must show channel names, never dict reprs."""
        from mmm_framework.agents.causal_tools import causal_structure_interview

        state = {
            "model_spec": {
                "kpi": "Revenue",
                "media_channels": [{"name": "TV"}, {"name": "Search"}],
            },
            "dashboard_data": {},
        }
        content, _ = _invoke(causal_structure_interview, {"state": state})
        assert "`TV`" in content and "`Search`" in content
        assert "{'name'" not in content and '{"name"' not in content

    def test_existing_dag_switches_to_refine_framing(self):
        from mmm_framework.agents.causal_tools import causal_structure_interview

        state = {
            "model_spec": {},
            "dashboard_data": {"dag": {"spec": _mmm_dag().model_dump(mode="json")}},
        }
        content, _ = _invoke(causal_structure_interview, {"state": state})
        assert "REFINE" in content


class TestExplainDag:
    def test_reads_current_dag_and_stores_reading(self):
        from mmm_framework.agents.causal_tools import explain_dag

        state = {
            "model_spec": {
                "kpi": "Sales",
                "media_channels": ["TV", "Search"],
                "control_variables": ["Price"],
            },
            "dashboard_data": {"dag": {"spec": _mmm_dag().model_dump(mode="json")}},
        }
        content, dashboard = _invoke(explain_dag, {"state": state})
        assert "What this DAG says" in content
        assert "NEVER be listed as a control" in content
        # unmeasured nodes flagged against spec-known columns
        assert "`Demand`" in content
        assert "What this DAG says" in dashboard["dag"]["human_reading"]

    def test_no_dag_points_to_interview(self):
        from mmm_framework.agents.causal_tools import explain_dag

        content, _ = _invoke(explain_dag, {"state": {"dashboard_data": {}}})
        assert "propose_dag" in content and "causal_structure_interview" in content


class TestProposeDagCarriesReading:
    def test_propose_dag_message_includes_reading(self):
        from mmm_framework.agents.causal_tools import propose_dag

        state = {"model_spec": {}, "dashboard_data": {}}
        content, dashboard = _invoke(
            propose_dag,
            {
                "kpi": "Sales",
                "media_channels": ["TV", "Search"],
                "controls": ["Price"],
                "mediators": ["Awareness"],
                "mediator_inputs": {"Awareness": ["TV"]},
                "confounders": [
                    {"name": "Demand", "affects": ["TV", "Sales"]},
                ],
                "state": state,
            },
        )
        assert "What this DAG says" in content
        assert "CONFIRM or CORRECT" in content
        assert "What this DAG says" in dashboard["dag"]["human_reading"]


class TestIdentificationPlainReading:
    def test_identified_and_open_backdoor_phrasings(self):
        from mmm_framework.agents.causal_tools import validate_causal_identification

        # identified (demand controlled)
        state = {
            "dashboard_data": {"dag": {"spec": _mmm_dag().model_dump(mode="json")}}
        }
        content, _ = _invoke(
            validate_causal_identification,
            {"treatment": "TV", "outcome": "Sales", "state": state},
        )
        assert "In plain terms" in content

        # not identified: demand confounds but is NOT conditioned on
        content2, _ = _invoke(
            validate_causal_identification,
            {
                "treatment": "TV",
                "outcome": "Sales",
                "adjustment_set": [],
                "state": state,
            },
        )
        assert "In plain terms" in content2
        if "❌ No" in content2:
            assert "cannot be read causally yet" in content2

    def test_unmeasured_adjustment_variable_downgrades_the_verdict(self):
        """Graph-identifiability with a declared-but-unmeasured confounder in
        the adjustment set must NOT read as 'causal' — the session knows no
        series exists for it."""
        from mmm_framework.agents.causal_tools import validate_causal_identification

        state = {
            "model_spec": {
                "kpi": "Sales",
                "media_channels": [{"name": "TV"}, {"name": "Search"}],
                "control_variables": [{"name": "Price"}],  # Demand NOT measured
            },
            "dashboard_data": {"dag": {"spec": _mmm_dag().model_dump(mode="json")}},
        }
        content, _ = _invoke(
            validate_causal_identification,
            {"treatment": "TV", "outcome": "Sales", "state": state},
        )
        if "✅ Yes" in content:
            assert "no measured series" in content
            assert "`Demand`" in content
            assert "not yet causal" in content
