"""Honest caveats on identifiability claims (Phase 4 / V2).

The DAG tool reports front-door / IV identifiability, but the framework only ever
delivers the back-door additive estimate. These tests pin that the report says so
plainly, so 'identifiable' is never mistaken for 'estimated via that route'.
"""

from __future__ import annotations

from mmm_framework.agents.causal_tools import validate_causal_identification
from mmm_framework.dag_model_builder.dag_spec import (
    DAGEdge,
    DAGNode,
    DAGSpec,
    NodeType,
)


def _invoke(dag: DAGSpec, treatment="TV", outcome="Sales") -> str:
    state = {"dashboard_data": {"dag": {"spec": dag.model_dump()}}}
    cmd = validate_causal_identification.invoke(
        {
            "name": "validate_causal_identification",
            "type": "tool_call",
            "id": "tc1",
            "args": {
                "treatment": treatment,
                "outcome": outcome,
                "state": state,
                "tool_call_id": "tc1",
            },
        }
    )
    return cmd.update["messages"][0].content


def _frontdoor_dag() -> DAGSpec:
    # TV -> M -> Sales (M intercepts the only causal path); Demand confounds TV<->Sales.
    nodes = [
        DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
        DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
        DAGNode(id="m", variable_name="Mediator", node_type=NodeType.MEDIATOR),
        DAGNode(id="demand", variable_name="Demand", node_type=NodeType.CONTROL),
    ]
    edges = [
        DAGEdge(source="tv", target="m"),
        DAGEdge(source="m", target="sales"),
        DAGEdge(source="demand", target="tv"),
        DAGEdge(source="demand", target="sales"),
    ]
    return DAGSpec(nodes=nodes, edges=edges)


def _iv_dag() -> DAGSpec:
    nodes = [
        DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
        DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
        DAGNode(id="z", variable_name="Coupon", node_type=NodeType.INSTRUMENT),
    ]
    edges = [
        DAGEdge(source="tv", target="sales"),
        DAGEdge(source="z", target="tv"),
    ]
    return DAGSpec(nodes=nodes, edges=edges)


def test_frontdoor_branch_carries_estimation_caveat():
    content = _invoke(_frontdoor_dag())
    assert "Front-door check" in content
    # The honest caveat: identifiable != estimated via front-door.
    assert "back-door additive estimator" in content
    assert "NOT the front-door estimate" in content


def test_iv_check_carries_estimation_caveat():
    content = _invoke(_iv_dag())
    assert "Instrumental-variable check" in content
    assert "does NOT run 2SLS/IV estimation" in content
    assert "back-door additive" in content
