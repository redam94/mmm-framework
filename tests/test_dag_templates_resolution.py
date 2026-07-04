"""The CausalPlanner's starter templates must resolve to the intended model.

Mirrors frontend/src/components/causal/templates.ts (the react_flow payloads
the editor loads verbatim) through the same adapter path the API uses
(react_flow_to_dag_spec) and pins each template's resolve_model_type — in
particular the 'combined' template (mediator + two outcomes), which must reach
CombinedMMM. If a template edit in the frontend breaks the resolution (e.g.
dropping the second outcome), this fails instead of silently downgrading the
model class.
"""

import pytest

from mmm_framework.dag_model_builder.frontend_adapter import react_flow_to_dag_spec
from mmm_framework.dag_model_builder.model_type_resolver import (
    ModelType,
    resolve_model_type,
)
from mmm_framework.dag_model_builder.validation import validate_dag


def _node(id_, type_, variable):
    return {
        "id": id_,
        "position": {"x": 0, "y": 0},
        "data": {"label": variable, "variableName": variable, "type": type_},
    }


def _edge(i, source, target):
    return {"id": f"e{i}", "source": source, "target": target}


# Faithful ports of DAG_TEMPLATES in frontend/src/components/causal/templates.ts
TEMPLATES = {
    "simple": (
        [
            _node("kpi_1", "kpi", "sales"),
            _node("media_1", "media", "tv_spend"),
            _node("media_2", "media", "digital_spend"),
            _node("control_1", "control", "price"),
        ],
        [
            _edge(1, "media_1", "kpi_1"),
            _edge(2, "media_2", "kpi_1"),
            _edge(3, "control_1", "kpi_1"),
        ],
        ModelType.BAYESIAN_MMM,
    ),
    "mediation": (
        [
            _node("kpi_1", "kpi", "sales"),
            _node("media_1", "media", "tv_spend"),
            _node("mediator_1", "mediator", "awareness"),
        ],
        [
            _edge(1, "media_1", "mediator_1"),
            _edge(2, "mediator_1", "kpi_1"),
            _edge(3, "media_1", "kpi_1"),
        ],
        ModelType.NESTED_MMM,
    ),
    "multivariate": (
        [
            _node("outcome_1", "outcome", "revenue"),
            _node("outcome_2", "outcome", "volume"),
            _node("media_1", "media", "marketing"),
        ],
        [
            _edge(1, "media_1", "outcome_1"),
            _edge(2, "media_1", "outcome_2"),
            _edge(3, "outcome_1", "outcome_2"),
        ],
        ModelType.MULTIVARIATE_MMM,
    ),
    "combined": (
        [
            _node("media_1", "media", "tv_spend"),
            _node("media_2", "media", "digital_spend"),
            _node("mediator_1", "mediator", "awareness"),
            _node("outcome_1", "outcome", "revenue"),
            _node("outcome_2", "outcome", "volume"),
        ],
        [
            _edge(1, "media_1", "mediator_1"),
            _edge(2, "mediator_1", "outcome_1"),
            _edge(3, "mediator_1", "outcome_2"),
            _edge(4, "media_1", "outcome_1"),
            _edge(5, "media_2", "outcome_1"),
            _edge(6, "outcome_1", "outcome_2"),
        ],
        ModelType.COMBINED_MMM,
    ),
}


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_resolves_to_intended_model_type(name):
    nodes, edges, expected = TEMPLATES[name]
    dag = react_flow_to_dag_spec(nodes, edges)
    assert resolve_model_type(dag) is expected


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_dag_is_structurally_valid(name):
    nodes, edges, _ = TEMPLATES[name]
    dag = react_flow_to_dag_spec(nodes, edges)
    result = validate_dag(dag)
    assert result.valid, result.errors
