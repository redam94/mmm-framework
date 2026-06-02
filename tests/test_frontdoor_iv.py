"""Tests for front-door and IV identification (P2-5).

Each criterion is checked against a *textbook* DAG with a known answer (the
classic smoking → tar → cancer front-door example and the canonical Z → T → Y
instrument), plus the precise failure modes for each condition.
"""

from __future__ import annotations

from mmm_framework.dag_model_builder.dag_spec import (
    DAGEdge,
    DAGNode,
    DAGSpec,
    NodeType,
)
from mmm_framework.dag_model_builder.identification import (
    frontdoor_criterion,
    iv_criterion,
)


def _dag(nodes: dict[str, NodeType], edges: list[tuple[str, str]]) -> DAGSpec:
    return DAGSpec(
        nodes=[
            DAGNode(id=i, variable_name=i.upper(), node_type=t)
            for i, t in nodes.items()
        ],
        edges=[DAGEdge(source=s, target=t) for s, t in edges],
    )


# =============================================================================
# Front-door criterion
# =============================================================================


class TestFrontdoor:
    def test_textbook_smoking_tar_cancer_is_identified(self):
        # T → M → Y with an unobserved common cause U of T and Y. The classic
        # front-door example: the T→Y effect is identified through M despite U.
        spec = _dag(
            {
                "t": NodeType.MEDIA,
                "m": NodeType.MEDIATOR,
                "y": NodeType.KPI,
                "u": NodeType.CONTROL,
            },
            [("t", "m"), ("m", "y"), ("u", "t"), ("u", "y")],
        )
        rep = frontdoor_criterion(spec, "t", ["m"], "y")
        assert rep.intercepts_all_paths is True
        assert rep.treatment_mediator_unconfounded is True
        assert rep.mediator_outcome_blocked_by_treatment is True
        assert rep.identifiable is True

    def test_direct_effect_breaks_front_door(self):
        # An unmediated T → Y edge means M does not intercept all directed paths.
        spec = _dag(
            {"t": NodeType.MEDIA, "m": NodeType.MEDIATOR, "y": NodeType.KPI},
            [("t", "m"), ("m", "y"), ("t", "y")],
        )
        rep = frontdoor_criterion(spec, "t", ["m"], "y")
        assert rep.intercepts_all_paths is False
        assert rep.identifiable is False

    def test_confounded_treatment_mediator_breaks_front_door(self):
        # W confounds T and M -> open back-door T → M, condition (b) fails.
        spec = _dag(
            {
                "t": NodeType.MEDIA,
                "m": NodeType.MEDIATOR,
                "y": NodeType.KPI,
                "w": NodeType.CONTROL,
            },
            [("t", "m"), ("m", "y"), ("w", "t"), ("w", "m")],
        )
        rep = frontdoor_criterion(spec, "t", ["m"], "y")
        assert rep.treatment_mediator_unconfounded is False
        assert rep.identifiable is False

    def test_unblocked_mediator_outcome_backdoor_breaks_front_door(self):
        # V confounds M and Y and is NOT blocked by conditioning on T.
        spec = _dag(
            {
                "t": NodeType.MEDIA,
                "m": NodeType.MEDIATOR,
                "y": NodeType.KPI,
                "v": NodeType.CONTROL,
            },
            [("t", "m"), ("m", "y"), ("v", "m"), ("v", "y")],
        )
        rep = frontdoor_criterion(spec, "t", ["m"], "y")
        assert rep.mediator_outcome_blocked_by_treatment is False
        assert rep.identifiable is False


# =============================================================================
# Instrumental-variable criterion
# =============================================================================


class TestIV:
    def _iv_dag(self, extra_edges=None):
        # Canonical IV: Z → T → Y, with U confounding T and Y; Z exogenous.
        edges = [("z", "t"), ("t", "y"), ("u", "t"), ("u", "y")]
        edges += extra_edges or []
        return _dag(
            {
                "z": NodeType.INSTRUMENT,
                "t": NodeType.MEDIA,
                "y": NodeType.KPI,
                "u": NodeType.CONTROL,
            },
            edges,
        )

    def test_canonical_instrument_is_valid(self):
        rep = iv_criterion(self._iv_dag(), "z", "t", "y")
        assert rep.is_relevant is True
        assert rep.is_exogenous is True
        assert rep.satisfies_exclusion is True
        assert rep.identifiable is True
        assert rep.weak_instrument_warning is not None  # graph can't assess strength

    def test_exclusion_violation(self):
        # A direct Z → Y edge violates the exclusion restriction.
        rep = iv_criterion(self._iv_dag([("z", "y")]), "z", "t", "y")
        assert rep.satisfies_exclusion is False
        assert rep.identifiable is False

    def test_irrelevant_instrument(self):
        # Z does not reach T at all.
        spec = _dag(
            {
                "z": NodeType.INSTRUMENT,
                "q": NodeType.CONTROL,
                "t": NodeType.MEDIA,
                "y": NodeType.KPI,
                "u": NodeType.CONTROL,
            },
            [("z", "q"), ("t", "y"), ("u", "t"), ("u", "y")],
        )
        rep = iv_criterion(spec, "z", "t", "y")
        assert rep.is_relevant is False
        assert rep.identifiable is False

    def test_endogenous_instrument(self):
        # W confounds Z and Y -> Z is not exogenous (open back-door Z → Y).
        spec = _dag(
            {
                "w": NodeType.CONTROL,
                "z": NodeType.INSTRUMENT,
                "t": NodeType.MEDIA,
                "y": NodeType.KPI,
            },
            [("w", "z"), ("w", "y"), ("z", "t"), ("t", "y")],
        )
        rep = iv_criterion(spec, "z", "t", "y")
        assert rep.is_exogenous is False
        assert rep.identifiable is False


# =============================================================================
# Agent surfacing: validate_causal_identification reports front-door / IV
# =============================================================================


def _named_dag(nodes, edges) -> DAGSpec:
    return DAGSpec(
        nodes=[
            DAGNode(id=i, variable_name=name, node_type=t)
            for i, (name, t) in nodes.items()
        ],
        edges=[DAGEdge(source=s, target=d) for s, d in edges],
    )


def _invoke_validate(dag: DAGSpec, treatment: str, outcome: str) -> str:
    from mmm_framework.agents.causal_tools import validate_causal_identification

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


class TestAgentReporting:
    def test_frontdoor_reported_for_mediation_dag(self):
        dag = _named_dag(
            {
                "tv": ("TV", NodeType.MEDIA),
                "aware": ("Awareness", NodeType.MEDIATOR),
                "sales": ("Sales", NodeType.KPI),
                "demand": ("Demand", NodeType.CONTROL),
            },
            [
                ("tv", "aware"),
                ("aware", "sales"),
                ("demand", "tv"),
                ("demand", "sales"),
            ],
        )
        content = _invoke_validate(dag, "TV", "Sales")
        assert "Front-door check" in content
        assert "✅ Yes" in content

    def test_iv_reported_for_instrument_dag(self):
        dag = _named_dag(
            {
                "z": ("PolicyShock", NodeType.INSTRUMENT),
                "tv": ("TV", NodeType.MEDIA),
                "sales": ("Sales", NodeType.KPI),
                "demand": ("Demand", NodeType.CONTROL),
            },
            [
                ("z", "tv"),
                ("tv", "sales"),
                ("demand", "tv"),
                ("demand", "sales"),
            ],
        )
        content = _invoke_validate(dag, "TV", "Sales")
        assert "Instrumental-variable check" in content
        assert "valid" in content

    def test_propose_dag_wires_instrument_node(self):
        from mmm_framework.agents.causal_tools import propose_dag
        from mmm_framework.dag_model_builder.dag_spec import DAGSpec as _Spec

        state = {"dashboard_data": {}}
        cmd = propose_dag.invoke(
            {
                "name": "propose_dag",
                "type": "tool_call",
                "id": "tc1",
                "args": {
                    "kpi": "Sales",
                    "media_channels": ["TV"],
                    "instruments": [{"name": "PolicyShock", "treatment": "TV"}],
                    "state": state,
                    "tool_call_id": "tc1",
                },
            }
        )
        spec_dict = cmd.update["dashboard_data"]["dag"]["spec"]
        spec = _Spec.model_validate(spec_dict)
        assert spec.has_instruments
        assert any(n.variable_name == "PolicyShock" for n in spec.instrument_nodes)

    def test_frontdoor_agent_reports_failure(self):
        # Direct TV→Sales effect breaks the front door -> agent reports ❌ No.
        dag = _named_dag(
            {
                "tv": ("TV", NodeType.MEDIA),
                "aware": ("Awareness", NodeType.MEDIATOR),
                "sales": ("Sales", NodeType.KPI),
            },
            [("tv", "aware"), ("aware", "sales"), ("tv", "sales")],
        )
        content = _invoke_validate(dag, "TV", "Sales")
        assert "Front-door check" in content
        assert "❌ No" in content

    def test_iv_agent_reports_failure(self):
        # PolicyShock → Sales directly violates exclusion -> agent reports invalid.
        dag = _named_dag(
            {
                "z": ("PolicyShock", NodeType.INSTRUMENT),
                "tv": ("TV", NodeType.MEDIA),
                "sales": ("Sales", NodeType.KPI),
            },
            [("z", "tv"), ("tv", "sales"), ("z", "sales")],
        )
        content = _invoke_validate(dag, "TV", "Sales")
        assert "Instrumental-variable check" in content
        assert "invalid" in content


class TestInstrumentIntegration:
    """NodeType.INSTRUMENT is wired through validation + translation correctly."""

    def _spec(self, edges, instrument_target_type=NodeType.MEDIA):
        from mmm_framework.dag_model_builder.dag_spec import DAGSpec

        return DAGSpec(
            nodes=[
                DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
                DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
                DAGNode(
                    id="z", variable_name="PolicyShock", node_type=NodeType.INSTRUMENT
                ),
            ],
            edges=[DAGEdge(source=s, target=t) for s, t in edges],
        )

    def test_valid_instrument_dag_passes_validation(self):
        from mmm_framework.dag_model_builder.validation import validate_dag

        spec = self._spec([("z", "tv"), ("tv", "sales")])
        assert validate_dag(spec).valid

    def test_instrument_pointing_to_kpi_is_rejected(self):
        from mmm_framework.dag_model_builder.validation import validate_dag

        spec = self._spec([("z", "tv"), ("z", "sales"), ("tv", "sales")])
        result = validate_dag(spec)
        assert not result.valid
        assert any("INSTRUMENT" in e for e in result.errors)

    def test_dag_to_mff_config_drops_instrument_without_error(self):
        import warnings

        from mmm_framework.dag_model_builder.config_translator import dag_to_mff_config

        spec = self._spec([("z", "tv"), ("tv", "sales")])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # an instrument must not warn
            cfg = dag_to_mff_config(spec)
        # Instrument is identification-only: not a media channel or control.
        assert [m.name for m in cfg.media_channels] == ["TV"]
        assert all(c.name != "PolicyShock" for c in cfg.controls)

    def test_instrument_not_proposed_as_confounder(self):
        # Conditioning on an instrument is bias-amplifying; it must never enter
        # the adjustment set.
        from mmm_framework.dag_model_builder.identification import (
            propose_adjustment_set,
        )

        spec = self._spec([("z", "tv"), ("tv", "sales")])
        assert "z" not in propose_adjustment_set(spec, "tv", "sales")

    def test_instrument_not_required_in_panel_data(self):
        # validate_dag_against_data must skip instruments (they need not be a
        # data column), like mediators.
        from types import SimpleNamespace

        import pandas as pd

        from mmm_framework.dag_model_builder.validation import validate_dag_against_data

        panel = SimpleNamespace(
            y=pd.Series([1.0, 2.0], name="Sales"),
            X_media=pd.DataFrame({"TV": [1.0, 2.0]}),
            X_controls=None,
            coords=SimpleNamespace(channels=["TV"], controls=[]),
        )
        spec = self._spec([("z", "tv"), ("tv", "sales")])
        result = validate_dag_against_data(spec, panel)
        # No error about PolicyShock missing from data.
        assert all("PolicyShock" not in e for e in result.errors)
