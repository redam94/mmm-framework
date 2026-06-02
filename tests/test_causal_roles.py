"""Tests for causal role typing (P1-2) and DAG adjustment-set wiring (P1-1).

These cover the "bad control" guardrails from critique.md §3.1 / §3.4:
- control variables carry a causal role (confounder / precision / mediator /
  collider);
- the core model routes confounders to a wide, un-shrunk prior and *refuses* to
  condition on mediators/colliders;
- the DAG builder infers those roles from the backdoor adjustment set and warns
  when the identification result the fitting path would otherwise ignore is bad.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    CausalControlRole,
    ControlVariableConfig,
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.dag_model_builder.config_translator import dag_to_mff_config
from mmm_framework.dag_model_builder.dag_spec import (
    DAGEdge,
    DAGNode,
    DAGSpec,
    NodeType,
)
from mmm_framework.dag_model_builder.identification import classify_dag_roles
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.model.base import (
    _CONFOUNDER_PRIOR_SIGMA,
    _PRECISION_CONTROL_PRIOR_SIGMA,
)


# =============================================================================
# Config layer (P1-2)
# =============================================================================


class TestCausalRoleConfig:
    def test_default_role_is_none(self):
        cfg = ControlVariableConfig(name="Temp")
        assert cfg.causal_role is None
        assert cfg.causal_role_reason is None

    def test_role_round_trips(self):
        cfg = ControlVariableConfig(
            name="Demand",
            causal_role=CausalControlRole.CONFOUNDER,
            causal_role_reason="in the adjustment set",
        )
        assert cfg.causal_role is CausalControlRole.CONFOUNDER
        assert cfg.causal_role_reason == "in the adjustment set"


# =============================================================================
# Helpers
# =============================================================================


def _panel_with_controls(control_roles: dict[str, CausalControlRole | None]):
    """Build a tiny panel with the given controls and their causal roles."""
    periods = pd.date_range("2021-01-04", periods=30, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(3)
    control_names = list(control_roles)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV"],
        controls=control_names,
    )
    y = pd.Series(100 + rng.normal(0, 5, n), name="Sales")
    X_media = pd.DataFrame({"TV": np.abs(rng.normal(100, 30, n))})
    X_controls = pd.DataFrame({name: rng.normal(0, 1, n) for name in control_names})
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
        ],
        controls=[
            ControlVariableConfig(
                name=name,
                dimensions=[DimensionType.PERIOD],
                causal_role=role,
            )
            for name, role in control_roles.items()
        ],
    )
    return PanelDataset(
        y=y,
        X_media=X_media,
        X_controls=X_controls,
        coords=coords,
        index=periods,
        config=config,
    )


def _model(panel):
    return BayesianMMM(
        panel,
        ModelConfig(inference_method=InferenceMethod.BAYESIAN_PYMC),
        TrendConfig(type=TrendType.LINEAR),
    )


# =============================================================================
# Model enforcement (P1-2)
# =============================================================================


class TestModelControlPriors:
    def test_confounder_gets_wider_prior_than_precision(self):
        # Demand is a confounder (wide, un-shrunk); Price unmarked (precision).
        panel = _panel_with_controls(
            {"Demand": CausalControlRole.CONFOUNDER, "Price": None}
        )
        model = _model(panel)
        roles = model._control_causal_roles
        assert roles == [CausalControlRole.CONFOUNDER, None]

        sigmas = model._control_prior_sigmas()
        assert sigmas[0] == pytest.approx(_CONFOUNDER_PRIOR_SIGMA)
        assert sigmas[1] == pytest.approx(_PRECISION_CONTROL_PRIOR_SIGMA)

        # The realised prior on beta_controls must reflect the per-element sigma.
        prior = model.get_prior(samples=4000, random_seed=0)
        arr = prior.prior["beta_controls"].values
        arr = arr.reshape(-1, arr.shape[-1])
        std = arr.std(axis=0)
        # Confounder column is noticeably wider than the precision column.
        assert std[0] > 2 * std[1]
        assert std[0] == pytest.approx(_CONFOUNDER_PRIOR_SIGMA, rel=0.15)
        assert std[1] == pytest.approx(_PRECISION_CONTROL_PRIOR_SIGMA, rel=0.15)

    def test_unmarked_controls_preserve_default_width(self):
        # Backward compatibility: with no roles, every control keeps sigma=0.5.
        panel = _panel_with_controls({"A": None, "B": None})
        model = _model(panel)
        sigmas = model._control_prior_sigmas()
        assert np.allclose(sigmas, _PRECISION_CONTROL_PRIOR_SIGMA)

    @pytest.mark.parametrize(
        "bad_role", [CausalControlRole.MEDIATOR, CausalControlRole.COLLIDER]
    )
    def test_mediator_or_collider_control_is_refused(self, bad_role):
        panel = _panel_with_controls({"InStore": bad_role, "Price": None})
        with pytest.raises(ValueError, match="post-treatment / collider"):
            _model(panel)

    def test_refusal_names_the_variable_and_reason(self):
        periods = pd.date_range("2021-01-04", periods=20, freq="W-MON")
        n = len(periods)
        rng = np.random.default_rng(1)
        coords = PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV"],
            controls=["InStore"],
        )
        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
            ],
            controls=[
                ControlVariableConfig(
                    name="InStore",
                    dimensions=[DimensionType.PERIOD],
                    causal_role=CausalControlRole.MEDIATOR,
                    causal_role_reason="post-treatment descendant of media 'tv'",
                )
            ],
        )
        panel = PanelDataset(
            y=pd.Series(100 + rng.normal(0, 5, n), name="Sales"),
            X_media=pd.DataFrame({"TV": np.abs(rng.normal(100, 30, n))}),
            X_controls=pd.DataFrame({"InStore": rng.normal(0, 1, n)}),
            coords=coords,
            index=periods,
            config=config,
        )
        with pytest.raises(ValueError) as exc:
            _model(panel)
        msg = str(exc.value)
        assert "InStore" in msg
        assert "post-treatment descendant of media 'tv'" in msg


# =============================================================================
# DAG classification + identification wiring (P1-1)
# =============================================================================


def _confounder_dag():
    """Demand confounds TV->Sales; Price is a precision control; InStore is a
    post-treatment mediator misused as a control."""
    nodes = [
        DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
        DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
        DAGNode(id="demand", variable_name="Demand", node_type=NodeType.CONTROL),
        DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL),
        DAGNode(id="instore", variable_name="InStore", node_type=NodeType.CONTROL),
    ]
    edges = [
        DAGEdge(source="tv", target="sales"),
        DAGEdge(source="demand", target="tv"),
        DAGEdge(source="demand", target="sales"),
        DAGEdge(source="price", target="sales"),
        DAGEdge(source="tv", target="instore"),
        DAGEdge(source="instore", target="sales"),
    ]
    return DAGSpec(nodes=nodes, edges=edges)


class TestDagRoleClassification:
    def test_roles_assigned_from_topology(self):
        cfg = dag_to_mff_config(_confounder_dag())
        roles = {c.name: c.causal_role for c in cfg.controls}
        assert roles["Demand"] is CausalControlRole.CONFOUNDER
        assert roles["Price"] is CausalControlRole.PRECISION_CONTROL
        assert roles["InStore"] is CausalControlRole.MEDIATOR

    def test_post_treatment_role_names_the_treatment(self):
        cfg = dag_to_mff_config(_confounder_dag())
        instore = next(c for c in cfg.controls if c.name == "InStore")
        assert "tv" in (instore.causal_role_reason or "")

    def test_post_outcome_control_is_refused_as_post_treatment(self):
        # A control that is a *consequence* of the KPI (Sales -> Downstream) is
        # also a consequence of the media driving Sales, so it is caught by the
        # exact post-treatment check and refused (labeled mediator). Either way
        # it is a bad control the model must reject.
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(
                id="downstream",
                variable_name="Downstream",
                node_type=NodeType.CONTROL,
            ),
        ]
        edges = [
            DAGEdge(source="tv", target="sales"),
            DAGEdge(source="sales", target="downstream"),
        ]
        classification = classify_dag_roles(
            DAGSpec(nodes=nodes, edges=edges), ["tv"], "sales"
        )
        role, _ = classification.role_for("downstream")
        assert role == "mediator"  # post-treatment subsumes post-outcome
        # And it surfaces in the config as a blocked role.
        cfg = dag_to_mff_config(DAGSpec(nodes=nodes, edges=edges))
        downstream = next(c for c in cfg.controls if c.name == "Downstream")
        assert downstream.causal_role is CausalControlRole.MEDIATOR

    def test_multi_treatment_post_treatment_wins_over_confounder(self):
        # Demand confounds both TV and Digital. Funnel is post-treatment for TV
        # (TV -> Funnel -> Sales). Across treatments, post-treatment must win.
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="digital", variable_name="Digital", node_type=NodeType.MEDIA),
            DAGNode(id="demand", variable_name="Demand", node_type=NodeType.CONTROL),
            DAGNode(id="funnel", variable_name="Funnel", node_type=NodeType.CONTROL),
            DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL),
        ]
        edges = [
            DAGEdge(source="tv", target="sales"),
            DAGEdge(source="digital", target="sales"),
            DAGEdge(source="demand", target="tv"),
            DAGEdge(source="demand", target="digital"),
            DAGEdge(source="demand", target="sales"),
            DAGEdge(source="tv", target="funnel"),
            DAGEdge(source="funnel", target="sales"),
            DAGEdge(source="price", target="sales"),
        ]
        classification = classify_dag_roles(
            DAGSpec(nodes=nodes, edges=edges), ["tv", "digital"], "sales"
        )
        assert classification.role_for("demand")[0] == "confounder"
        assert classification.role_for("price")[0] == "precision_control"
        funnel_role, funnel_reason = classification.role_for("funnel")
        assert funnel_role == "mediator"
        assert "tv" in (funnel_reason or "")

    def _m_bias_dag(self):
        # M-bias: A -> TV, A -> C, B -> C, B -> Sales, TV -> Sales.
        # The back-door path TV <- A -> C <- B -> Sales has C as a collider.
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="a", variable_name="A", node_type=NodeType.CONTROL),
            DAGNode(id="b", variable_name="B", node_type=NodeType.CONTROL),
            DAGNode(id="c", variable_name="C", node_type=NodeType.CONTROL),
        ]
        edges = [
            DAGEdge(source="a", target="tv"),
            DAGEdge(source="a", target="c"),
            DAGEdge(source="b", target="c"),
            DAGEdge(source="b", target="sales"),
            DAGEdge(source="tv", target="sales"),
        ]
        return DAGSpec(nodes=nodes, edges=edges)

    def test_collider_auto_detected_when_blocking_fork_not_conditioned(self):
        # Only C is conditioned on; the forks A, B are not -> conditioning on the
        # collider C OPENS the back-door path, so C must be flagged a collider.
        classification = classify_dag_roles(
            self._m_bias_dag(), ["tv"], "sales", control_ids=["c"]
        )
        assert "c" in classification.colliders
        role, reason = classification.role_for("c")
        assert role == "collider"
        assert "spurious" in (reason or "")

    def test_collider_not_flagged_when_blocking_fork_is_conditioned(self):
        # All of A, B, C are conditioned. The fork A blocks the path, so
        # conditioning on C is harmless -> C must NOT be flagged a collider.
        classification = classify_dag_roles(
            self._m_bias_dag(), ["tv"], "sales", control_ids=["a", "b", "c"]
        )
        assert "c" not in classification.colliders
        assert classification.role_for("c")[0] != "collider"

    def test_collider_role_reaches_config_via_translator(self):
        # End-to-end through dag_to_mff_config: when the fork parents are not
        # controls (here typed as mediators, so not conditioned), the collider C
        # is the only control and is tagged COLLIDER in the emitted config.
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="a", variable_name="A", node_type=NodeType.MEDIATOR),
            DAGNode(id="b", variable_name="B", node_type=NodeType.MEDIATOR),
            DAGNode(id="c", variable_name="C", node_type=NodeType.CONTROL),
        ]
        edges = [
            DAGEdge(source="a", target="tv"),
            DAGEdge(source="a", target="c"),
            DAGEdge(source="b", target="c"),
            DAGEdge(source="b", target="sales"),
            DAGEdge(source="tv", target="sales"),
        ]
        cfg = dag_to_mff_config(DAGSpec(nodes=nodes, edges=edges))
        c = next(ctrl for ctrl in cfg.controls if ctrl.name == "C")
        assert c.causal_role is CausalControlRole.COLLIDER

    def test_unidentified_when_no_treatments(self):
        # No media nodes -> nothing to classify; all controls stay unknown.
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL),
        ]
        cfg = dag_to_mff_config(DAGSpec(nodes=nodes, edges=[]))
        assert all(c.causal_role is None for c in cfg.controls)


class TestIdentificationWarnings:
    def test_missing_confounder_warns(self):
        # Demand confounds but is not a control node -> open backdoor.
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="demand", variable_name="Demand", node_type=NodeType.MEDIATOR),
            DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL),
        ]
        edges = [
            DAGEdge(source="tv", target="sales"),
            DAGEdge(source="demand", target="tv"),
            DAGEdge(source="demand", target="sales"),
            DAGEdge(source="price", target="sales"),
        ]
        with pytest.warns(UserWarning, match="adjustment set but are NOT included"):
            dag_to_mff_config(DAGSpec(nodes=nodes, edges=edges))

    def test_well_specified_dag_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Demand IS a control here, so the backdoor is closed -> no warning.
            dag_to_mff_config(_confounder_dag())

    def test_enforce_flag_can_be_disabled(self):
        cfg = dag_to_mff_config(_confounder_dag(), enforce_identification=False)
        assert all(c.causal_role is None for c in cfg.controls)

    def test_unidentified_effect_warns(self):
        # Drive the "not identified" branch via the warning helper with a
        # classification whose identifiable flag is False (the heuristic
        # adjustment set is robust enough that crafting an open-backdoor DAG is
        # awkward; the warning logic is what we are covering here).
        from mmm_framework.dag_model_builder.config_translator import (
            _warn_on_identification,
        )
        from mmm_framework.dag_model_builder.identification import (
            DagRoleClassification,
        )

        dag = _confounder_dag()
        classification = DagRoleClassification(
            confounders=set(),
            post_treatment={},
            colliders={},
            adjustment_set=[],
            identifiable=False,
            notes=["open backdoor path remains"],
        )
        with pytest.warns(UserWarning, match="NOT identified"):
            _warn_on_identification(
                dag, classification, {c.variable_name for c in dag.control_nodes}
            )


# =============================================================================
# End-to-end: DAG-classified mediator propagates to a model refusal
# =============================================================================


class TestDagToModelEnforcement:
    def test_dag_classified_mediator_is_refused_at_model_construction(self):
        # The config produced by the DAG translator carries the inferred roles;
        # feeding it to the model must refuse the post-treatment control. This is
        # the P1-1 -> P1-2 hand-off: identification result -> config -> fitting.
        dag = _confounder_dag()
        config = dag_to_mff_config(dag)
        assert (
            next(c for c in config.controls if c.name == "InStore").causal_role
            is CausalControlRole.MEDIATOR
        )

        periods = pd.date_range("2021-01-04", periods=30, freq="W-MON")
        n = len(periods)
        rng = np.random.default_rng(7)
        control_names = [c.name for c in config.controls]
        coords = PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV"],
            controls=control_names,
        )
        panel = PanelDataset(
            y=pd.Series(100 + rng.normal(0, 5, n), name="Sales"),
            X_media=pd.DataFrame({"TV": np.abs(rng.normal(100, 30, n))}),
            X_controls=pd.DataFrame(
                {name: rng.normal(0, 1, n) for name in control_names}
            ),
            coords=coords,
            index=periods,
            config=config,
        )
        with pytest.raises(ValueError, match="post-treatment / collider"):
            _model(panel)


# =============================================================================
# Agent guidance: the causal-identification tool explains roles to non-experts
# =============================================================================


class TestAgentRoleGuidance:
    def _invoke(self, dag: DAGSpec, treatment="TV", outcome="Sales") -> str:
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

    def test_roles_section_explains_each_control(self):
        content = self._invoke(_confounder_dag())
        assert "Control variable roles (auto-detected)" in content
        # Confounder kept, mediator flagged for removal, precision kept.
        assert "`Demand` — **confounder**" in content
        assert "`InStore` — **mediator**" in content and "⛔ REMOVE" in content
        assert "`Price` — **precision control**" in content
        # Plain-language guidance for non-experts on what to do with bad controls.
        assert "bad controls" in content

    def test_no_roles_section_without_controls(self):
        nodes = [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
        ]
        edges = [DAGEdge(source="tv", target="sales")]
        content = self._invoke(DAGSpec(nodes=nodes, edges=edges))
        assert "Control variable roles" not in content
