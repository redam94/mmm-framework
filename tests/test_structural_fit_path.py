"""StructuralNestedMMM through the agent fit path.

The DAG resolver upgrades to STRUCTURAL_NESTED_MMM when the DAG uses features
only the structural model can express (mediator->mediator chains, per-mediator
likelihood/dynamics keys, control->mediator drivers, latent factors);
``agents/fitting.build_model`` routes ``dag_model_type='structural_nested_mmm'``
through DAGModelBuilder, which pulls the survey series (NaN = unobserved week)
from the raw MFF table and translates node configs into MediatorSpecs.
"""

import numpy as np
import pytest

from mmm_framework.agents.fitting import (
    build_model,
    unconsumed_prior_path,
    unconsumed_spec_path,
)
from mmm_framework.dag_model_builder.dag_spec import (
    DAGEdge,
    DAGNode,
    DAGSpec,
    NodeType,
)
from mmm_framework.dag_model_builder.model_type_resolver import (
    ModelType,
    describe_model_type,
    resolve_model_type,
)
from mmm_framework.synth.mff import brand_funnel_mff

# ---------------------------------------------------------------------------
# fixtures / spec builders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def funnel_mff(tmp_path_factory) -> tuple[str, dict]:
    mff, key = brand_funnel_mff(seed=21, n_weeks=60)
    path = tmp_path_factory.mktemp("structural_fit") / "funnel_mff.csv"
    mff.to_csv(path, index=False)
    return str(path), key


def _funnel_dag() -> DAGSpec:
    """The motivating brand-funnel DAG: TV -> awareness (binomial AR1 tracker)
    -> consideration (Likert, + Display/Social/Price/demand) -> Sales."""
    nodes = [
        DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
        DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
        DAGNode(id="display", variable_name="Display", node_type=NodeType.MEDIA),
        DAGNode(id="social", variable_name="Social", node_type=NodeType.MEDIA),
        DAGNode(id="search", variable_name="Search", node_type=NodeType.MEDIA),
        DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL),
        DAGNode(
            id="awareness",
            variable_name="awareness",
            node_type=NodeType.MEDIATOR,
            config={
                "likelihood": "binomial",
                "trials_variable": "awareness_trials",
                "dynamics": "ar1",
                "rho_prior_alpha": 9.0,
                "rho_prior_beta": 1.5,
                # affects_outcome deliberately NOT set: the DAG-faithful
                # default derives False from the absence of an awareness ->
                # Sales edge (design-review fix).
                "direct_effect_sigma": 0.3,
            },
        ),
        DAGNode(
            id="consideration",
            variable_name="consideration",
            node_type=NodeType.MEDIATOR,
            config={
                "likelihood": "ordered",
                "category_variables": [f"consideration_cat_{k}" for k in range(1, 6)],
                "latent_factors": ["demand"],
            },
        ),
    ]
    edges = [
        DAGEdge(source="tv", target="awareness"),
        DAGEdge(source="awareness", target="consideration"),
        DAGEdge(source="display", target="consideration"),
        DAGEdge(source="social", target="consideration"),
        DAGEdge(source="price", target="consideration"),
        DAGEdge(source="price", target="sales"),
        DAGEdge(source="consideration", target="sales"),
        DAGEdge(source="search", target="sales"),
    ]
    return DAGSpec(
        nodes=nodes,
        edges=edges,
        # Direct DAGModelBuilder users declare factors here; the agent path
        # overwrites this from spec["latent_factors"] when present.
        metadata={"latent_factors": [{"name": "demand", "affects_outcome": True}]},
    )


def _structural_spec() -> dict:
    dag = _funnel_dag()
    # awareness_count is the counts column; the mediator's variable_name must
    # match the MFF variable carrying the counts.
    dag_dict = dag.model_dump()
    for node in dag_dict["nodes"]:
        if node["id"] == "awareness":
            node["variable_name"] = "awareness_count"
    return {
        "kpi": "Sales",
        "media_channels": [
            {"name": "TV"},
            {"name": "Display"},
            {"name": "Social"},
            {"name": "Search"},
        ],
        "control_variables": [{"name": "Price"}],
        "time_granularity": "weekly",
        "dag_model_type": "structural_nested_mmm",
        "dag_spec": dag_dict,
        "latent_factors": [{"name": "demand", "affects_outcome": True}],
        "inference": {"chains": 1, "draws": 40, "tune": 40, "target_accept": 0.8},
    }


# ---------------------------------------------------------------------------
# resolver upgrade rules
# ---------------------------------------------------------------------------


class TestResolver:
    def _base_nodes(self):
        return [
            DAGNode(id="sales", variable_name="Sales", node_type=NodeType.KPI),
            DAGNode(id="tv", variable_name="TV", node_type=NodeType.MEDIA),
            DAGNode(id="aw", variable_name="Aw", node_type=NodeType.MEDIATOR),
        ]

    def test_plain_mediator_dag_stays_nested(self):
        dag = DAGSpec(
            nodes=self._base_nodes(),
            edges=[
                DAGEdge(source="tv", target="aw"),
                DAGEdge(source="aw", target="sales"),
            ],
        )
        assert resolve_model_type(dag) == ModelType.NESTED_MMM

    def test_mediator_chain_upgrades(self):
        nodes = self._base_nodes() + [
            DAGNode(id="cons", variable_name="Cons", node_type=NodeType.MEDIATOR)
        ]
        dag = DAGSpec(
            nodes=nodes,
            edges=[
                DAGEdge(source="tv", target="aw"),
                DAGEdge(source="aw", target="cons"),
                DAGEdge(source="cons", target="sales"),
            ],
        )
        assert resolve_model_type(dag) == ModelType.STRUCTURAL_NESTED_MMM

    def test_structural_config_key_upgrades(self):
        nodes = self._base_nodes()
        nodes[2] = DAGNode(
            id="aw",
            variable_name="Aw",
            node_type=NodeType.MEDIATOR,
            config={"dynamics": "ar1"},
        )
        dag = DAGSpec(
            nodes=nodes,
            edges=[
                DAGEdge(source="tv", target="aw"),
                DAGEdge(source="aw", target="sales"),
            ],
        )
        assert resolve_model_type(dag) == ModelType.STRUCTURAL_NESTED_MMM

    def test_control_to_mediator_edge_upgrades(self):
        nodes = self._base_nodes() + [
            DAGNode(id="price", variable_name="Price", node_type=NodeType.CONTROL)
        ]
        dag = DAGSpec(
            nodes=nodes,
            edges=[
                DAGEdge(source="tv", target="aw"),
                DAGEdge(source="aw", target="sales"),
                DAGEdge(source="price", target="aw"),
            ],
        )
        assert resolve_model_type(dag) == ModelType.STRUCTURAL_NESTED_MMM

    def test_describe_model_type_covers_every_member(self):
        """describe_model_type crashed with KeyError for the structural type
        (blocker: build_model_from_dag calls it for every DAG); pin that the
        funnel DAG describes cleanly and that no enum member can crash it."""
        desc = describe_model_type(_funnel_dag())
        assert "StructuralNestedMMM" in desc

    def test_affects_outcome_derived_from_edges(self):
        """A mediator without an edge into the KPI must not get a gamma path
        (the MediatorSpec default True would add one the DAG doesn't draw)."""
        from mmm_framework.dag_model_builder.config_translator import (
            dag_to_structural_config,
        )

        cfg, _ = dag_to_structural_config(_funnel_dag())
        by_name = {m.name: m for m in cfg.mediators}
        assert by_name["awareness"].affects_outcome is False  # no edge to Sales
        assert by_name["consideration"].affects_outcome is True

    def test_metadata_override_upgrades(self):
        dag = DAGSpec(
            nodes=self._base_nodes(),
            edges=[
                DAGEdge(source="tv", target="aw"),
                DAGEdge(source="aw", target="sales"),
            ],
            metadata={"model_type": "structural_nested_mmm"},
        )
        assert resolve_model_type(dag) == ModelType.STRUCTURAL_NESTED_MMM

    def test_structural_plus_multi_outcome_rejected(self):
        nodes = self._base_nodes() + [
            DAGNode(id="out2", variable_name="Sales2", node_type=NodeType.OUTCOME)
        ]
        nodes[2] = DAGNode(
            id="aw",
            variable_name="Aw",
            node_type=NodeType.MEDIATOR,
            config={"dynamics": "ar1"},
        )
        dag = DAGSpec(
            nodes=nodes,
            edges=[
                DAGEdge(source="tv", target="aw"),
                DAGEdge(source="aw", target="sales"),
                DAGEdge(source="tv", target="out2"),
            ],
        )
        with pytest.raises(ValueError, match="single-outcome"):
            resolve_model_type(dag)


# ---------------------------------------------------------------------------
# build_model E2E
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_routes_to_structural(self, funnel_mff):
        path, key = funnel_mff
        mmm = build_model(_structural_spec(), path)
        assert type(mmm).__name__ == "StructuralNestedMMM"
        assert mmm.channel_names == ["TV", "Display", "Social", "Search"]
        assert mmm.mediator_names == ["awareness_count", "consideration"]

    def test_config_translation(self, funnel_mff):
        path, _ = funnel_mff
        mmm = build_model(_structural_spec(), path)
        aw = mmm._med_by_name["awareness_count"]
        cons = mmm._med_by_name["consideration"]
        assert aw.dynamics.value == "ar1"
        assert aw.measurement.likelihood.value == "binomial"
        assert aw.affects_outcome is False
        assert aw.channels == ("TV",)
        assert aw.adstock_enabled is False  # AR1 auto-resolves adstock off
        assert cons.measurement.likelihood.value == "ordered"
        assert cons.measurement.n_categories == 5
        assert cons.parents == ("awareness_count",)
        assert cons.controls == ("Price",)
        assert cons.latent_factors == ("demand",)
        assert [f.name for f in mmm.config.latent_factors] == ["demand"]
        assert mmm.config.outcome_controls == ("Price",)

    def test_survey_masks_from_sparse_mff(self, funnel_mff):
        """NaN weeks in the MFF (no survey rows) become unobserved masks."""
        path, key = funnel_mff
        mmm = build_model(_structural_spec(), path)
        aw_mask = mmm.mediator_masks["awareness_count"]
        cons_mask = mmm.mediator_masks["consideration"]
        assert 0 < aw_mask.sum() < len(aw_mask)  # sparse, not fully covered
        assert 0 < cons_mask.sum() < len(cons_mask)
        trials = mmm.mediator_trials["awareness_count"]
        assert np.all(trials[aw_mask] > 0)

    def test_graph_builds_and_map_fits(self, funnel_mff):
        path, _ = funnel_mff
        mmm = build_model(_structural_spec(), path)
        with pytest.warns(UserWarning, match="Approximate fits"):
            results = mmm.fit(method="map", random_seed=0)
        assert results.approximate is True

    def test_prior_injection_reaches_mediator_spec(self, funnel_mff):
        path, _ = funnel_mff
        spec = _structural_spec()
        spec["priors"] = {
            "mediator": {
                "awareness_count": {
                    "rho_prior_alpha": 3.0,
                    "media_effect_sigma": 2.5,
                }
            }
        }
        mmm = build_model(spec, path)
        aw = mmm._med_by_name["awareness_count"]
        assert aw.rho_prior_alpha == 3.0
        assert aw.media_effect.sigma == 2.5


# ---------------------------------------------------------------------------
# registry validation
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_structural_mediator_prior_keys_accepted(self):
        spec = _structural_spec()
        err = unconsumed_prior_path(
            ["priors", "mediator", "awareness_count", "innovation_sigma"],
            0.5,
            spec,
        )
        assert err is None

    def test_structural_key_rejected_for_plain_nested(self):
        spec = _structural_spec()
        spec["dag_model_type"] = "nested_mmm"
        err = unconsumed_prior_path(
            ["priors", "mediator", "awareness_count", "innovation_sigma"],
            0.5,
            spec,
        )
        assert err is not None and "innovation_sigma" in err

    def test_latent_factors_spec_key_accepted_for_structural(self):
        spec = _structural_spec()
        err = unconsumed_spec_path(
            ["latent_factors"],
            [{"name": "demand", "affects_outcome": True}],
            spec,
        )
        assert err is None

    def test_latent_factors_rejected_for_plain_spec(self):
        err = unconsumed_spec_path(
            ["latent_factors"],
            [{"name": "demand"}],
            {"dag_model_type": "nested_mmm"},
        )
        assert err is not None and "structural" in err

    def test_latent_factors_unknown_key_rejected(self):
        spec = _structural_spec()
        err = unconsumed_spec_path(
            ["latent_factors"],
            [{"name": "demand", "bogus_knob": 1}],
            spec,
        )
        assert err is not None and "bogus_knob" in err

    def test_latent_factor_subpath_accepted(self):
        """The update_model_setting idiom: latent_factors.<name>.<key>."""
        spec = _structural_spec()
        assert (
            unconsumed_spec_path(
                ["latent_factors", "demand", "affects_outcome"], True, spec
            )
            is None
        )
        err = unconsumed_spec_path(["latent_factors", "demand", "bogus"], 1.0, spec)
        assert err is not None and "bogus" in err
        err2 = unconsumed_spec_path(
            ["latent_factors", "demand"], {"affects_outcome": True}, spec
        )
        assert err2 is not None and "latent_factors.demand.<key>" in err2

    def test_latent_factors_dict_value_rejected(self):
        spec = _structural_spec()
        err = unconsumed_spec_path(["latent_factors"], {"name": "demand"}, spec)
        assert err is not None and "LIST" in err
