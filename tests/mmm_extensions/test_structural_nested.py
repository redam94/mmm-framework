"""Tests for StructuralNestedMMM -- the multi-mediator structural MMM.

Fast tests build graphs without MCMC (RV structure, validation, prior
predictive, MAP smoke, counterfactual mechanics). The slow recovery test fits
the ``make_brand_funnel`` ground-truth world with NUTS and asserts the funnel
structure is recovered (signs, persistence, latent correlations, and -- per the
design review -- a NONZERO mediated total, the failure mode the in-graph
centering blocker would have produced).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.mmm_extensions import (
    EffectPriorConfig,
    LatentFactorSpec,
    MediatorDynamics,
    MediatorLikelihood,
    MediatorMeasurement,
    MediatorSpec,
    StructuralNestedConfig,
    binary_survey_mediator,
    latent_demand_factor,
    likert_mediator,
)
from mmm_framework.synth.dgp import make_brand_funnel

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _funnel_config(**overrides) -> StructuralNestedConfig:
    kwargs = dict(
        mediators=(
            binary_survey_mediator(
                "awareness", ["TV"], persistence="high", affects_outcome=False
            ),
            likert_mediator(
                "consideration",
                ["Display", "Social"],
                parents=["awareness"],
                controls=["Price"],
                latent_factors=["demand"],
                n_categories=5,
            ),
        ),
        latent_factors=(latent_demand_factor("demand"),),
    )
    kwargs.update(overrides)
    return StructuralNestedConfig(**kwargs)


def _funnel_model(n_weeks: int = 60, seed: int = 21, **model_kwargs):
    from mmm_framework.mmm_extensions import StructuralNestedMMM

    sc = make_brand_funnel(seed=seed, n_weeks=n_weeks)
    cfg = model_kwargs.pop("config", _funnel_config())
    med_names = {m.name for m in cfg.mediators}
    data = {
        "awareness": sc.notes["awareness_counts"],
        "consideration": sc.notes["consideration_counts"],
    }
    trials = {"awareness": sc.notes["awareness_trials"]}
    model = StructuralNestedMMM(
        sc.spend.to_numpy(float),
        sc.y.to_numpy(float),
        sc.channels,
        cfg,
        mediator_data={k: v for k, v in data.items() if k in med_names},
        mediator_trials={k: v for k, v in trials.items() if k in med_names},
        X_controls=sc.controls.to_numpy(float),
        control_names=["Price"],
        index=sc.weeks,
        **model_kwargs,
    )
    return model, sc


def _rv_names(pymc_model) -> set[str]:
    return set(pymc_model.named_vars)


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_motivating_example_topological_order(self):
        cfg = _funnel_config()
        assert cfg.topological_order() == ["awareness", "consideration"]

    def test_cycle_rejected(self):
        a = MediatorSpec(name="a", parents=("b",), channels=("TV",))
        b = MediatorSpec(name="b", parents=("a",), channels=("TV",))
        with pytest.raises(ValueError, match="cycle"):
            StructuralNestedConfig(mediators=(a, b))

    def test_self_parent_rejected(self):
        with pytest.raises(ValueError, match="own parent"):
            MediatorSpec(name="a", parents=("a",), channels=("TV",))

    def test_unknown_parent_rejected(self):
        a = MediatorSpec(name="a", parents=("ghost",), channels=("TV",))
        with pytest.raises(ValueError, match="unknown parents"):
            StructuralNestedConfig(mediators=(a,))

    def test_unknown_factor_rejected(self):
        a = MediatorSpec(name="a", channels=("TV",), latent_factors=("ghost",))
        with pytest.raises(ValueError, match="unknown latent factors"):
            StructuralNestedConfig(mediators=(a,))

    def test_duplicate_names_rejected(self):
        a = MediatorSpec(name="a", channels=("TV",))
        with pytest.raises(ValueError, match="Duplicate"):
            StructuralNestedConfig(mediators=(a, a))

    def test_ordered_requires_categories(self):
        with pytest.raises(ValueError, match="n_categories"):
            MediatorMeasurement(likelihood=MediatorLikelihood.ORDERED)

    def test_design_effect_below_one_rejected(self):
        with pytest.raises(ValueError, match="design_effect"):
            MediatorMeasurement(design_effect=0.5)

    def test_latent_mediator_without_drivers_rejected(self):
        with pytest.raises(ValueError, match="no drivers"):
            MediatorSpec(
                name="a",
                measurement=MediatorMeasurement(likelihood=MediatorLikelihood.LATENT),
            )

    def test_factor_needs_two_observation_channels(self):
        # outcome-only factor: unidentified residual absorber
        a = MediatorSpec(name="a", channels=("TV",))
        with pytest.raises(ValueError, match="observation channels"):
            StructuralNestedConfig(
                mediators=(a,),
                latent_factors=(LatentFactorSpec(name="demand"),),
            )
        # single measured mediator, no outcome: confounded with its noise
        b = MediatorSpec(name="b", channels=("TV",), latent_factors=("demand",))
        with pytest.raises(ValueError, match="observation channels"):
            StructuralNestedConfig(
                mediators=(b,),
                latent_factors=(
                    LatentFactorSpec(name="demand", affects_outcome=False),
                ),
            )

    def test_state_parameterization_validated(self):
        with pytest.raises(ValueError, match="state_parameterization"):
            MediatorSpec(name="a", channels=("TV",), state_parameterization="bogus")

    def test_adstock_default_resolves_by_dynamics(self):
        static = MediatorSpec(name="a", channels=("TV",))
        ar = MediatorSpec(name="b", channels=("TV",), dynamics=MediatorDynamics.AR1)
        assert static.adstock_enabled is True
        assert ar.adstock_enabled is False
        forced = MediatorSpec(
            name="c",
            channels=("TV",),
            dynamics=MediatorDynamics.AR1,
            apply_adstock=True,
        )
        assert forced.adstock_enabled is True


# ---------------------------------------------------------------------------
# model input validation
# ---------------------------------------------------------------------------


class TestModelValidation:
    def test_unknown_channel_rejected(self):
        cfg = StructuralNestedConfig(
            mediators=(MediatorSpec(name="m", channels=("Ghost",)),)
        )
        with pytest.raises(ValueError, match="unknown channels"):
            _funnel_model(config=cfg)

    def test_binomial_requires_trials(self):
        from mmm_framework.mmm_extensions import StructuralNestedMMM

        sc = make_brand_funnel(seed=21, n_weeks=60)
        cfg = _funnel_config()
        with pytest.raises(ValueError, match="mediator_trials"):
            StructuralNestedMMM(
                sc.spend.to_numpy(float),
                sc.y.to_numpy(float),
                sc.channels,
                cfg,
                mediator_data={
                    "awareness": sc.notes["awareness_counts"],
                    "consideration": sc.notes["consideration_counts"],
                },
                X_controls=sc.controls.to_numpy(float),
                control_names=["Price"],
            )

    def test_counts_exceeding_trials_rejected(self):
        model, sc = _funnel_model()
        bad = sc.notes["awareness_counts"].copy()
        obs = np.isfinite(bad)
        bad[obs] = sc.notes["awareness_trials"][obs] + 5
        from mmm_framework.mmm_extensions import StructuralNestedMMM

        with pytest.raises(ValueError, match="exceed trials"):
            StructuralNestedMMM(
                sc.spend.to_numpy(float),
                sc.y.to_numpy(float),
                sc.channels,
                _funnel_config(),
                mediator_data={
                    "awareness": bad,
                    "consideration": sc.notes["consideration_counts"],
                },
                mediator_trials={"awareness": sc.notes["awareness_trials"]},
                X_controls=sc.controls.to_numpy(float),
                control_names=["Price"],
            )

    def test_name_collision_with_channel_rejected(self):
        cfg = StructuralNestedConfig(
            mediators=(MediatorSpec(name="TV", channels=("TV",)),)
        )
        with pytest.raises(ValueError, match="collides"):
            _funnel_model(config=cfg)

    def test_controls_without_names_rejected(self):
        from mmm_framework.mmm_extensions import StructuralNestedMMM

        sc = make_brand_funnel(seed=21, n_weeks=60)
        with pytest.raises(ValueError, match="control_names"):
            StructuralNestedMMM(
                sc.spend.to_numpy(float),
                sc.y.to_numpy(float),
                sc.channels,
                StructuralNestedConfig(
                    mediators=(MediatorSpec(name="m", channels=("TV",)),)
                ),
                X_controls=sc.controls.to_numpy(float),
            )


# ---------------------------------------------------------------------------
# graph structure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    @pytest.fixture(scope="class")
    def built(self):
        model, sc = _funnel_model()
        return model, model.model, sc

    def test_expected_rvs_present(self, built):
        _, graph, _ = built
        names = _rv_names(graph)
        expected = {
            # media transforms
            "alpha_TV",
            "lambda_TV",
            # awareness equation (AR1 binomial, no adstock via factory;
            # 75% survey coverage -> auto resolves to the CENTERED AR noise)
            "beta_TV_to_awareness",
            "awareness_persistence",
            "awareness_state_noise",
            "awareness_innovation_sigma",
            "level_awareness",
            "awareness_latent",
            "awareness_probability",
            "awareness_obs",
            # consideration equation (static ordered)
            "beta_Display_to_consideration",
            "beta_Social_to_consideration",
            "lambda_awareness_to_consideration",
            "phi_Price_to_consideration",
            "w_demand_to_consideration",
            "consideration_cutpoint_anchor",
            "consideration_cutpoint_gaps",
            "consideration_cutpoints",
            "consideration_latent",
            "consideration_obs",
            # STATIC + ORDERED keeps overdispersion slack
            "consideration_innovation",
            "consideration_innovation_sigma",
            # latent factor
            "demand_innovation",
            "demand_persistence",
            "factor_demand",
            "w_demand_to_y",
            # outcome
            "alpha_y",
            "gamma_consideration",
            "delta_direct_TV",
            "delta_direct_Display",
            "delta_direct_Social",
            "beta_Search",
            "beta_ctrl_Price",
            "sigma_y",
            "mu",
            "y_obs",
            # report contract
            "effect_consideration_on_y",
            "direct_effect_TV",
            "direct_effect_Search",
            "controls_total",
            "effect_factor_demand_on_y",
            "indirect_TV_via_consideration",
            "indirect_Display_via_consideration",
        }
        missing = expected - names
        assert not missing, f"missing RVs: {sorted(missing)}"

    def test_funnel_semantics(self, built):
        _, graph, _ = built
        names = _rv_names(graph)
        # awareness does not feed the outcome directly (affects_outcome=False)
        assert "gamma_awareness" not in names
        assert "indirect_TV_via_awareness" not in names
        # ordered mediator has no free level (cutpoints absorb location)
        assert "level_consideration" not in names

    def test_no_dead_free_rvs(self, built):
        _, graph, _ = built
        from pytensor.graph.traversal import ancestors

        anc = set(ancestors(graph.observed_RVs))
        dead = [rv.name for rv in graph.free_RVs if rv not in anc]
        assert not dead, f"dead free RVs: {dead}"

    def test_logp_and_grad_finite(self, built):
        _, graph, _ = built
        ip = graph.initial_point()
        assert np.isfinite(graph.compile_logp()(ip))
        assert np.all(np.isfinite(graph.compile_dlogp()(ip)))

    def test_prior_predictive_cutpoints_ordered(self, built):
        model, _, _ = built
        prior = model.sample_prior_predictive(samples=100, random_seed=0)
        cp = prior.prior["consideration_cutpoints"].values
        assert np.all(np.diff(cp, axis=-1) >= 0)

    def test_adstock_on_ar1_warns(self):
        cfg = StructuralNestedConfig(
            mediators=(
                MediatorSpec(
                    name="m",
                    channels=("TV",),
                    dynamics=MediatorDynamics.AR1,
                    apply_adstock=True,
                    measurement=MediatorMeasurement(
                        likelihood=MediatorLikelihood.LATENT
                    ),
                ),
            )
        )
        model, _ = _funnel_model(config=cfg)
        with pytest.warns(UserWarning, match="carryover"):
            model.model  # noqa: B018 -- graph builds lazily

    def test_no_dead_adstock_alpha_for_fully_mediated_ar_channel(self):
        """A channel routed ONLY to a non-adstock AR equation with no direct
        outcome path must not get an adstock alpha RV (it would be a dead
        free RV sampling its prior -- confirmed code-review finding)."""
        cfg = _funnel_config(
            mediators=(
                binary_survey_mediator(
                    "awareness",
                    ["TV"],
                    persistence="high",
                    affects_outcome=False,
                    allow_direct_effect=False,
                ),
                likert_mediator(
                    "consideration",
                    ["Display", "Social"],
                    parents=["awareness"],
                    controls=["Price"],
                    latent_factors=["demand"],
                    n_categories=5,
                ),
            )
        )
        model, _ = _funnel_model(config=cfg)
        names = _rv_names(model.model)
        assert "alpha_TV" not in names
        assert "lambda_TV" in names  # saturation is still consumed
        assert "delta_direct_TV" not in names
        # channels with adstocked consumers keep their alpha
        assert "alpha_Display" in names and "alpha_Search" in names

    def test_random_walk_has_no_level(self):
        cfg = StructuralNestedConfig(
            mediators=(
                MediatorSpec(
                    name="m",
                    channels=("TV",),
                    dynamics=MediatorDynamics.RANDOM_WALK,
                    measurement=MediatorMeasurement(
                        likelihood=MediatorLikelihood.LATENT
                    ),
                ),
            )
        )
        model, _ = _funnel_model(config=cfg)
        names = _rv_names(model.model)
        assert "level_m" not in names


# ---------------------------------------------------------------------------
# experiment calibration eligibility
# ---------------------------------------------------------------------------


class TestExperimentHandles:
    def test_exactness_rules(self):
        model, _ = _funnel_model()
        # Search feeds no mediator -> exact (direct only)
        assert model._channel_mediated_exact("Search") is True
        # TV routes through an AR1 binomial mediator -> inexact
        assert model._channel_mediated_exact("TV") is False
        # Display routes through STATIC ordered consideration with adstock -> exact
        assert model._channel_mediated_exact("Display") is True

    def test_experiment_on_inexact_channel_skipped_with_warning(self):
        from mmm_framework.calibration.likelihood import ExperimentMeasurement

        model, _ = _funnel_model()
        model.add_experiment_calibration(
            [
                ExperimentMeasurement(
                    channel="TV",
                    test_period=(10, 20),
                    value=500.0,
                    se=100.0,
                )
            ]
        )
        with pytest.warns(UserWarning, match="calibration skipped"):
            model.model  # noqa: B018

    def test_experiment_on_direct_channel_attaches(self):
        from mmm_framework.calibration.likelihood import ExperimentMeasurement

        model, _ = _funnel_model()
        model.add_experiment_calibration(
            [
                ExperimentMeasurement(
                    channel="Search",
                    test_period=(10, 20),
                    value=500.0,
                    se=100.0,
                )
            ]
        )
        names = _rv_names(model.model)
        assert any("experiment" in n for n in names), sorted(names)


# ---------------------------------------------------------------------------
# MAP smoke + counterfactual mechanics
# ---------------------------------------------------------------------------


class TestMapSmoke:
    @pytest.fixture(scope="class")
    def fitted(self):
        model, sc = _funnel_model()
        with pytest.warns(UserWarning, match="Approximate fits"):
            res = model.fit(method="map", random_seed=0)
        return model, res, sc

    def test_approximate_contract(self, fitted):
        _, res, _ = fitted
        assert res.approximate is True
        assert res.diagnostics.get("rhat_max") is None

    def test_mediation_effects_schema(self, fitted):
        model, _, sc = fitted
        me = model.get_mediation_effects()
        assert set(me["channel"]) == set(sc.channels)
        for col in (
            "direct_effect",
            "total_indirect",
            "total_effect",
            "proportion_mediated",
            "indirect_via_consideration",
        ):
            assert col in me.columns, me.columns
        assert np.all(np.isfinite(me["total_effect"]))

    def test_counterfactual_restores_data(self, fitted):
        model, _, _ = fitted
        model.get_channel_roas()
        assert np.allclose(model.model["X_media"].get_value(), model.X_media)

    def test_pathway_effects(self, fitted):
        model, _, _ = fitted
        pw = model.get_pathway_effects()
        pairs = set(zip(pw["channel"], pw["mediator"]))
        # TV reaches consideration THROUGH awareness (path traced across edge)
        assert ("TV", "consideration") in pairs
        assert ("Display", "consideration") in pairs
        # awareness has affects_outcome=False -> no outcome-terminated path
        assert ("TV", "awareness") not in pairs


# ---------------------------------------------------------------------------
# slow: full NUTS recovery on the brand-funnel truth
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBrandFunnelRecovery:
    _FIT = dict(
        draws=500,
        tune=1000,
        chains=4,
        target_accept=0.95,
        random_seed=11,
        nuts_sampler="numpyro",
    )

    @pytest.fixture(scope="class")
    def fitted(self):
        model, sc = _funnel_model(n_weeks=156)
        res = model.fit(**self._FIT)
        return model, res, sc

    def test_converged(self, fitted):
        _, res, _ = fitted
        rhat = res.diagnostics.get("rhat_max")
        assert rhat is not None and rhat < 1.1, res.diagnostics

    def test_awareness_persistence_recovered(self, fitted):
        _, res, _ = fitted
        rho = float(res.trace.posterior["awareness_persistence"].mean())
        assert rho > 0.6  # truth 0.85

    def test_structural_signs(self, fitted):
        """Sign-constrained params get magnitude floors (bare positivity would
        be prior-satisfiable); the unconstrained price effect (symmetric
        Normal(0, 1) prior) must land clearly negative -- purely data-driven.
        Measured recovery run: b_tv 0.99 (true 0.9), lam 2.0 (2.5),
        phi -0.98 (-0.8), w_dem_cons 0.59 (0.6)."""
        _, res, _ = fitted
        post = res.trace.posterior
        assert float(post["beta_TV_to_awareness"].mean()) > 0.4
        assert float(post["lambda_awareness_to_consideration"].mean()) > 0.8
        assert float(post["phi_Price_to_consideration"].mean()) < -0.3
        assert float(post["w_demand_to_consideration"].mean()) > 0.3
        assert float(post["gamma_consideration"].mean()) > 0.1

    def test_latent_series_recovered(self, fitted):
        model, res, sc = fitted
        post = res.trace.posterior
        p = post["awareness_probability"].mean(("chain", "draw")).values
        assert np.corrcoef(p, sc.notes["p_awareness"])[0, 1] > 0.7
        zc = post["consideration_latent"].mean(("chain", "draw")).values
        assert np.corrcoef(zc, sc.notes["z_consideration"])[0, 1] > 0.7
        f = post["factor_demand"].mean(("chain", "draw")).values
        assert abs(np.corrcoef(f, sc.notes["latent_demand"])[0, 1]) > 0.6

    def test_mediation_is_nonzero_and_dominant(self, fitted):
        """The design-review guard: the counterfactual mediated total must be
        genuinely nonzero for the fully-mediated brand channels (the in-graph
        centering blocker would have made it identically zero)."""
        model, _, sc = fitted
        me = model.get_mediation_effects().set_index("channel")
        for ch in ("TV", "Display"):
            row = me.loc[ch]
            assert row["total_effect"] > 0
            assert row["total_indirect"] > 0.25 * abs(row["total_effect"])
            assert row["proportion_mediated"] > 0.25  # truth 1.0

    def test_roas_ordering_and_scale(self, fitted):
        model, _, sc = fitted
        roas = model.get_channel_roas().set_index("channel")["roas"]
        true = sc.true_roas
        # direction: every channel's contribution is positive in truth
        assert (roas > 0).all()
        # Search (pure direct, strongest identification) within 2x of truth
        assert 0.5 < roas["Search"] / true["Search"] < 2.0
