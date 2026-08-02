"""Tests for experiment-readout bias priors and their propagation into calibration.

Three things are pinned here, in rising order of how expensive it would be to get
them wrong:

1. **The threat model follows assignment, not the estimator.** A randomized geo
   split must never be handed a confounding tipping point.
2. **The placebo spread is not double-counted.** ``geo_lift_design`` already
   inflates its own standard error when it has enough placebo windows; adding
   that same spread again as a bias would widen every calibrated interval by a
   spurious factor.
3. **The default calibration graph is byte-identical.** The bias fields exist on
   ``ExperimentMeasurement``, so an existing measurement must produce exactly the
   likelihood it produced before.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from mmm_framework.calibration.likelihood import (
    ExperimentEstimand,
    ExperimentMeasurement,
)
from mmm_framework.planning.experiment_sensitivity import (
    METHOD_THREATS,
    UNKNOWN_METHOD_THREAT,
    derive_bias_prior,
    experiment_sensitivity,
    threat_for,
)


def _geo_design(
    *,
    randomized: bool = False,
    se_source: str = "placebo_calibrated",
    calibration_factor: float = 1.5,
    se_roas: float = 0.45,
    parallel_trends_warning: bool = False,
    n_windows: int = 40,
) -> dict:
    """A design payload shaped like ``geo_lift_design``'s return value."""
    return {
        "design_key": "geo_lift",
        "randomized": randomized,
        "se_source": se_source,
        "se_roas": se_roas,
        "placebo": {"n_windows": n_windows, "sd": 120.0, "p95_abs": 210.0},
        "diagnostics": {
            "calibration_factor": calibration_factor,
            "parallel_trends_warning": parallel_trends_warning,
        },
        "weekly_spend_delta": 50_000.0,
    }


# =========================================================================== #
# the threat model
# =========================================================================== #


class TestThreatModel:
    def test_randomization_repoints_the_threat_away_from_confounding(self):
        """The distinction the whole module turns on."""
        observational = threat_for("did_mmt", _geo_design(randomized=False))
        randomized = threat_for("did_mmt", _geo_design(randomized=True))
        assert observational.threat == "unmeasured_confounding"
        assert randomized.threat == "interference"
        assert randomized.floor < observational.floor

    def test_a_randomized_caveat_says_confounding_is_not_the_threat(self):
        t = threat_for("did_mmt", _geo_design(randomized=True))
        assert "NOT the threat" in t.caveat
        assert "interference" in t.caveat.lower()

    def test_ghost_ads_is_about_external_validity_not_confounding(self):
        t = threat_for("ghost_ads")
        assert t.threat == "external_validity"
        assert "no unmeasured confounder" in t.caveat

    def test_a_failing_parallel_trends_screen_widens_the_floor(self):
        clean = threat_for("did_mmt", _geo_design())
        warned = threat_for("did_mmt", _geo_design(parallel_trends_warning=True))
        assert warned.floor > clean.floor
        assert "parallel-trends screen is failing" in warned.caveat

    def test_a_carryover_warning_widens_a_switchback(self):
        d = {"design_key": "national_flighting", "carryover_warning": True}
        assert threat_for("switchback", d).floor > METHOD_THREATS["switchback"].floor

    def test_an_unknown_method_gets_the_widest_default_and_says_so(self):
        t = threat_for(None)
        assert t is UNKNOWN_METHOD_THREAT
        assert t.floor >= max(m.floor for m in METHOD_THREATS.values())
        assert "was not identified" in t.caveat

    def test_design_key_resolves_when_the_method_is_unnamed(self):
        assert threat_for(None, {"design_key": "geo_lift"}).method_key == "did_mmt"
        assert (
            threat_for(None, {"design_key": "national_flighting"}).method_key
            == "national_flighting"
        )

    def test_every_registered_threat_is_self_consistent(self):
        for key, t in METHOD_THREATS.items():
            assert t.method_key == key
            assert 0.0 < t.floor < 1.0
            assert t.threat in (
                "unmeasured_confounding",
                "interference",
                "external_validity",
            )
            assert len(t.caveat) > 40
            json.dumps(t.to_dict())


# =========================================================================== #
# deriving the width
# =========================================================================== #


class TestDeriveBiasPrior:
    def test_excess_is_the_placebo_spread_beyond_the_analytic_model(self):
        design = _geo_design(calibration_factor=2.0, se_roas=0.40)
        # se_analytic = 0.40 / 2 = 0.20; excess = 0.20 * sqrt(4 - 1)
        d = derive_bias_prior(
            value=2.0, se=0.15, estimand="roas", design=design, method_key="did_mmt"
        )
        assert d.excess == pytest.approx(0.20 * math.sqrt(3.0))
        assert d.is_measured

    def test_an_already_calibrated_se_is_not_double_counted(self):
        """The trap: the design's SE already contains the placebo spread."""
        design = _geo_design(calibration_factor=2.0, se_roas=0.40)
        quoted_design_se = derive_bias_prior(
            value=2.0, se=0.40, estimand="roas", design=design, method_key="did_mmt"
        )
        own_estimator_se = derive_bias_prior(
            value=2.0, se=0.15, estimand="roas", design=design, method_key="did_mmt"
        )
        assert quoted_design_se.absorbed
        assert not own_estimator_se.absorbed
        assert quoted_design_se.prior.sigma < own_estimator_se.prior.sigma
        assert any("already inside it" in n for n in quoted_design_se.notes)

    def test_absorbed_still_applies_the_floor(self):
        """Absorbing the sampling inflation does not absorb spillover."""
        design = _geo_design(calibration_factor=2.0, se_roas=0.40)
        d = derive_bias_prior(
            value=2.0, se=0.40, estimand="roas", design=design, method_key="did_mmt"
        )
        assert d.prior.sigma == pytest.approx(d.floor_component)
        assert d.prior.sigma > 0

    def test_no_placebo_windows_falls_back_to_the_floor_and_says_so(self):
        design = _geo_design(se_source="analytic", calibration_factor=1.0, n_windows=3)
        d = derive_bias_prior(
            value=2.0, se=0.3, estimand="roas", design=design, method_key="did_mmt"
        )
        assert d.excess == 0.0
        assert not d.is_measured
        assert any("at least 12 are needed" in n for n in d.notes)

    def test_width_combines_excess_and_floor_in_quadrature(self):
        design = _geo_design(calibration_factor=2.0, se_roas=0.40)
        d = derive_bias_prior(
            value=2.0, se=0.15, estimand="roas", design=design, method_key="did_mmt"
        )
        assert d.prior.sigma == pytest.approx(math.hypot(d.excess, d.floor_component))

    def test_floor_scales_with_the_measured_value(self):
        design = _geo_design(se_source="analytic", calibration_factor=1.0)
        small = derive_bias_prior(
            value=1.0, se=0.2, design=design, method_key="did_mmt"
        )
        big = derive_bias_prior(value=10.0, se=0.2, design=design, method_key="did_mmt")
        assert big.prior.sigma == pytest.approx(10.0 * small.prior.sigma)

    def test_a_non_roas_estimand_refuses_the_design_evidence_and_explains(self):
        design = _geo_design(calibration_factor=2.0)
        d = derive_bias_prior(
            value=5000.0,
            se=800.0,
            estimand="contribution",
            design=design,
            method_key="did_mmt",
        )
        assert d.excess == 0.0
        assert any("only defined on the ROAS scale" in n for n in d.notes)
        assert d.prior.sigma > 0  # the floor still applies

    def test_explicit_sigma_overrides_everything(self):
        d = derive_bias_prior(
            value=2.0,
            se=0.2,
            design=_geo_design(),
            method_key="did_mmt",
            explicit_sigma=0.9,
            explicit_mu=-0.1,
        )
        assert d.prior.sigma == pytest.approx(0.9)
        assert d.prior.mu == pytest.approx(-0.1)
        assert d.prior.source == "explicit"

    def test_the_default_direction_is_unsigned(self):
        """A placebo spread is a width, not a signed offset.

        Inferring a direction would mean dividing by ``weekly_spend_delta``, which
        the design stores unsigned — so it inverts on a go-dark holdout.
        """
        d = derive_bias_prior(
            value=2.0, se=0.2, design=_geo_design(), method_key="did_mmt"
        )
        assert d.prior.mu == 0.0

    def test_source_records_where_each_component_came_from(self):
        design = _geo_design(calibration_factor=2.0, se_roas=0.40)
        measured = derive_bias_prior(
            value=2.0, se=0.15, estimand="roas", design=design, method_key="did_mmt"
        )
        assert "placebo:excess-over-analytic" in measured.prior.source
        assert "floor:did_mmt@" in measured.prior.source
        assert measured.prior.is_measured

        guessed = derive_bias_prior(value=2.0, se=0.15, method_key="did_mmt")
        assert guessed.prior.source == "named"
        assert not guessed.prior.is_measured

    def test_to_dict_is_json_safe(self):
        d = derive_bias_prior(
            value=2.0, se=0.15, design=_geo_design(), method_key="did_mmt"
        )
        json.dumps(d.to_dict())


# =========================================================================== #
# the report
# =========================================================================== #


class TestExperimentSensitivityReport:
    def test_a_comfortable_readout_survives_its_own_residual_bias(self):
        r = experiment_sensitivity(
            value=3.5,
            se=0.25,
            estimand="roas",
            channel="TV",
            design=_geo_design(),
            method_key="did_mmt",
            include_surface=False,
        )
        assert r.verdict in ("resilient", "fragile")
        assert r.sensitivity.prob_at_zero_bias > 0.99

    def test_a_marginal_readout_is_overturned_before_any_bias(self):
        r = experiment_sensitivity(
            value=1.05,
            se=0.5,
            estimand="roas",
            channel="Search",
            design=_geo_design(),
            method_key="did_mmt",
            include_surface=False,
        )
        assert r.verdict == "overturned"

    def test_the_break_even_reference_follows_the_estimand(self):
        roas = experiment_sensitivity(
            value=2.0, se=0.3, estimand="roas", include_surface=False
        )
        contribution = experiment_sensitivity(
            value=5000.0, se=800.0, estimand="contribution", include_surface=False
        )
        assert roas.reference == 1.0
        assert contribution.reference == 0.0

    def test_a_randomized_design_is_told_what_it_is_pricing(self):
        r = experiment_sensitivity(
            value=2.5,
            se=0.3,
            design=_geo_design(randomized=True),
            method_key="did_mmt",
            include_surface=False,
        )
        assert r.threat.threat == "interference"
        assert "not confounding" in r.caveat
        assert "do not read it as a confounding sensitivity" in r.caveat

    def test_an_unmeasured_width_is_disclosed_in_the_caveat(self):
        r = experiment_sensitivity(
            value=2.5, se=0.3, method_key="did_mmt", include_surface=False
        )
        assert "entirely the method's default floor" in r.caveat

    def test_a_measured_width_does_not_claim_to_be_a_floor(self):
        r = experiment_sensitivity(
            value=2.5,
            se=0.15,
            design=_geo_design(calibration_factor=2.0),
            method_key="did_mmt",
            include_surface=False,
        )
        assert "entirely the method's default floor" not in r.caveat

    def test_wider_assumed_bias_lowers_the_decision_probability(self):
        narrow = experiment_sensitivity(
            value=2.0, se=0.3, explicit_sigma=0.1, include_surface=False
        )
        wide = experiment_sensitivity(
            value=2.0, se=0.3, explicit_sigma=1.5, include_surface=False
        )
        assert (
            wide.sensitivity.scenarios[0].prob_above
            < narrow.sensitivity.scenarios[0].prob_above
        )

    def test_to_dict_is_json_safe(self):
        r = experiment_sensitivity(
            value=2.0, se=0.3, design=_geo_design(), method_key="did_mmt"
        )
        json.dumps(r.to_dict())


# =========================================================================== #
# propagation into the calibration likelihood
# =========================================================================== #


class TestCalibrationPropagation:
    def test_measurement_kwargs_round_trip_onto_a_measurement(self):
        r = experiment_sensitivity(
            value=2.0,
            se=0.3,
            estimand="roas",
            channel="TV",
            design=_geo_design(calibration_factor=2.0),
            method_key="did_mmt",
            include_surface=False,
        )
        m = ExperimentMeasurement(
            channel="TV",
            test_period=(0, 10),
            value=2.0,
            se=0.3,
            estimand=ExperimentEstimand.ROAS,
            **r.as_measurement_kwargs(),
        )
        assert m.has_bias
        assert m.bias_sigma == pytest.approx(r.derived.prior.sigma)
        assert m.bias_source and "floor:did_mmt@" in m.bias_source

    def test_defaults_leave_the_graph_byte_identical(self):
        """The invariant every existing calibrated fit depends on."""
        import pymc as pm
        import pytensor.tensor as pt

        from mmm_framework.calibration.likelihood import attach_experiment_likelihood

        plain = ExperimentMeasurement(
            channel="TV", test_period=(0, 10), value=2.0, se=0.4
        )
        assert not plain.has_bias

        with pm.Model() as model:
            theta = pm.Normal("theta", 0.0, 1.0)
            attach_experiment_likelihood("exp", pt.as_tensor_variable(theta), plain)
        obs = model["exp"]

        # The observed value and sigma must be the RAW floats, bit for bit — not
        # `value - 0.0` and `hypot(se, 0.0)` recomputations, neither of which is
        # guaranteed to round-trip identically.
        sigma_val = float(obs.owner.inputs[-1].eval())
        observed = float(np.asarray(model.rvs_to_values[obs].eval()))
        assert sigma_val == 0.4
        assert observed == 2.0
        # And the log-probability must match a hand-built reference graph.
        with pm.Model() as reference:
            theta_r = pm.Normal("theta", 0.0, 1.0)
            pm.Deterministic("exp_model_estimand", pt.as_tensor_variable(theta_r))
            pm.Normal("exp", mu=theta_r, sigma=0.4, observed=2.0)
        point = {"theta": 0.37}
        assert model.compile_logp()(point) == reference.compile_logp()(point)

    def test_a_bias_widens_the_likelihood_in_quadrature(self):
        import pymc as pm
        import pytensor.tensor as pt

        from mmm_framework.calibration.likelihood import attach_experiment_likelihood

        biased = ExperimentMeasurement(
            channel="TV",
            test_period=(0, 10),
            value=2.0,
            se=0.4,
            bias_mu=0.2,
            bias_sigma=0.3,
        )
        with pm.Model() as model:
            theta = pm.Normal("theta", 0.0, 1.0)
            attach_experiment_likelihood("exp", pt.as_tensor_variable(theta), biased)
        obs = model["exp"]
        sigma_val = float(obs.owner.inputs[-1].eval())
        assert sigma_val == pytest.approx(math.hypot(0.4, 0.3))
        observed = np.asarray(model.rvs_to_values[obs].eval())
        assert float(observed) == pytest.approx(2.0 - 0.2)

    def test_a_biased_measurement_pulls_the_posterior_less(self):
        """The point of the whole propagation path."""
        import pymc as pm
        import pytensor.tensor as pt

        from mmm_framework.calibration.likelihood import attach_experiment_likelihood

        def _posterior_sd(measurement):
            with pm.Model():
                theta = pm.Normal("theta", 0.0, 5.0)
                attach_experiment_likelihood(
                    "exp", pt.as_tensor_variable(theta), measurement
                )
                idata = pm.sample(
                    draws=800,
                    tune=500,
                    chains=2,
                    random_seed=3,
                    progressbar=False,
                    compute_convergence_checks=False,
                )
            return float(idata.posterior["theta"].values.std())

        raw = ExperimentMeasurement(
            channel="TV", test_period=(0, 10), value=2.0, se=0.4
        )
        biased = ExperimentMeasurement(
            channel="TV", test_period=(0, 10), value=2.0, se=0.4, bias_sigma=0.6
        )
        assert _posterior_sd(biased) > _posterior_sd(raw) * 1.2

    def test_lognormal_requires_a_log_scale_bias(self):
        with pytest.raises(ValueError, match="bias_scale='log'"):
            ExperimentMeasurement(
                channel="TV",
                test_period=(0, 10),
                value=2.0,
                se=0.4,
                distribution="lognormal",
                bias_mu=0.1,
            )

    def test_lognormal_bias_stays_on_the_log_scale(self):
        import pymc as pm
        import pytensor.tensor as pt

        from mmm_framework.calibration.likelihood import (
            attach_experiment_likelihood,
            lognormal_sigma_from_moments,
        )

        m = ExperimentMeasurement(
            channel="TV",
            test_period=(0, 10),
            value=2.0,
            se=0.4,
            distribution="lognormal",
            bias_scale="log",
            bias_mu=0.1,
            bias_sigma=0.25,
        )
        with pm.Model() as model:
            theta = pm.Normal("theta", 1.0, 1.0)
            attach_experiment_likelihood("exp", pt.exp(theta), m)
        obs = model["exp"]
        expected_sigma = math.hypot(lognormal_sigma_from_moments(2.0, 0.4), 0.25)
        assert float(obs.owner.inputs[-1].eval()) == pytest.approx(expected_sigma)
        observed = float(np.asarray(model.rvs_to_values[obs].eval()))
        assert observed == pytest.approx(math.log(2.0) - 0.1)

    def test_a_legacy_spec_entry_round_trips_unbiased(self):
        payload = {
            "channel": "TV",
            "test_period": [0, 10],
            "value": 2.0,
            "se": 0.4,
            "estimand": "roas",
        }
        m = ExperimentMeasurement.from_dict(payload)
        assert not m.has_bias
        assert m.bias_scale == "natural" and m.bias_source is None
