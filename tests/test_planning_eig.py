"""Tests for the EIG side of the priority engine (planning/eig.py): the
Gaussian closed form, the Monte Carlo estimator's agreement with it in the
Gaussian limit, design-grounded experiment precision, and the information
decay / re-test trigger math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmm_framework.planning.eig import (
    DEFAULT_RETEST_THRESHOLD_NATS,
    channel_half_life,
    decayed_sigma,
    eig_gaussian,
    eig_monte_carlo,
    reexperiment_due,
    sigma_exp_for_design,
    use_gaussian,
)


class TestGaussianClosedForm:
    def test_known_values(self):
        # equal prior and experiment sd: 0.5 * ln(2)
        assert eig_gaussian(1.0, 1.0) == pytest.approx(0.5 * math.log(2))
        # experiment 2x as precise: 0.5 * ln(1 + 4)
        assert eig_gaussian(1.0, 0.5) == pytest.approx(0.5 * math.log(5))

    def test_monotonic_in_prior_uncertainty_and_precision(self):
        assert eig_gaussian(2.0, 0.5) > eig_gaussian(1.0, 0.5)
        assert eig_gaussian(1.0, 0.25) > eig_gaussian(1.0, 0.5)

    def test_degenerate_inputs(self):
        assert eig_gaussian(0.0, 0.5) == 0.0
        with pytest.raises(ValueError):
            eig_gaussian(1.0, 0.0)


class TestMonteCarlo:
    def test_matches_closed_form_on_gaussian_draws(self):
        rng = np.random.default_rng(7)
        draws = rng.normal(2.0, 1.0, size=4000)
        sigma_exp = 0.5
        closed = eig_gaussian(float(draws.std()), sigma_exp)
        mc = eig_monte_carlo(draws, sigma_exp, n_outcomes=256, rng=rng)
        assert mc == pytest.approx(closed, abs=0.1)

    def test_nonnegative_and_finite_on_skewed_draws(self):
        rng = np.random.default_rng(7)
        draws = rng.lognormal(0.0, 1.0, size=2000)
        mc = eig_monte_carlo(draws, 0.5, n_outcomes=128, rng=rng)
        assert np.isfinite(mc) and mc >= 0.0

    def test_degenerate_draws(self):
        assert eig_monte_carlo(np.full(100, 2.0), 0.5) == 0.0

    def test_normality_gate(self):
        rng = np.random.default_rng(0)
        assert use_gaussian(rng.normal(size=2000))
        assert not use_gaussian(rng.lognormal(0, 1.0, size=2000))


class TestSigmaExp:
    def test_design_grounded_not_posterior_scaled(self):
        # 10% relative precision for a geo holdout, on the ROI scale
        assert sigma_exp_for_design("geo_holdout", 2.0) == pytest.approx(0.2)
        assert sigma_exp_for_design("national_pulse", 2.0) == pytest.approx(0.5)
        # fuzzy design names map to a known class
        assert sigma_exp_for_design("geo holdout / geo lift test", 1.0) == (
            pytest.approx(0.10)
        )
        assert sigma_exp_for_design("something weird", 1.0) == pytest.approx(0.25)
        # floor keeps sigma_exp positive for near-zero ROI channels
        assert sigma_exp_for_design("geo_holdout", 0.0) == pytest.approx(0.005)


class TestInformationDecay:
    def test_variance_doubles_after_one_half_life(self):
        s = decayed_sigma(0.4, weeks_elapsed=26.0, half_life_weeks=26.0)
        assert s == pytest.approx(0.4 * math.sqrt(2.0))
        assert decayed_sigma(0.4, 0.0, 26.0) == pytest.approx(0.4)

    def test_monotone_in_time(self):
        sds = [decayed_sigma(0.3, w, 39.0) for w in (0, 10, 30, 80)]
        assert all(a < b for a, b in zip(sds, sds[1:]))

    def test_channel_half_life_classes_and_overrides(self):
        assert channel_half_life("Paid_Search_Brand") == 26.0
        assert channel_half_life("Linear_TV") == 52.0
        assert channel_half_life("Mystery_Channel") == 39.0
        assert channel_half_life("Linear_TV", {"Linear_TV": 10.0}) == 10.0

    def test_retest_trigger_crosses_threshold_over_time(self):
        # freshly calibrated: tight posterior, EIG below threshold -> not due
        due_now, eig_now = reexperiment_due(0.05, 0.0, 26.0, sigma_exp=0.2)
        assert not due_now and eig_now < DEFAULT_RETEST_THRESHOLD_NATS
        # two years later the decayed uncertainty makes a fresh test worthwhile
        due_later, eig_later = reexperiment_due(0.05, 104.0, 26.0, sigma_exp=0.2)
        assert due_later and eig_later > eig_now

    def test_freshness_floor_overrides_wide_posterior(self):
        """A channel calibrated weeks ago whose posterior is STILL wide clears
        the EIG threshold immediately — but it isn't 'stale'; the freshness
        floor holds the re-test flag until the evidence has actually aged."""
        # wide posterior, fresh evidence: EIG huge, but not due
        due, eig = reexperiment_due(0.8, 4.0, 26.0, sigma_exp=0.2)
        assert eig > DEFAULT_RETEST_THRESHOLD_NATS and not due
        # same posterior past the floor: due
        due_old, _ = reexperiment_due(0.8, 14.0, 26.0, sigma_exp=0.2)
        assert due_old
        # the floor is tunable
        due_tight, _ = reexperiment_due(0.8, 4.0, 26.0, sigma_exp=0.2, min_age_weeks=2)
        assert due_tight
