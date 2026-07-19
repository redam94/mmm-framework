"""Tests for cost-per-conversion power math (planning/cpa.py) and the
ghost-ads CPA wiring: the reciprocal of a Gaussian lift is right-skewed, the
naive delta-method inversion under-covers, the inverted-bound interval is
honest (incl. going unbounded), and the planning summary that survives
inversion is the MAXIMUM detectable CPA."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmm_framework.planning import (
    cpa_interval,
    cpa_power,
    max_detectable_cpa,
    simulate_cpa_distribution,
)
from mmm_framework.planning.methods import (
    GhostAdsDesign,
    ghost_ads_power,
    ghost_ads_users_for_cpa,
)


class TestMaxDetectableCpa:
    def test_basic_and_degenerate(self):
        assert max_detectable_cpa(0.05, 0.002) == pytest.approx(25.0)
        assert math.isnan(max_detectable_cpa(0.05, 0.0))
        assert math.isnan(max_detectable_cpa(0.0, 0.002))

    def test_more_reach_certifies_worse_cpa(self):
        """A bigger test detects a smaller lift -> can certify a HIGHER
        (worse) maximum CPA. The planning curve must be monotone."""
        mdes = [
            ghost_ads_power(GhostAdsDesign(users_reached=n, baseline_rate=0.02))[
                "mde_abs"
            ]
            for n in (100_000, 400_000, 1_600_000)
        ]
        caps = [max_detectable_cpa(0.05, m) for m in mdes]
        assert caps == sorted(caps)  # increasing in reach


class TestCpaInterval:
    def test_bounded_asymmetric(self):
        # strong signal: lift interval strictly positive
        r = cpa_interval(lift=0.004, se_lift=0.0005, cost=0.05)
        assert r["status"] == "bounded"
        assert r["lo"] < r["cpa"] < r["hi"]
        # asymmetric: the expensive arm is longer than the cheap arm
        assert (r["hi"] - r["cpa"]) > (r["cpa"] - r["lo"])
        # endpoints are the images of the lift bounds
        lo_l, hi_l = r["lift_interval"]
        assert r["lo"] == pytest.approx(0.05 / hi_l)
        assert r["hi"] == pytest.approx(0.05 / lo_l)

    def test_upper_unbounded_when_lift_touches_zero(self):
        r = cpa_interval(lift=0.0008, se_lift=0.0005, cost=0.05)
        assert r["status"] == "upper_unbounded"
        assert math.isinf(r["hi"]) and r["lo"] > 0

    def test_undefined_when_no_positive_lift(self):
        r = cpa_interval(lift=-0.002, se_lift=0.0005, cost=0.05)
        assert r["status"] == "undefined"
        assert math.isnan(r["lo"]) and math.isnan(r["hi"])

    def test_naive_interval_can_go_negative(self):
        """The symmetric delta-method interval nonsensically dips below zero
        exactly in the regime where the honest interval goes unbounded."""
        r = cpa_interval(lift=0.0008, se_lift=0.0005, cost=0.05)
        assert r["naive_lo"] < 0  # a negative cost per conversion...
        assert r["status"] == "upper_unbounded"  # honest form: only a lower bound


class TestCpaPower:
    def test_bounded_power_matches_mde_construction(self):
        # at true lift = (z_{.975}+z_{.80})*se the two-sided design is ~80%
        # powered; power-to-get-a-bounded-CPA is the same detection event.
        se = 0.001
        p = cpa_power(0.05, se, true_lift=2.8 * se)
        assert p == pytest.approx(0.80, abs=0.02)

    def test_target_threshold_and_monotonicity(self):
        se = 0.001
        # certifying CPA <= target is harder than merely bounding it
        p_bound = cpa_power(0.05, se, true_lift=0.004)
        p_cert = cpa_power(0.05, se, true_lift=0.004, target_cpa=20.0)
        assert p_cert < p_bound
        # a laxer target is easier to certify
        assert cpa_power(0.05, se, true_lift=0.004, target_cpa=30.0) > p_cert
        # sharper design -> more power for the same target
        assert cpa_power(0.05, se / 2, true_lift=0.004, target_cpa=20.0) > p_cert

    def test_degenerate_inputs(self):
        assert math.isnan(cpa_power(0.05, 0.0, true_lift=0.004))
        assert math.isnan(cpa_power(0.05, 0.001, true_lift=0.004, target_cpa=0.0))


class TestSimulatedSkew:
    """The demonstration facts: the reciprocal is right-skewed, its mean
    overshoots, the naive interval under-covers, the inverted bound holds."""

    @pytest.fixture(scope="class")
    def sim(self):
        # a realistically marginal design: true lift = 1.25x the design MDE
        se = 0.001
        return simulate_cpa_distribution(
            true_lift=3.5 * se, se_lift=se, cost=0.05, n_sims=40_000, seed=7
        )

    def test_right_skew_mean_overshoots_median(self, sim):
        assert sim["skewness"] > 1.0
        assert sim["mean"] > sim["median"]
        assert sim["mean_over_true"] > 1.02  # the tail drags the mean up
        assert sim["p_over_2x_true"] > 0.0

    def test_inverted_bound_covers_naive_undercovers(self, sim):
        assert sim["coverage_inverted"] == pytest.approx(sim["nominal"], abs=0.02)
        assert sim["coverage_naive"] < sim["coverage_inverted"] - 0.02

    def test_strong_signal_regime_is_benign(self):
        """Far from zero the reciprocal is nearly Gaussian and both intervals
        agree — the skew problem is a LOW-SIGNAL problem."""
        se = 0.001
        strong = simulate_cpa_distribution(
            true_lift=12 * se, se_lift=se, cost=0.05, n_sims=40_000, seed=7
        )
        assert abs(strong["skewness"]) < 0.6
        assert strong["coverage_naive"] == pytest.approx(strong["nominal"], abs=0.02)


class TestGhostAdsWiring:
    def test_power_output_carries_max_detectable_cpa(self):
        d = GhostAdsDesign(
            users_reached=400_000, baseline_rate=0.021, cost_per_user=0.05
        )
        p = ghost_ads_power(d)
        assert p["max_detectable_cpa"] == pytest.approx(0.05 / p["mde_abs"])
        assert p["cpa_basis"] == "itt"
        # no cost -> no CPA claim
        p2 = ghost_ads_power(GhostAdsDesign(users_reached=400_000, baseline_rate=0.021))
        assert "max_detectable_cpa" not in p2

    def test_users_for_cpa_round_trips(self):
        d = GhostAdsDesign(
            users_reached=400_000, baseline_rate=0.021, cost_per_user=0.05
        )
        target = 20.0
        n = ghost_ads_users_for_cpa(d, target)
        sized = GhostAdsDesign(
            users_reached=int(n), baseline_rate=0.021, cost_per_user=0.05
        )
        got = ghost_ads_power(sized)["max_detectable_cpa"]
        assert got == pytest.approx(target, rel=0.05)

    def test_users_for_cpa_guards(self):
        with pytest.raises(ValueError, match="cost_per_user"):
            ghost_ads_users_for_cpa(
                GhostAdsDesign(users_reached=1000, baseline_rate=0.02), 20.0
            )
        d = GhostAdsDesign(users_reached=1000, baseline_rate=0.02, cost_per_user=0.05)
        with pytest.raises(ValueError, match="positive"):
            ghost_ads_users_for_cpa(d, 0.0)
