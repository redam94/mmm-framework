"""Tests for the bias-parameter sensitivity engine.

All fast and analytic — the engine is closed-form, so every claim it makes can be
checked against either an exact formula or a brute-force Monte Carlo, with no
model fitting anywhere.

The load-bearing tests are the ones that pin *design decisions* rather than
arithmetic:

* :class:`TestFlatTauLimit` makes the "the draws path is the flat-prior limit of
  the conjugate model" claim executable rather than asserted in a docstring.
* :class:`TestReferenceZeroIsNotDataFree` is a regression gate. A per-draw
  multiplicative bias would make ``P(tau > 0)`` equal ``P(b < 1)`` exactly for any
  positive-support posterior — the same number for every channel, with no data in
  it. These tests fail if anyone reintroduces that parameterization.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import stats

from mmm_framework.diagnostics.bias_sensitivity import (
    DEFAULT_DECISION_THRESHOLD,
    NAMED_BIAS_PRIORS,
    VERDICTS,
    BiasPrior,
    BiasSensitivity,
    bias_adjusted_moments,
    bias_sensitivity_report,
    evalue,
    mixture_interval,
    mixture_moments,
    named_prior_ladder,
    prior_dominance_caveat,
    prob_above,
    sensitivity_surface,
    tipping_point,
    tipping_point_mu,
)

RNG = np.random.default_rng(20260802)


def _roi_draws(mean: float, sd: float, n: int = 4000) -> np.ndarray:
    """Deterministic draws standing in for a channel's ROI posterior."""
    return np.asarray(stats.norm(mean, sd).ppf(np.linspace(0.0005, 0.9995, n)))


# --------------------------------------------------------------------------- #
# the prior
# --------------------------------------------------------------------------- #


class TestBiasPrior:
    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError, match="non-negative"):
            BiasPrior(mu=0.0, sigma=-0.1)

    def test_rejects_non_finite_mu(self):
        with pytest.raises(ValueError, match="finite"):
            BiasPrior(mu=float("inf"), sigma=0.1)

    def test_rejects_a_per_draw_multiplicative_scale(self):
        # The name someone would reach for; the module must refuse it and say why.
        with pytest.raises(ValueError, match="multiplicative"):
            BiasPrior(mu=0.0, sigma=0.3, scale="relative")  # type: ignore[arg-type]

    def test_to_absolute_scales_both_moments_by_magnitude(self):
        p = BiasPrior(mu=0.2, sigma=0.3, scale="fraction_of_mean")
        a = p.to_absolute(4.0)
        assert a.scale == "absolute"
        assert a.mu == pytest.approx(0.8)
        assert a.sigma == pytest.approx(1.2)

    def test_to_absolute_uses_magnitude_so_sign_does_not_flip_direction(self):
        p = BiasPrior(mu=0.25, sigma=0.5, scale="fraction_of_mean")
        assert p.to_absolute(-4.0).mu == pytest.approx(p.to_absolute(4.0).mu)

    def test_absolute_prior_is_its_own_absolute(self):
        p = BiasPrior(mu=1.0, sigma=2.0, scale="absolute")
        assert p.to_absolute(99.0) is p

    def test_source_distinguishes_a_guess_from_a_measurement(self):
        assert not BiasPrior(sigma=0.3, source="named").is_measured
        assert BiasPrior(sigma=0.3, source="placebo").is_measured
        assert BiasPrior(sigma=0.3, source="benchmark:Price").is_measured

    def test_ladder_matches_the_documented_rungs(self):
        ladder = named_prior_ladder()
        assert [p.label for p in ladder] == list(NAMED_BIAS_PRIORS)
        assert [p.sigma for p in ladder] == list(NAMED_BIAS_PRIORS.values())
        assert all(p.mu == 0.0 and not p.is_measured for p in ladder)


# --------------------------------------------------------------------------- #
# the conjugate solve and its flat-tau limit
# --------------------------------------------------------------------------- #


class TestFlatTauLimit:
    """The claim that makes the draws path legitimate, made executable."""

    def test_default_is_the_closed_form_limit(self):
        prior = BiasPrior(mu=0.4, sigma=0.3, scale="absolute")
        mean, sd = bias_adjusted_moments(1.5, 0.2, prior)
        assert mean == pytest.approx(1.5 - 0.4)
        assert sd == pytest.approx(math.hypot(0.2, 0.3))

    def test_wide_tau_prior_converges_to_the_limit(self):
        prior = BiasPrior(mu=0.4, sigma=0.3, scale="absolute")
        limit = bias_adjusted_moments(1.5, 0.2, prior)
        wide = bias_adjusted_moments(1.5, 0.2, prior, tau_prior_sd=1e6)
        assert wide[0] == pytest.approx(limit[0], rel=1e-8)
        assert wide[1] == pytest.approx(limit[1], rel=1e-8)

    def test_finite_tau_prior_shrinks_toward_it(self):
        prior = BiasPrior(mu=0.0, sigma=0.3, scale="absolute")
        mean, sd = bias_adjusted_moments(
            2.0, 0.2, prior, tau_prior_sd=0.5, tau_prior_mean=0.0
        )
        # A finite prior centred at zero pulls the estimate down and tightens it.
        assert 0.0 < mean < 2.0
        assert sd < math.hypot(0.2, 0.3)

    def test_matches_an_independent_two_parameter_solve(self):
        # Explicit bivariate-normal conjugate update, written differently.
        d_hat, se, mu_b, s_b, s_tau = 1.2, 0.25, 0.1, 0.4, 0.9
        v = 1.0 / (1.0 / s_tau**2 + 1.0 / (se**2 + s_b**2))
        m = v * (d_hat - mu_b) / (se**2 + s_b**2)
        prior = BiasPrior(mu=mu_b, sigma=s_b, scale="absolute")
        mean, sd = bias_adjusted_moments(d_hat, se, prior, tau_prior_sd=s_tau)
        assert mean == pytest.approx(m, rel=1e-10)
        assert sd == pytest.approx(math.sqrt(v), rel=1e-10)

    def test_point_mass_bias_prior_does_not_divide_by_zero(self):
        prior = BiasPrior(mu=0.3, sigma=0.0, scale="absolute")
        mean, sd = bias_adjusted_moments(1.0, 0.2, prior, tau_prior_sd=10.0)
        assert math.isfinite(mean) and math.isfinite(sd)

    def test_rejects_non_positive_se(self):
        with pytest.raises(ValueError, match="se must be positive"):
            bias_adjusted_moments(1.0, 0.0, BiasPrior(sigma=0.1, scale="absolute"))


# --------------------------------------------------------------------------- #
# the mixture arithmetic
# --------------------------------------------------------------------------- #


class TestMixtureArithmetic:
    def test_prob_above_matches_brute_force_monte_carlo(self):
        draws = _roi_draws(2.0, 0.6, 3000)
        prior = BiasPrior(mu=0.2, sigma=0.5, scale="absolute")
        exact = prob_above(draws, prior, reference=1.0)

        rng = np.random.default_rng(7)
        idx = rng.integers(0, draws.size, 400_000)
        sim = draws[idx] - rng.normal(0.2, 0.5, 400_000)
        assert exact == pytest.approx(float((sim > 1.0).mean()), abs=0.003)

    def test_zero_sigma_returns_the_unmodified_probability(self):
        draws = _roi_draws(2.0, 0.6)
        zero = BiasPrior(mu=0.0, sigma=0.0, scale="absolute")
        assert prob_above(draws, zero, reference=1.0) == pytest.approx(
            float((draws > 1.0).mean())
        )

    def test_widening_the_prior_moves_probability_toward_one_half(self):
        draws = _roi_draws(2.0, 0.4)
        probs = [
            prob_above(
                draws, BiasPrior(mu=0.0, sigma=s, scale="absolute"), reference=1.0
            )
            for s in (0.0, 1.0, 5.0, 50.0)
        ]
        assert probs == sorted(probs, reverse=True)
        assert probs[-1] == pytest.approx(0.5, abs=0.02)

    def test_moments_match_the_law_of_total_variance(self):
        draws = _roi_draws(2.0, 0.6)
        prior = BiasPrior(mu=0.1, sigma=0.4, scale="absolute")
        from mmm_framework.diagnostics.bias_sensitivity import _components

        m, s = _components(draws, prior, magnitude=abs(draws.mean()))
        mean, sd = mixture_moments(m, s)
        assert mean == pytest.approx(draws.mean() - 0.1)
        assert sd == pytest.approx(math.hypot(draws.std(), 0.4), rel=1e-6)

    def test_interval_brackets_the_mean_and_widens_with_the_prior(self):
        draws = _roi_draws(2.0, 0.5)
        from mmm_framework.diagnostics.bias_sensitivity import _components

        widths = []
        for sigma in (0.0, 0.5, 1.5):
            m, s = _components(
                draws,
                BiasPrior(mu=0.0, sigma=sigma, scale="absolute"),
                magnitude=abs(draws.mean()),
            )
            lo, hi = mixture_interval(m, s, 0.90)
            assert lo < draws.mean() < hi
            widths.append(hi - lo)
        assert widths == sorted(widths)

    def test_interval_at_zero_bias_matches_the_raw_percentiles(self):
        draws = _roi_draws(2.0, 0.5)
        from mmm_framework.diagnostics.bias_sensitivity import _components

        m, s = _components(
            draws,
            BiasPrior(mu=0.0, sigma=0.0, scale="absolute"),
            magnitude=abs(draws.mean()),
        )
        lo, hi = mixture_interval(m, s, 0.90)
        assert lo == pytest.approx(float(np.percentile(draws, 5)), abs=0.02)
        assert hi == pytest.approx(float(np.percentile(draws, 95)), abs=0.02)


# --------------------------------------------------------------------------- #
# the regression gate: reference 0 must still read the data
# --------------------------------------------------------------------------- #


class TestReferenceZeroIsNotDataFree:
    """An efficiency channel's break-even reference is 0, and every media
    coefficient here has positive support — so a per-draw multiplicative bias
    would give ``P(tau > 0) = P(b < 1)`` for *every* channel alike, a number with
    no data in it. These gates fail if that parameterization comes back.
    """

    def test_two_different_posteriors_give_different_answers(self):
        prior = BiasPrior(mu=0.0, sigma=0.4, scale="fraction_of_mean")
        # Same mean, very different precision. A data-free rule cannot tell them
        # apart; the shipped one must.
        tight = prob_above(_roi_draws(1.0, 0.05), prior, reference=0.0)
        loose = prob_above(_roi_draws(1.0, 0.80), prior, reference=0.0)
        assert tight > loose + 0.02

    def test_a_stronger_channel_is_harder_to_overturn(self):
        prior = BiasPrior(mu=0.0, sigma=0.4, scale="fraction_of_mean")
        weak = prob_above(_roi_draws(1.0, 0.5), prior, reference=0.0)
        strong = prob_above(_roi_draws(4.0, 0.5), prior, reference=0.0)
        assert strong > weak

    def test_all_positive_draws_do_not_force_a_constant_probability(self):
        # Strictly positive support (as Gamma / LogNormal priors guarantee), and
        # genuinely different SHAPES rather than a rescaling of one another.
        a = np.exp(_roi_draws(0.0, 0.15))  # tight relative to its mean
        b = np.exp(_roi_draws(0.0, 0.90))  # diffuse relative to its mean
        assert (a > 0).all() and (b > 0).all()
        prior = BiasPrior(mu=0.0, sigma=0.5, scale="fraction_of_mean")
        p_a = prob_above(a, prior, reference=0.0)
        p_b = prob_above(b, prior, reference=0.0)
        assert p_a > p_b + 0.02

    def test_at_reference_zero_the_answer_is_scale_invariant_by_design(self):
        """Two posteriors differing only by a unit change must agree.

        This is the property that makes ``fraction_of_mean`` reportable across a
        portfolio: "a bias worth 50% of the estimate" means the same thing for a
        channel measured in dollars and one measured in thousands. It is a
        *different* statement from the broken per-draw multiplicative rule, which
        collapses to ``P(b < 1)`` for every channel regardless of the posterior's
        shape — pinned by the two neighbouring tests.
        """
        base = np.exp(_roi_draws(0.0, 0.3))
        prior = BiasPrior(mu=0.0, sigma=0.5, scale="fraction_of_mean")
        assert prob_above(base, prior, reference=0.0) == pytest.approx(
            prob_above(1000.0 * base, prior, reference=0.0), abs=1e-12
        )

    def test_sign_of_a_draw_never_flips_the_bias_direction(self):
        # Draws straddling zero. A positive mu must shift EVERY draw down.
        draws = np.array([-1.0, -0.2, 0.5, 2.0])
        from mmm_framework.diagnostics.bias_sensitivity import _components

        prior = BiasPrior(mu=0.5, sigma=0.0, scale="fraction_of_mean")
        m, _ = _components(draws, prior, magnitude=abs(draws.mean()))
        assert np.all(m < draws)


# --------------------------------------------------------------------------- #
# tipping points
# --------------------------------------------------------------------------- #


class TestTippingPoint:
    def test_the_returned_sigma_actually_sits_on_the_threshold(self):
        draws = _roi_draws(2.0, 0.35)
        tp = tipping_point(draws, reference=1.0, scale="absolute")
        assert tp.crossed and tp.value is not None
        at = prob_above(
            draws,
            BiasPrior(mu=0.0, sigma=tp.value, scale="absolute"),
            reference=1.0,
        )
        assert at == pytest.approx(DEFAULT_DECISION_THRESHOLD, abs=1e-6)

    def test_the_returned_mu_actually_sits_on_the_threshold(self):
        draws = _roi_draws(2.0, 0.35)
        tp = tipping_point_mu(draws, reference=1.0, scale="fraction_of_mean")
        assert tp.crossed and tp.value is not None
        at = prob_above(
            draws,
            BiasPrior(mu=tp.value, sigma=0.0, scale="fraction_of_mean"),
            reference=1.0,
        )
        assert at == pytest.approx(DEFAULT_DECISION_THRESHOLD, abs=1e-6)

    def test_already_below_is_distinguished_from_never_crossing(self):
        # Not supported even at zero bias.
        weak = tipping_point(_roi_draws(1.05, 0.5), reference=1.0, scale="absolute")
        assert weak.already_below and weak.crossed
        # Supported across the whole scanned range.
        strong = tipping_point_mu(
            _roi_draws(20.0, 0.2), reference=1.0, scale="fraction_of_mean", max_mu=0.1
        )
        assert not strong.crossed and strong.value is None
        assert not strong.already_below
        assert strong.max_scanned == pytest.approx(0.1)

    def test_describe_never_claims_robustness_when_nothing_crossed(self):
        tp = tipping_point_mu(
            _roi_draws(20.0, 0.2), reference=1.0, scale="fraction_of_mean", max_mu=0.1
        )
        text = tp.describe()
        assert "widest bias scanned" in text and "10%" in text
        assert "robust" not in text.lower()

    def test_a_stronger_channel_has_a_later_tipping_point(self):
        weak = tipping_point_mu(_roi_draws(1.5, 0.3), reference=1.0)
        strong = tipping_point_mu(_roi_draws(4.0, 0.3), reference=1.0)
        assert weak.value is not None and strong.value is not None
        assert strong.value > weak.value

    def test_empty_draws_are_not_assessable_rather_than_zero(self):
        tp = tipping_point(np.array([]), reference=1.0)
        assert tp.value is None and not tp.crossed and not tp.already_below

    def test_non_finite_draws_are_dropped(self):
        clean = _roi_draws(2.0, 0.4, 500)
        dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
        assert tipping_point_mu(dirty, reference=1.0).value == pytest.approx(
            tipping_point_mu(clean, reference=1.0).value
        )


# --------------------------------------------------------------------------- #
# the surface
# --------------------------------------------------------------------------- #


class TestSurface:
    def test_shape_and_orientation(self):
        draws = _roi_draws(2.0, 0.5, 500)
        s = sensitivity_surface(draws, reference=1.0, scale="fraction_of_mean")
        assert len(s.prob) == len(s.sigma_grid)
        assert all(len(row) == len(s.mu_grid) for row in s.prob)

    def test_a_row_reproduces_the_one_dimensional_sweep(self):
        draws = _roi_draws(2.0, 0.5, 500)
        s = sensitivity_surface(
            draws,
            reference=1.0,
            scale="absolute",
            mu_grid=[0.0],
            sigma_grid=[0.0, 0.3, 0.9],
        )
        for i, sigma in enumerate(s.sigma_grid):
            direct = prob_above(
                draws, BiasPrior(mu=0.0, sigma=sigma, scale="absolute"), reference=1.0
            )
            assert s.prob[i][0] == pytest.approx(direct, abs=1e-9)

    def test_probability_falls_as_the_assumed_overstatement_grows(self):
        draws = _roi_draws(2.0, 0.4, 500)
        s = sensitivity_surface(
            draws,
            reference=1.0,
            scale="fraction_of_mean",
            mu_grid=[-0.2, 0.0, 0.2, 0.4],
            sigma_grid=[0.2],
        )
        row = list(s.prob[0])
        assert row == sorted(row, reverse=True)

    def test_thinning_is_deterministic(self):
        draws = _roi_draws(2.0, 0.5, 9000)
        a = sensitivity_surface(draws, reference=1.0, max_draws=500)
        b = sensitivity_surface(draws, reference=1.0, max_draws=500)
        assert a.prob == b.prob


# --------------------------------------------------------------------------- #
# E-value
# --------------------------------------------------------------------------- #


class TestEValue:
    def test_matches_the_published_worked_example(self):
        # VanderWeele & Ding (2017): RR = 3.9, CI (1.8, 8.7) -> E = 7.26, CI E = 3.0
        r = evalue(3.9, measure="risk_ratio", ci_low=1.8, ci_high=8.7)
        assert r.available
        assert r.point == pytest.approx(7.26, abs=0.01)
        assert r.ci_limit == pytest.approx(3.0, abs=0.01)

    def test_rr_of_two(self):
        assert evalue(2.0).point == pytest.approx(2.0 + math.sqrt(2.0), abs=1e-12)

    def test_protective_effect_is_inverted_first(self):
        assert evalue(0.5).point == pytest.approx(evalue(2.0).point)

    def test_null_crossing_interval_needs_no_confounding_at_all(self):
        r = evalue(1.5, ci_low=0.8, ci_high=2.9)
        assert r.ci_limit == pytest.approx(1.0)

    def test_refuses_a_non_ratio_measure_and_names_the_alternative(self):
        r = evalue(2.4, measure="roi")
        assert not r.available
        assert "not a risk ratio" in (r.reason or "")
        assert "tipping point" in (r.reason or "")

    def test_refuses_a_non_positive_ratio(self):
        assert not evalue(0.0).available
        assert not evalue(float("nan")).available


# --------------------------------------------------------------------------- #
# the assembled report
# --------------------------------------------------------------------------- #


class TestBiasSensitivity:
    def test_point_estimate_path_matches_the_conjugate_moments(self):
        prior = BiasPrior(mu=0.2, sigma=0.5, scale="absolute", label="p")
        res = bias_sensitivity_report(
            estimate=2.0, se=0.4, reference=1.0, priors=[prior], include_surface=False
        )
        expected = bias_adjusted_moments(2.0, 0.4, prior)
        assert res.scenarios[0].mean == pytest.approx(expected[0])
        assert res.scenarios[0].sd == pytest.approx(expected[1], rel=1e-9)

    def test_draws_and_point_paths_agree_for_a_gaussian_posterior(self):
        draws = _roi_draws(2.0, 0.4, 20000)
        prior = BiasPrior(mu=0.0, sigma=0.5, scale="absolute")
        from_draws = prob_above(draws, prior, reference=1.0)
        from_point = prob_above(np.array([2.0]), prior, reference=1.0, base_sd=0.4)
        assert from_draws == pytest.approx(from_point, abs=0.002)

    def test_verdict_vocabulary_is_closed(self):
        for mean in (0.9, 1.3, 3.0, 30.0):
            res = bias_sensitivity_report(
                _roi_draws(mean, 0.3), reference=1.0, include_surface=False
            )
            assert res.verdict in VERDICTS

    def test_a_marginal_channel_is_overturned_before_any_bias(self):
        res = bias_sensitivity_report(
            _roi_draws(1.05, 0.5), reference=1.0, include_surface=False
        )
        assert res.verdict == "overturned"
        assert "not supported even before" in res.describe()

    def test_a_thin_margin_is_fragile_and_a_wide_one_is_resilient(self):
        thin = bias_sensitivity_report(
            _roi_draws(1.6, 0.2), reference=1.0, include_surface=False
        )
        wide = bias_sensitivity_report(
            _roi_draws(9.0, 0.4), reference=1.0, include_surface=False
        )
        assert thin.verdict == "fragile"
        assert wide.verdict == "resilient"

    def test_empty_draws_are_not_assessable(self):
        res = bias_sensitivity_report(
            np.array([]), reference=1.0, include_surface=False
        )
        assert res.verdict == "not_assessable"
        assert not res.is_assessable

    def test_requires_either_draws_or_estimate_and_se(self):
        with pytest.raises(ValueError, match="either posterior"):
            bias_sensitivity_report(reference=1.0)

    def test_named_only_priors_are_flagged_as_guesses(self):
        res = bias_sensitivity_report(
            _roi_draws(3.0, 0.4), reference=1.0, include_surface=False
        )
        assert res.measured_priors == ()
        assert any("named guess" in c for c in res.caveats)

    def test_a_measured_prior_suppresses_the_guess_caveat(self):
        res = bias_sensitivity_report(
            _roi_draws(3.0, 0.4),
            reference=1.0,
            include_surface=False,
            priors=[BiasPrior(sigma=0.2, source="placebo", label="placebo")],
        )
        assert len(res.measured_priors) == 1
        assert not any("named guess" in c for c in res.caveats)

    def test_positivity_constrained_posterior_is_flagged(self):
        # Every draw above the reference: the probability is 1 by construction.
        res = bias_sensitivity_report(
            np.exp(_roi_draws(1.0, 0.2)), reference=0.0, include_surface=False
        )
        assert res.positivity_constrained
        assert any("no mass below the reference" in c for c in res.caveats)

    def test_prior_draws_disclose_what_the_prior_already_believed(self):
        # A prior that already clears the bar: the conclusion is substantially
        # the prior's, and the report has to say so.
        res = bias_sensitivity_report(
            _roi_draws(3.0, 0.4),
            reference=1.0,
            include_surface=False,
            prior_draws=_roi_draws(4.0, 1.0),
        )
        assert res.implied_prior_prob == pytest.approx(0.9987, abs=0.002)
        assert any("before seeing any data" in c for c in res.caveats)

    def test_an_uncommitted_prior_raises_no_such_caveat(self):
        res = bias_sensitivity_report(
            _roi_draws(3.0, 0.4),
            reference=1.0,
            include_surface=False,
            prior_draws=_roi_draws(1.0, 2.0),
        )
        assert res.implied_prior_prob == pytest.approx(0.5, abs=0.02)
        assert not any("before seeing any data" in c for c in res.caveats)

    def test_no_prior_draws_means_no_claim_either_way(self):
        res = bias_sensitivity_report(
            _roi_draws(3.0, 0.4), reference=1.0, include_surface=False
        )
        assert res.implied_prior_prob is None

    def test_describe_never_asserts_causality(self):
        res = bias_sensitivity_report(
            _roi_draws(6.0, 0.4), reference=1.0, include_surface=False
        )
        text = res.describe().lower()
        assert "causal" not in text and "proven" not in text

    def test_to_dict_is_json_and_msgpack_safe(self):
        res = bias_sensitivity_report(
            _roi_draws(3.0, 0.5, 500),
            reference=1.0,
            label="TV",
            units="$",
            prior_draws=_roi_draws(2.0, 1.0, 200),
        )
        payload = res.to_dict()
        json.dumps(payload)  # raises on numpy scalars

        def assert_builtin(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    assert isinstance(k, str)
                    assert_builtin(v)
            elif isinstance(node, list):
                for v in node:
                    assert_builtin(v)
            else:
                assert node is None or isinstance(
                    node, (bool, int, float, str)
                ), f"non-builtin {type(node)} in payload"

        assert_builtin(payload)


class TestPriorDominanceCaveat:
    def test_silent_when_contraction_is_unknown(self):
        assert prior_dominance_caveat(None, "resilient") is None

    def test_silent_when_the_posterior_learned_from_data(self):
        assert prior_dominance_caveat(0.85, "resilient") is None

    def test_silent_when_the_verdict_already_flags_a_problem(self):
        # Nobody is being over-reassured, so a second warning is just noise.
        assert prior_dominance_caveat(0.02, "fragile") is None
        assert prior_dominance_caveat(0.02, "overturned") is None

    def test_refuses_resilience_bought_by_a_tight_prior(self):
        msg = prior_dominance_caveat(0.05, "resilient")
        assert msg is not None
        assert "prior-dominated" in msg and "not quote it" in msg


def test_module_exports_are_importable():
    import mmm_framework.diagnostics.bias_sensitivity as mod

    for name in mod.__all__:
        assert hasattr(mod, name), name


def test_result_is_frozen():
    res = bias_sensitivity_report(
        _roi_draws(3.0, 0.4), reference=1.0, include_surface=False
    )
    assert isinstance(res, BiasSensitivity)
    with pytest.raises(Exception):
        res.verdict = "robust"  # type: ignore[misc]
