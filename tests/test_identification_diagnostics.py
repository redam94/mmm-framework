"""Weak-identification diagnostics (#203 follow-up).

Two separable contracts:

1. **The guardrail.** A bounded parameter driven onto its support boundary used
   to raise a bare ``ZeroDivisionError`` from a compiled Composite op. The
   optimization is now retried inside a float64-safe box. A fit that never
   approaches a boundary must be untouched — no bounds passed, same scipy path.
2. **The report.** Which directions the fit could not resolve. The
   discriminating property is *scale invariance*: rescaling a parameter must
   not change whether it is flagged, which is why the eigenanalysis runs on the
   correlation-scaled Hessian rather than the raw one.

The analytic models here are deliberately tiny and MCMC-free, so the contracts
are pinned by construction rather than by a fit that happens to converge.
"""

from __future__ import annotations

import warnings

import numpy as np
import pymc as pm
import pytest

from mmm_framework.diagnostics.identification import (
    SAFE_UNCONSTRAINED_LIMITS,
    FlatDirection,
    IdentificationReport,
    bounded_find_MAP,
    boundary_failure_message,
    guarded_find_MAP,
    has_guardable_parameters,
    unconstrained_box,
    weak_identification_report,
)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def _identified_model() -> pm.Model:
    """Both parameters informed by data, no trade-off."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 1.5 + 2.0 * x + rng.normal(scale=0.1, size=200)
    with pm.Model() as model:
        a = pm.Normal("a", 0.0, 5.0)
        b = pm.Normal("b", 0.0, 5.0)
        pm.Normal("y", a + b * x, 0.1, observed=y)
    return model


def _ridge_model() -> pm.Model:
    """Two coefficients on the SAME regressor: only their sum is identified."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 3.0 * x + rng.normal(scale=0.1, size=200)
    with pm.Model() as model:
        b1 = pm.Normal("b1", 0.0, 100.0)
        b2 = pm.Normal("b2", 0.0, 100.0)
        pm.Normal("y", (b1 + b2) * x, 0.1, observed=y)
    return model


def _bounded_model() -> pm.Model:
    """A [0,1] parameter the data says nothing about."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=100)
    with pm.Model() as model:
        alpha = pm.Beta("alpha", 1.0, 1.0)
        mu = pm.Normal("mu", 0.0, 5.0)
        pm.Normal("y", mu + 0.0 * alpha, 1.0, observed=y)
    return model


def _map_point(model: pm.Model) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm.find_MAP(model=model, progressbar=False, seed=0)


# --------------------------------------------------------------------------
# the box
# --------------------------------------------------------------------------
class TestTheGuardrailBox:
    def test_limits_keep_the_forward_map_off_the_boundary_in_float64(self):
        """The whole point: at the limit the transform is still representable."""
        logit_limit = SAFE_UNCONSTRAINED_LIMITS["logodds"]
        alpha = 1.0 / (1.0 + np.exp(-logit_limit))
        assert alpha < 1.0
        assert 1.0 - alpha > 0.0
        assert np.isfinite(1.0 / (1.0 - alpha))
        # ...and just past the documented cliff, it is not.
        assert 1.0 / (1.0 + np.exp(-37.0)) == 1.0

        log_limit = SAFE_UNCONSTRAINED_LIMITS["log"]
        assert np.exp(-log_limit) > 0.0
        assert np.isfinite(1.0 / np.exp(-log_limit))
        assert np.exp(-746.0) == 0.0

    def test_the_limit_never_binds_on_a_parameter_the_data_informs(self):
        """logit(0.9999) is 9.2 — an order of magnitude inside the guardrail."""
        assert np.log(0.9999 / 0.0001) < SAFE_UNCONSTRAINED_LIMITS["logodds"]

    def test_bounds_align_with_the_raveled_unconstrained_vector(self):
        model = _bounded_model()
        bounds, labels, limits = unconstrained_box(model)
        assert len(bounds) == len(labels) == len(limits)
        assert len(bounds) == sum(
            int(np.asarray(model.initial_point()[v.name]).size)
            for v in model.continuous_value_vars
        )
        by_label = dict(zip(labels, limits))
        assert by_label["alpha"] == SAFE_UNCONSTRAINED_LIMITS["logodds"]
        assert by_label["mu"] is None  # untransformed, left free

    def test_a_model_with_no_transformed_parameters_is_not_guardable(self):
        assert has_guardable_parameters(_bounded_model()) is True
        assert has_guardable_parameters(_identified_model()) is False


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------
class TestGuardedFindMAP:
    def test_a_healthy_fit_passes_no_bounds_and_returns_no_report(self):
        """Byte-identical to pm.find_MAP: same scipy code path."""
        model = _identified_model()
        calls: list[dict] = []
        real = pm.find_MAP

        def spy(**kwargs):
            calls.append(kwargs)
            return real(**kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pm, "find_MAP", spy)
            point, report = guarded_find_MAP(model, progressbar=False, seed=0)

        assert len(calls) == 1
        assert "bounds" not in calls[0]
        assert report is None
        assert np.isclose(float(point["a"]), 1.5, atol=0.05)

    def test_the_report_is_computed_on_demand_for_a_successful_fit(self):
        _point, report = guarded_find_MAP(
            _identified_model(), report=True, progressbar=False, seed=0
        )
        assert isinstance(report, IdentificationReport)
        assert report.verdict == "ok"
        assert report.guardrail_engaged is False

    def test_a_zero_division_is_retried_inside_the_box_and_reported(self):
        """The crash path, forced deterministically rather than hoped for."""
        model = _bounded_model()
        real = pm.find_MAP
        seen: list[bool] = []

        def flaky(**kwargs):
            bounded = "bounds" in kwargs
            seen.append(bounded)
            if not bounded:
                raise ZeroDivisionError("division by zero")
            return real(**kwargs)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pm, "find_MAP", flaky)
            with pytest.warns(UserWarning, match="guardrail"):
                point, report = guarded_find_MAP(model, progressbar=False, seed=0)

        assert seen == [False, True]  # tried unguarded first, then the box
        assert report is not None and report.guardrail_engaged is True
        assert point is not None

    def test_an_unguardable_model_re_raises_with_the_explanation(self):
        model = _identified_model()  # no transformed parameters to bound

        def always_fails(**kwargs):
            raise ZeroDivisionError("division by zero")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pm, "find_MAP", always_fails)
            with pytest.raises(ZeroDivisionError, match="issues/203"):
                guarded_find_MAP(model, progressbar=False, seed=0)

    def test_a_retry_that_also_fails_surfaces_both_causes(self):
        model = _bounded_model()

        def always_fails(**kwargs):
            raise ZeroDivisionError("division by zero")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(pm, "find_MAP", always_fails)
            with pytest.raises(ZeroDivisionError) as exc:
                guarded_find_MAP(model, progressbar=False, seed=0)
        assert "bounded retry was attempted and also failed" in str(exc.value)

    def test_a_bounds_incapable_optimizer_is_swapped_and_the_swap_is_recorded(self):
        _point, report = bounded_find_MAP(
            _bounded_model(), method="BFGS", progressbar=False, seed=0
        )
        assert any("L-BFGS-B" in note for note in report.notes)

    def test_the_failure_message_names_this_models_bounded_parameters(self):
        message = boundary_failure_message(_bounded_model())
        assert "alpha" in message
        assert "NUTS" in message
        assert "issues/203" in message


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
class TestWeakIdentificationReport:
    def test_an_identified_model_is_clean(self):
        model = _identified_model()
        report = weak_identification_report(model, _map_point(model))
        assert report.verdict == "ok"
        assert report.flat_directions == ()
        assert report.uninformed == ()
        assert report.laplace_usable is True

    def test_a_ridge_is_found_and_its_loadings_name_both_parameters(self):
        model = _ridge_model()
        report = weak_identification_report(model, _map_point(model))
        assert report.verdict == "weak"
        assert report.flat_directions, "the b1+b2 ridge should be reported"

        ridge = report.flat_directions[0]
        assert ridge.kind == "ridge"
        assert ridge.condition_index > 30.0
        named = {name for name, _ in ridge.loadings}
        assert named == {"b1", "b2"}
        # Only the SUM is identified, so the flat direction is the difference:
        # equal magnitudes, opposite signs.
        loadings = dict(ridge.loadings)
        assert np.isclose(abs(loadings["b1"]), abs(loadings["b2"]), atol=0.05)
        assert loadings["b1"] * loadings["b2"] < 0
        assert report.laplace_usable is False

    def test_an_uninformed_parameter_is_reported_separately_from_a_ridge(self):
        """Nothing trades off — one coordinate is simply flat."""
        model = _bounded_model()
        point = dict(_map_point(model))
        # Park alpha far out where its own curvature vanishes.
        point["alpha_logodds__"] = np.array(-25.0)
        point["alpha"] = np.array(1.0 / (1.0 + np.exp(25.0)))

        report = weak_identification_report(model, point)
        assert {u.parameter for u in report.uninformed} == {"alpha"}
        assert report.uninformed[0].relative_curvature < 1e-8
        assert report.verdict == "non-identified"
        assert report.laplace_usable is False

    def test_the_verdict_is_invariant_to_rescaling_a_parameter(self):
        """The property that raw Hessian eigenvalues do NOT have.

        Multiplying a regressor by 1000 rescales that coefficient's curvature by
        1e6 and moves the raw spectrum by six orders of magnitude, without
        changing anything about what is identified.
        """
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        y = 2.0 * x + rng.normal(scale=0.1, size=200)

        def build(scale: float) -> pm.Model:
            with pm.Model() as model:
                b = pm.Normal("b", 0.0, 10.0)
                c = pm.Normal("c", 0.0, 10.0)
                pm.Normal("y", b * x + c * (x * scale), 0.1, observed=y)
            return model

        verdicts, raw_spreads = [], []
        for scale in (1.0, 1000.0):
            model = build(scale)
            point = _map_point(model)
            verdicts.append(weak_identification_report(model, point).verdict)

            rvs = [model.values_to_rvs[v] for v in model.continuous_value_vars]
            hessian = -np.asarray(
                model.compile_d2logp(vars=rvs, jacobian=True, negate_output=False)(
                    {v.name: point[v.name] for v in model.continuous_value_vars}
                )
            )
            raw = np.linalg.eigvalsh((hessian + hessian.T) / 2)
            raw_spreads.append(abs(raw).max())

        assert verdicts[0] == verdicts[1] == "weak"
        # The raw spectrum moved by ~1e6; the verdict did not move at all.
        assert raw_spreads[1] / raw_spreads[0] > 1e4

    def test_a_saddle_point_is_flagged_as_nonconvex(self):
        """Sign survives the diagonal scaling (Sylvester's law of inertia)."""
        with pm.Model() as model:
            x = pm.Normal("x", 0.0, 10.0)
            y = pm.Normal("y", 0.0, 10.0)
            pm.Potential("saddle", -(x**2) + y**2)

        point = {"x": np.array(0.0), "y": np.array(0.0)}
        report = weak_identification_report(model, point)
        assert report.nonconvex is True
        assert report.verdict == "non-identified"
        assert any(d.kind == "nonconvex" for d in report.flat_directions)
        assert report.laplace_usable is False

    def test_a_diagnostic_failure_degrades_to_a_note_and_never_raises(self):
        model = _identified_model()
        report = weak_identification_report(model, {"a": np.array(0.0)})
        assert report.verdict == "unknown"
        assert report.notes and "curvature unavailable" in report.notes[0]

    def test_the_payload_is_checkpointer_safe(self):
        """numpy scalars crash the LangGraph msgpack checkpointer."""
        model = _ridge_model()
        payload = weak_identification_report(model, _map_point(model)).to_dict()

        def assert_plain(value, path="report"):
            if isinstance(value, dict):
                for key, item in value.items():
                    assert isinstance(key, str), path
                    assert_plain(item, f"{path}.{key}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    assert_plain(item, f"{path}[{i}]")
            else:
                assert value is None or isinstance(
                    value, (str, bool, int, float)
                ), f"{path} is {type(value)}"
                assert not isinstance(value, np.generic), f"{path} is a numpy scalar"

        assert_plain(payload)

    def test_the_summary_reads_as_prose(self):
        model = _ridge_model()
        text = weak_identification_report(model, _map_point(model)).summary()
        assert "trade off" in text
        assert "b1" in text and "b2" in text
        assert "Laplace" in text

    def test_the_ridge_wording_follows_the_attribution(self):
        """A prior-held ridge and a merely-imprecise one need different fixes,
        so the text must not claim "the prior decides this" when the data
        supplied most of the curvature."""
        prior_held = FlatDirection(
            eigenvalue=1e-9,
            condition_index=200.0,
            loadings=(("a", 0.71), ("b", -0.71)),
            kind="ridge",
            informed_fraction=0.01,
        ).describe()
        assert "experiment" in prior_held
        assert "weak evidence" not in prior_held

        imprecise = FlatDirection(
            eigenvalue=1e-9,
            condition_index=200.0,
            loadings=(("a", 0.71), ("b", -0.71)),
            kind="ridge",
            informed_fraction=0.64,
        ).describe()
        assert "weak evidence rather than none" in imprecise
        assert "experiment" not in imprecise


# --------------------------------------------------------------------------
# prior vs likelihood attribution
# --------------------------------------------------------------------------
def _shrunk_model() -> pm.Model:
    """A prior tight enough to swamp five noisy observations."""
    rng = np.random.default_rng(4)
    y = 0.3 + rng.normal(scale=3.0, size=5)
    with pm.Model() as model:
        theta = pm.Normal("theta", 0.0, 0.05)
        pm.Normal("y", theta, 3.0, observed=y)
    return model


def _hessian_of(model: pm.Model, cost, point) -> np.ndarray:
    from pymc.pytensorf import hessian, rewrite_pregrad

    graph = hessian(
        rewrite_pregrad(cost), list(model.continuous_value_vars), negate_output=False
    )
    fn = model.compile_fn(inputs=model.value_vars, outs=graph, on_unused_input="ignore")
    matrix = -np.asarray(fn({v.name: point[v.name] for v in model.value_vars}))
    return (matrix + matrix.T) / 2


class TestCurvatureAttribution:
    """The whole split rests on H_post == H_prior + H_lik. Pin it."""

    @pytest.mark.parametrize(
        "factory", [_identified_model, _ridge_model, _bounded_model, _shrunk_model]
    )
    def test_the_posterior_curvature_is_exactly_prior_plus_likelihood(self, factory):
        model = factory()
        point = _map_point(model)
        posterior = _hessian_of(model, model.logp(jacobian=True), point)
        prior = _hessian_of(model, model.varlogp, point)
        likelihood = _hessian_of(model, model.datalogp, point)

        residual = np.abs(posterior - (prior + likelihood)).max()
        assert residual <= 1e-8 * max(np.abs(posterior).max(), 1.0)

    def test_effective_parameters_counts_what_the_data_determines(self):
        """Two well-measured coefficients: the data determines both."""
        model = _identified_model()
        report = weak_identification_report(model, _map_point(model))
        assert report.effective_parameters == pytest.approx(2.0, abs=0.05)
        assert report.n_parameters == 2
        assert report.prior_determined == ()

    def test_two_coefficients_sharing_one_regressor_count_as_one(self):
        """The classic result: n parameters, but only their sum is measured."""
        model = _ridge_model()
        report = weak_identification_report(model, _map_point(model))
        assert report.n_parameters == 2
        assert report.effective_parameters == pytest.approx(1.0, abs=0.05)

        # The ridge is the DIFFERENCE, and the data contributes none of it.
        ridge = report.flat_directions[0]
        assert ridge.informed_fraction == pytest.approx(0.0, abs=1e-3)

    def test_a_tight_prior_swamping_the_data_counts_as_almost_nothing(self):
        model = _shrunk_model()
        report = weak_identification_report(model, _map_point(model))
        assert report.effective_parameters < 0.05

    def test_a_parameter_the_likelihood_ignores_is_flagged_prior_determined(self):
        """The gap the other two tests cannot see.

        `alpha` enters the likelihood multiplied by zero. Its posterior is NOT
        flat — the Beta prior curves it perfectly well — and it is nowhere near
        a boundary, so neither the ridge test nor the uninformed test fires.
        Only the attribution catches it.
        """
        model = _bounded_model()
        report = weak_identification_report(model, _map_point(model))

        assert {p.parameter for p in report.prior_determined} == {"alpha"}
        assert report.prior_determined[0].informed_fraction == pytest.approx(
            0.0, abs=1e-6
        )
        assert report.uninformed == ()  # not flat: the prior holds it up
        assert report.flat_directions == ()  # not a ridge: nothing trades off
        assert report.verdict == "weak"
        assert report.effective_parameters == pytest.approx(1.0, abs=0.05)

    def test_finding_a_prior_determined_parameter_is_silent(self):
        """PyTensor warns when differentiating a disconnected variable, which is
        exactly what a prior-determined parameter is. The diagnostic must not
        get loudest when it succeeds."""
        model = _bounded_model()
        point = _map_point(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = weak_identification_report(model, point)
        assert report.prior_determined  # it did find it
        assert not [
            w for w in caught if "not part of the computational graph" in str(w.message)
        ]

    def test_the_per_parameter_fraction_is_not_used_to_claim_identification(self):
        """The trap: the DIAGONAL cannot see a ridge.

        In the shared-regressor model both coefficients score ~1.0 on the
        diagonal — the likelihood does curve in each coordinate — while only
        their sum is identified. So a high diagonal fraction must never be read
        as "identified", and nothing here reports it as such.
        """
        model = _ridge_model()
        point = _map_point(model)
        posterior = _hessian_of(model, model.logp(jacobian=True), point)
        likelihood = _hessian_of(model, model.datalogp, point)
        diagonal_fractions = np.diag(likelihood) / np.diag(posterior)
        assert np.allclose(diagonal_fractions, 1.0, atol=1e-3)

        # ...and yet:
        report = weak_identification_report(model, point)
        assert report.effective_parameters == pytest.approx(1.0, abs=0.05)
        assert report.verdict == "weak"
        assert report.prior_determined == ()

    def test_attribution_can_be_switched_off(self):
        report = weak_identification_report(
            _identified_model(), _map_point(_identified_model()), attribute=False
        )
        assert report.effective_parameters is None
        assert report.prior_determined == ()
        assert all(d.informed_fraction is None for d in report.flat_directions)

    def test_a_failed_split_leaves_the_rest_of_the_report_intact(self):
        model = _ridge_model()
        point = _map_point(model)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "mmm_framework.diagnostics.identification._likelihood_curvature",
                lambda *a, **k: None,
            )
            report = weak_identification_report(model, point)

        assert report.effective_parameters is None
        assert report.flat_directions  # the ridge is still found
        assert report.verdict == "weak"

    def test_the_summary_states_the_effective_parameter_count(self):
        model = _bounded_model()
        text = weak_identification_report(model, _map_point(model)).summary()
        assert "alpha" in text
        assert "prior" in text
        assert "determines about" in text
