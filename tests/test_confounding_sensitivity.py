"""Tests for decision-scale confounding sensitivity and its calibration.

The suite is in three layers.

**The algebra** is checked against an *exact identity* rather than against
remembered reference numbers: Cinelli & Hazlett's Theorem 1 says that when ``Z``
really is the omitted confounder, ``|bias| = se * sqrt(df) * BF`` holds with
equality. So fitting the same data with and without ``Z`` and comparing the
predicted bias to the realized one pins the formula to machine precision, and
would catch any transcription error in either direction.

**The column-selection guard** is checked on hand-built matrices, because the
failure it prevents is silent: a rank-deficient design makes every partial
``R^2`` depend on an arbitrary choice of least-squares solution, and ``pinv``
would hand one back without complaint.

**The end-to-end behaviour** runs on the planted-truth confounding worlds, whose
two variants share a seed and differ only in whether the demand proxy is adjusted
for.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mmm_framework.validation.confounding_sensitivity import (
    BenchmarkReport,
    _ols,
    _partial_r2,
    _select_independent_columns,
    adjusted_se_from_r2,
    benchmark_bias_priors,
    bias_bound,
    ovb_partial_r2_bound,
    run_confounding_sensitivity,
)

# =========================================================================== #
# the algebra
# =========================================================================== #


def _ovb_case(seed: int, *, lam: float, gamma: float, n: int = 400):
    """Fit the same linear model with and without the confounder ``Z``.

    Returns everything Cinelli & Hazlett's identity relates: the observed
    ("short") estimate and its standard error, the adjusted ("long") ones, and
    the two partial ``R^2`` values the bound is expressed in.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    d = lam * z + 0.3 * x + rng.normal(0, 1.0, n)
    y = 1.5 * d + gamma * z - 0.5 * x + rng.normal(0, 1.0, n)
    one = np.ones(n)

    short = _ols(np.column_stack([one, d, x]), y)
    long_ = _ols(np.column_stack([one, d, x, z]), y)
    treat = _ols(np.column_stack([one, x, z]), d)
    return {
        "b_short": float(short.coef[1]),
        "se_short": float(short.se[1]),
        "df_short": int(short.dof),
        "b_long": float(long_.coef[1]),
        "se_long": float(long_.se[1]),
        "r2yz_dx": _partial_r2(float(long_.t[3]), long_.dof),
        "r2dz_x": _partial_r2(float(treat.t[2]), treat.dof),
    }


class TestOmittedVariableBiasIdentity:
    """``|bias| = se * sqrt(df) * BF`` holds with equality for the true confounder."""

    @pytest.mark.parametrize(
        "seed,lam,gamma",
        [(0, 0.8, 2.0), (1, 0.2, 0.5), (2, 1.5, -3.0), (3, 0.05, 0.1)],
    )
    def test_predicted_bias_equals_realized_bias(self, seed, lam, gamma):
        c = _ovb_case(seed, lam=lam, gamma=gamma)
        predicted = bias_bound(c["se_short"], c["df_short"], c["r2dz_x"], c["r2yz_dx"])
        realized = abs(c["b_short"] - c["b_long"])
        assert predicted == pytest.approx(realized, rel=1e-9)

    @pytest.mark.parametrize("seed,lam,gamma", [(0, 0.8, 2.0), (2, 1.5, -3.0)])
    def test_predicted_adjusted_se_equals_realized(self, seed, lam, gamma):
        c = _ovb_case(seed, lam=lam, gamma=gamma)
        predicted = adjusted_se_from_r2(
            c["se_short"], c["df_short"], c["r2dz_x"], c["r2yz_dx"]
        )
        assert predicted == pytest.approx(c["se_long"], rel=1e-9)

    def test_a_confounder_with_no_treatment_link_produces_no_bias(self):
        """A back-door needs *both* arms.

        Needs a large sample to make the point cleanly: at ``lam = 0`` the two are
        independent in the population but not in any finite draw, and a strong
        outcome arm (``gamma = 3``) amplifies whatever sample correlation exists —
        which is itself the honest lesson, so the comparison against a genuinely
        linked confounder is asserted alongside the absolute bound.
        """
        unlinked = _ovb_case(9, lam=0.0, gamma=3.0, n=40_000)
        linked = _ovb_case(9, lam=1.0, gamma=3.0, n=40_000)
        b_unlinked = bias_bound(
            unlinked["se_short"],
            unlinked["df_short"],
            unlinked["r2dz_x"],
            unlinked["r2yz_dx"],
        )
        b_linked = bias_bound(
            linked["se_short"],
            linked["df_short"],
            linked["r2dz_x"],
            linked["r2yz_dx"],
        )
        assert b_unlinked < 0.05 * abs(unlinked["b_long"])
        assert b_linked > 20 * b_unlinked

    def test_guards_return_nan_rather_than_a_wrong_number(self):
        assert np.isnan(bias_bound(0.1, 0, 0.2, 0.2))
        assert np.isnan(bias_bound(0.1, 100, 1.0, 0.2))
        assert np.isnan(adjusted_se_from_r2(0.1, 1, 0.2, 0.2))


class TestOvbPartialR2Bound:
    def test_reproduces_the_sensemakr_recursion(self):
        r2dxj, r2yxj, kd = 0.10, 0.20, 1.0
        r2dz, r2yz, r2zxj, saturated, reason = ovb_partial_r2_bound(r2dxj, r2yxj, kd=kd)
        assert r2dz == pytest.approx(kd * r2dxj / (1 - r2dxj))
        expected_zxj = kd * r2dxj**2 / ((1 - kd * r2dxj) * (1 - r2dxj))
        assert r2zxj == pytest.approx(expected_zxj)
        expected_yz = (
            (np.sqrt(kd) + np.sqrt(expected_zxj)) / np.sqrt(1 - expected_zxj)
        ) ** 2 * (r2yxj / (1 - r2yxj))
        assert r2yz == pytest.approx(expected_yz)
        assert not saturated and reason == ""

    def test_the_binding_validity_condition_is_r2dxj_below_one_over_one_plus_kd(self):
        """At ``kd = 1`` the bound dies at 0.5, not at 1.0.

        The naive condition ``kd * r2dxj < 1`` is the *non-binding* one. Past the
        real threshold the published formula square-roots a negative number and
        yields ``NaN`` — which compares ``False`` against every fragility test and
        would therefore be read as "not fragile". It has to be a refusal.
        """
        # Just inside the boundary, and weak enough on the outcome side not to
        # saturate (the two failure modes are separate and are tested apart).
        ok = ovb_partial_r2_bound(0.45, 0.01, kd=1.0)
        assert np.isfinite(ok[1]) and ok[4] == ""

        for bad in (0.50, 0.60, 0.80):
            r2dz, r2yz, _z, _sat, reason = ovb_partial_r2_bound(bad, 0.01, kd=1.0)
            assert not np.isfinite(r2yz), f"r2dxj={bad} must not yield a number"
            assert "impossible" in reason
            assert "largest admissible kd" in reason or "degenerate" in reason

    def test_the_threshold_moves_with_kd(self):
        # r2dxj = 0.30: fine at kd=1 (0.30 < 0.5), impossible at kd=3 (0.30 > 0.25)
        assert np.isfinite(ovb_partial_r2_bound(0.30, 0.2, kd=1.0)[1])
        assert not np.isfinite(ovb_partial_r2_bound(0.30, 0.2, kd=3.0)[1])

    def test_the_reason_names_the_largest_admissible_multiplier(self):
        reason = ovb_partial_r2_bound(0.6, 0.2, kd=1.0)[4]
        # r2dxj = 0.6 -> max kd = (1 - 0.6) / 0.6 = 0.667
        assert "0.67" in reason

    def test_outcome_side_saturation_is_clipped_and_flagged(self):
        _r2dz, r2yz, _z, saturated, reason = ovb_partial_r2_bound(0.1, 0.9, kd=3.0)
        assert saturated and r2yz == pytest.approx(1.0)
        assert "saturates" in reason

    def test_stronger_assumed_confounders_imply_more_bias(self):
        fracs = []
        for kd in (0.5, 1.0, 2.0):
            r2dz, r2yz, *_ = ovb_partial_r2_bound(0.05, 0.10, kd=kd)
            fracs.append(bias_bound(0.02, 500, r2dz, r2yz))
        assert fracs == sorted(fracs)

    def test_rejects_out_of_range_inputs(self):
        assert not np.isfinite(ovb_partial_r2_bound(1.5, 0.2)[1])
        assert not np.isfinite(ovb_partial_r2_bound(0.2, -0.1)[1])


# =========================================================================== #
# the column-selection guard
# =========================================================================== #


def _span_projector(X: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(X)
    return q @ q.T


class TestColumnSelection:
    def test_a_full_rank_design_passes_through_untouched(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
        names = ["intercept", "media_TV", "control_Price", "control_Temp"]
        blocks = {
            "intercept": slice(0, 1),
            "media": slice(1, 2),
            "controls": slice(2, 4),
        }
        Xk, kept, dropped, reason = _select_independent_columns(X, names, blocks)
        assert reason == "" and dropped == [] and kept == names
        assert np.allclose(Xk, X)

    def test_a_redundant_control_is_dropped_not_the_basis_it_duplicates(self):
        """Priority order is the whole point of the scan.

        A control that is an exact function of the seasonal basis carries no
        independent information. Dropping the *seasonality* column instead would
        let that control stand in for it and then be benchmarked as though it
        were a real confounder.
        """
        t = np.arange(120)
        season = np.cos(2 * np.pi * t / 52.0)
        rng = np.random.default_rng(1)
        price = 12.0 + 0.5 * season  # exactly what the synth world builds
        X = np.column_stack([np.ones(120), season, rng.normal(size=120), price])
        names = ["intercept", "season_yearly[0]", "media_TV", "control_Price"]
        blocks = {
            "intercept": slice(0, 1),
            "seasonality": slice(1, 2),
            "media": slice(2, 3),
            "controls": slice(3, 4),
        }
        Xk, kept, dropped, reason = _select_independent_columns(X, names, blocks)
        assert reason == ""
        assert dropped == ["control_Price"]
        assert "season_yearly[0]" in kept

    def test_dropping_preserves_the_column_space_exactly(self):
        t = np.arange(120)
        season = np.cos(2 * np.pi * t / 52.0)
        rng = np.random.default_rng(2)
        X = np.column_stack(
            [np.ones(120), season, rng.normal(size=120), 12.0 + 0.5 * season]
        )
        names = ["intercept", "season_yearly[0]", "media_TV", "control_Price"]
        blocks = {
            "intercept": slice(0, 1),
            "seasonality": slice(1, 2),
            "media": slice(2, 3),
            "controls": slice(3, 4),
        }
        Xk, *_ = _select_independent_columns(X, names, blocks)
        assert np.allclose(_span_projector(Xk), _span_projector(X[:, :3]), atol=1e-10)

    def test_a_geo_dummy_set_loses_exactly_one_reference_level(self):
        n_geo, n_per = 3, 20
        geo = np.repeat(np.arange(n_geo), n_per)
        dummies = np.column_stack([(geo == g).astype(float) for g in range(n_geo)])
        rng = np.random.default_rng(3)
        X = np.column_stack(
            [np.ones(n_geo * n_per), dummies, rng.normal(size=n_geo * n_per)]
        )
        names = [
            "intercept",
            "geo_effect[0]",
            "geo_effect[1]",
            "geo_effect[2]",
            "media_TV",
        ]
        blocks = {"intercept": slice(0, 1), "geo": slice(1, 4), "media": slice(4, 5)}
        Xk, kept, dropped, reason = _select_independent_columns(X, names, blocks)
        assert reason == ""
        assert len(dropped) == 1 and dropped[0].startswith("geo_effect")
        assert Xk.shape[1] == 4
        assert np.linalg.matrix_rank(Xk) == 4

    def test_collinear_media_is_refused_and_names_the_channel(self):
        rng = np.random.default_rng(4)
        tv = rng.normal(size=60)
        X = np.column_stack([np.ones(60), tv, 2.0 * tv])
        names = ["intercept", "media_TV", "media_Clone"]
        blocks = {"intercept": slice(0, 1), "media": slice(1, 3)}
        _Xk, _kept, _dropped, reason = _select_independent_columns(X, names, blocks)
        assert "Clone" in reason
        assert "not separately identified" in reason

    def test_a_zero_column_is_dropped_rather_than_dividing_by_zero(self):
        X = np.column_stack([np.ones(30), np.zeros(30), np.arange(30.0)])
        names = ["intercept", "control_Dead", "media_TV"]
        blocks = {
            "intercept": slice(0, 1),
            "controls": slice(1, 2),
            "media": slice(2, 3),
        }
        _Xk, kept, dropped, reason = _select_independent_columns(X, names, blocks)
        assert reason == "" and dropped == ["control_Dead"]
        assert "media_TV" in kept


class TestBenchmarkReportContainer:
    def test_unavailable_report_is_still_a_usable_object(self):
        r = BenchmarkReport(status="unavailable", reason="no design matrix")
        assert r.strongest("TV") is None
        payload = r.to_dict()
        json.dumps(payload)
        assert payload["status"] == "unavailable"

    def test_refused_bounds_never_win_the_strongest_comparison(self):
        from mmm_framework.validation.confounding_sensitivity import BenchmarkBound

        def _bound(status, frac):
            return BenchmarkBound(
                channel="TV",
                covariate="Price",
                kd=1.0,
                ky=1.0,
                r2dxj_x=0.1,
                r2yxj_dx=0.1,
                r2dz_x=0.1,
                r2yz_dx=0.1,
                estimate=1.0,
                se=0.1,
                dof=100,
                bias=frac,
                fractional_bias=frac,
                adjusted_estimate=1.0 - frac,
                adjusted_se=0.1,
                saturated=False,
                status=status,
            )

        r = BenchmarkReport(
            bounds={"TV": [_bound("ok", 0.1), _bound("refused", float("nan"))]}
        )
        best = r.strongest("TV")
        assert best is not None and best.fractional_bias == pytest.approx(0.1)


# =========================================================================== #
# end to end on the planted-truth worlds
# =========================================================================== #


def _fit_world(name: str, *, extra_noise_control: bool = False, n_weeks: int = 120):
    from mmm_framework.config import InferenceMethod, ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
    from mmm_framework.synth import dgp

    sc = dgp.build(name, seed=1, n_weeks=n_weeks)
    if extra_noise_control:
        from mmm_framework.config import CausalControlRole

        rng = np.random.default_rng(99)
        sc.controls = sc.controls.assign(PureNoise=rng.normal(0, 1, len(sc.controls)))
        roles = dict(sc.control_roles or {})
        roles["PureNoise"] = CausalControlRole.CONFOUNDER
        sc.control_roles = roles

    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=2,
        n_draws=300,
        n_tune=300,
        use_parametric_adstock=True,
        optim_seed=7,
    )
    mmm = BayesianMMM(sc.panel(), cfg, TrendConfig(type=TrendType.LINEAR))
    mmm.fit(random_seed=7, progressbar=False)
    return sc, mmm


@pytest.fixture(scope="module")
def controlled_fit():
    return _fit_world("confounding_controlled", extra_noise_control=True)


@pytest.fixture(scope="module")
def confounded_fit():
    return _fit_world("unobserved_confounding")


@pytest.mark.slow
class TestBenchmarkOnPlantedTruth:
    def test_a_real_confounder_proxy_is_benchmarkable(self, controlled_fit):
        _sc, mmm = controlled_fit
        report = benchmark_bias_priors(mmm)
        assert report.status == "ok", report.reason
        assert "CategoryDemand" in report.covariates

    def test_a_pure_noise_control_implies_far_less_bias(self, controlled_fit):
        """The benchmark must discriminate signal from noise.

        ``CategoryDemand`` is the demand proxy that actually drives both spend and
        sales; ``PureNoise`` is an independent normal draw carrying the same role
        label. If the bound cannot separate them it is measuring the design rather
        than the confounding.
        """
        _sc, mmm = controlled_fit
        report = benchmark_bias_priors(mmm, kd=(1.0,))
        ratios = []
        for _ch, rows in report.bounds.items():
            by_cov = {r.covariate: r.fractional_bias for r in rows if r.status == "ok"}
            if {"CategoryDemand", "PureNoise"} <= set(by_cov):
                ratios.append(
                    by_cov["CategoryDemand"] / max(by_cov["PureNoise"], 1e-12)
                )
        assert ratios, "expected both covariates to be benchmarked"
        assert min(ratios) > 2.0, f"noise not separated from signal: {ratios}"

    def test_bias_grows_monotonically_with_the_multiplier(self, controlled_fit):
        _sc, mmm = controlled_fit
        report = benchmark_bias_priors(mmm, kd=(1.0, 2.0, 3.0))
        for ch, rows in report.bounds.items():
            per_cov: dict[str, list[tuple[float, float]]] = {}
            for r in rows:
                if r.status == "ok":
                    per_cov.setdefault(r.covariate, []).append(
                        (r.kd, r.fractional_bias)
                    )
            for cov, pairs in per_cov.items():
                fracs = [f for _k, f in sorted(pairs)]
                assert fracs == sorted(fracs), f"{ch}/{cov} not monotone: {fracs}"

    def test_a_degenerate_control_is_reported_not_silently_skipped(
        self, confounded_fit
    ):
        """That world's ``Price`` is exactly ``12 + 0.5*cos(2*pi*t/52)``.

        It is an exact function of the yearly Fourier basis, so it carries no
        information to benchmark against. The right behaviour is to say so — a
        silent skip would read as "we checked and found nothing".
        """
        _sc, mmm = confounded_fit
        report = benchmark_bias_priors(mmm)
        assert report.status == "unavailable"
        assert "no control variable is available" in report.reason
        assert "control_Price" in report.dropped_columns


@pytest.mark.slow
class TestConfoundingSensitivityReport:
    def test_every_channel_gets_a_tipping_point(self, confounded_fit):
        _sc, mmm = confounded_fit
        report = run_confounding_sensitivity(mmm, include_surface=False)
        assert len(report.channels) == len(mmm.channel_names)
        for c in report.channels:
            assert c.sensitivity.verdict in (
                "overturned",
                "fragile",
                "resilient",
                "not_assessable",
            )
            assert np.isfinite(c.sensitivity.estimate)

    def test_a_thinner_margin_gets_a_smaller_tipping_point(self, confounded_fit):
        """The one ordering the device must always respect."""
        _sc, mmm = confounded_fit
        report = run_confounding_sensitivity(mmm, include_surface=False)
        pairs = [
            (c.sensitivity.estimate, c.sensitivity.tipping_mu.value)
            for c in report.channels
            if c.sensitivity.tipping_mu.crossed
            and c.sensitivity.tipping_mu.value is not None
        ]
        assert len(pairs) >= 2
        by_roi = sorted(pairs)
        tips = [t for _r, t in by_roi]
        assert tips == sorted(tips), f"tipping points not ordered by margin: {by_roi}"

    def test_an_unavailable_benchmark_is_stated_in_the_caveat(self, confounded_fit):
        _sc, mmm = confounded_fit
        report = run_confounding_sensitivity(mmm, include_surface=False, benchmark=True)
        assert "No observed-covariate benchmark was available" in report.caveat

    def test_the_caveat_never_claims_causality(self, confounded_fit):
        _sc, mmm = confounded_fit
        report = run_confounding_sensitivity(mmm, include_surface=False)
        assert "does not test it" in report.caveat
        assert "never evidence that the effect is causal" in report.caveat

    def test_benchmarks_become_scenarios_labelled_by_their_source(self, controlled_fit):
        _sc, mmm = controlled_fit
        report = run_confounding_sensitivity(mmm, include_surface=False)
        measured = [
            s
            for c in report.channels
            for s in c.sensitivity.measured_priors
            if s.prior.source.startswith("benchmark:")
        ]
        assert measured, "benchmark bounds should ride into the scenario ladder"

    def test_benchmark_comparison_is_none_when_there_is_nothing_to_compare(
        self, confounded_fit
    ):
        """``None`` is not ``False``.

        With no benchmark there is no evidence either way, and rendering that as
        "the benchmark does not overturn it" would be reassurance the analysis
        never earned.
        """
        _sc, mmm = confounded_fit
        report = run_confounding_sensitivity(mmm, include_surface=False)
        assert all(c.benchmark_exceeds_tipping_point is None for c in report.channels)

    def test_summary_has_one_row_per_channel(self, confounded_fit):
        _sc, mmm = confounded_fit
        report = run_confounding_sensitivity(mmm, include_surface=False)
        df = report.summary()
        assert len(df) == len(mmm.channel_names)
        assert "Tipping point" in df.columns

    def test_to_dict_is_json_safe(self, controlled_fit):
        _sc, mmm = controlled_fit
        report = run_confounding_sensitivity(mmm, include_surface=True, max_draws=100)
        json.dumps(report.to_dict())

    def test_unassessable_channels_are_tracked_separately_from_fragile(
        self, confounded_fit
    ):
        _sc, mmm = confounded_fit
        report = run_confounding_sensitivity(mmm, include_surface=False)
        assert set(report.unassessable_channels).isdisjoint(report.fragile_channels)

    def test_an_unfitted_model_is_refused(self):
        from types import SimpleNamespace

        with pytest.raises(ValueError, match="must be fitted"):
            run_confounding_sensitivity(SimpleNamespace(_trace=None))


# =========================================================================== #
# reporting surfaces (fast — canned payloads, no fitting)
# =========================================================================== #


def _report_payload(**overrides) -> dict:
    channel = {
        "channel": "TV",
        "metric_label": "ROI",
        "benchmark_exceeds_tipping_point": False,
        "sensitivity": {
            "verdict": "resilient",
            "estimate": 2.8,
            "prob_at_zero_bias": 0.99,
            "tipping_mu": {
                "value": 0.42,
                "crossed": True,
                "already_below": False,
                "max_scanned": 1.5,
                "curve": [[0.0, 0.99], [0.4, 0.96], [0.8, 0.6], [1.5, 0.1]],
            },
        },
        "benchmarks": [
            {
                "status": "ok",
                "fractional_bias": 0.18,
                "kd": 3.0,
                "covariate": "Price",
            }
        ],
    }
    channel.update(overrides)
    return {
        "threshold": 0.95,
        "caveat": "Prices the assumption.",
        "channels": [channel],
    }


class TestReportSection:
    def _render(self, payload: dict) -> str:
        from mmm_framework.reporting.config import ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle
        from mmm_framework.reporting.sections import CausalAssumptionsSection

        bundle = MMMDataBundle(causal_assumptions={"confounding_sensitivity": payload})
        return CausalAssumptionsSection(bundle, ReportConfig()).render()

    def test_renders_the_tipping_point_table(self):
        html = self._render(_report_payload())
        assert "How much hidden bias would change the call?" in html
        assert "42%" in html and "18%" in html

    def test_a_benchmark_beyond_the_tipping_point_changes_the_status(self):
        """The comparison that turns a slider into an argument."""
        clean = self._render(_report_payload())
        exceeded = self._render(_report_payload(benchmark_exceeds_tipping_point=True))
        assert "Benchmark would overturn it" not in clean
        assert "Benchmark would overturn it" in exceeded

    def test_never_prints_a_literal_nan(self):
        html = self._render(
            _report_payload(
                sensitivity={
                    "verdict": "not_assessable",
                    "estimate": float("nan"),
                    "prob_at_zero_bias": float("nan"),
                    "tipping_mu": {
                        "value": None,
                        "crossed": False,
                        "already_below": False,
                        "max_scanned": 1.5,
                    },
                },
                benchmarks=[],
            )
        )
        assert "Not assessable" in html
        assert ">nan<" not in html.lower()
        assert html.count("n/a") >= 4

    def test_channel_names_are_escaped(self):
        html = self._render(_report_payload(channel="<script>alert(1)</script>"))
        assert "&lt;script&gt;" in html
        assert "<script>alert(1)</script>" not in html

    def test_a_missing_payload_still_renders_the_standing_caveat(self):
        from mmm_framework.reporting.config import ReportConfig
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle
        from mmm_framework.reporting.sections import CausalAssumptionsSection

        html = CausalAssumptionsSection(MMMDataBundle(), ReportConfig()).render()
        assert "no unobserved confounding" in html
        assert "How much hidden bias" not in html

    def test_the_status_wording_never_asserts_causality(self):
        html = self._render(_report_payload())
        assert "Withstands the range tested" in html
        assert "proven" not in html.lower()


class TestCharts:
    def _cfg(self):
        from mmm_framework.reporting.config import ReportConfig

        return ReportConfig()

    def test_tipping_chart_renders_bars_and_benchmarks(self):
        from mmm_framework.reporting.charts import create_tipping_point_chart

        html = create_tipping_point_chart(_report_payload(), self._cfg())
        assert "Plotly.newPlot" in html
        assert "Strongest measured benchmark" in html

    def test_curve_chart_uses_the_transported_curve(self):
        from mmm_framework.reporting.charts import create_bias_curve_chart

        html = create_bias_curve_chart(_report_payload(), self._cfg())
        assert "Plotly.newPlot" in html
        assert "decision threshold" in html

    def test_contour_chart_renders(self):
        from mmm_framework.reporting.charts import create_bias_contour_chart

        surface = {
            "mu_grid": [-0.2, 0.0, 0.2],
            "sigma_grid": [0.1, 0.5],
            "prob": [[0.99, 0.97, 0.80], [0.80, 0.70, 0.50]],
            "threshold": 0.95,
            "scale": "fraction_of_mean",
        }
        html = create_bias_contour_chart(surface, self._cfg(), title="TV")
        assert "contour" in html and "Plotly.newPlot" in html

    def test_every_chart_returns_empty_rather_than_raising_on_no_data(self):
        from mmm_framework.reporting.charts import (
            create_bias_contour_chart,
            create_bias_curve_chart,
            create_tipping_point_chart,
        )

        cfg = self._cfg()
        assert create_tipping_point_chart({}, cfg) == ""
        assert create_bias_curve_chart({}, cfg) == ""
        assert create_bias_contour_chart({}, cfg) == ""

    def test_channel_names_are_escaped_in_charts(self):
        from mmm_framework.reporting.charts import create_tipping_point_chart

        html = create_tipping_point_chart(
            _report_payload(channel="<img src=x onerror=alert(1)>"), self._cfg()
        )
        assert "onerror=alert(1)>" not in html
        assert "&lt;img" in html


# =========================================================================== #
# the in-graph sweep (Phase 4)
# =========================================================================== #


class TestConfounderConstruction:
    """Fast: the construction is pure numpy, so nothing needs fitting."""

    def _model(self):
        from mmm_framework.config import InferenceMethod, ModelConfig
        from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
        from mmm_framework.synth import dgp

        sc = dgp.build("unobserved_confounding", seed=1, n_weeks=104)
        cfg = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            use_parametric_adstock=True,
        )
        # Unfitted is fine — the construction reads raw arrays only.
        return BayesianMMM(sc.panel(), cfg, TrendConfig(type=TrendType.LINEAR))

    def test_the_assumed_strength_is_actually_delivered(self):
        """`strength` has to mean what it says.

        Mixing at ``w = sqrt(strength)`` looks right and is not: the driver
        averages the targeted channels, so its correlation with any one of them
        is diluted and the delivered partial R^2 lands far below the assumption.
        Measured at 0.01-0.05 against an assumed 0.15 before the calibration.
        """
        from mmm_framework.validation.confounding_sensitivity import build_confounder

        mmm = self._model()
        for assumed in (0.05, 0.15, 0.30):
            _u, _lam, delivered = build_confounder(
                mmm, strength=assumed, random_seed=42
            )
            mean_delivered = float(np.mean(list(delivered.values())))
            assert mean_delivered == pytest.approx(assumed, abs=0.01)

    def test_it_is_orthogonal_to_everything_the_model_adjusts_for(self):
        """Otherwise it is not *unobserved* — and the trend would absorb it."""
        from mmm_framework.validation.confounding_sensitivity import (
            _adjustment_basis,
            build_confounder,
        )

        mmm = self._model()
        u, _lam, _d = build_confounder(mmm, strength=0.2, random_seed=42)
        basis = _adjustment_basis(mmm)
        projection = basis @ np.linalg.lstsq(basis, u, rcond=None)[0]
        assert np.abs(projection).max() < 1e-8 * max(np.abs(u).max(), 1.0)

    def test_it_lives_on_the_period_axis(self):
        from mmm_framework.validation.confounding_sensitivity import build_confounder

        mmm = self._model()
        u, _lam, _d = build_confounder(mmm, strength=0.2, random_seed=42)
        assert u.shape == (mmm.n_periods,)

    def test_the_sign_flips_the_loading_not_the_series(self):
        from mmm_framework.validation.confounding_sensitivity import build_confounder

        mmm = self._model()
        pos, lam_p, _ = build_confounder(mmm, strength=0.2, sign=1, random_seed=42)
        neg, lam_n, _ = build_confounder(mmm, strength=0.2, sign=-1, random_seed=42)
        assert lam_p == pytest.approx(-lam_n)
        assert np.allclose(pos, -neg)

    def test_zero_strength_contributes_nothing(self):
        """At strength 0 the outcome loading is 0, so the scaled series vanishes.

        Note what is NOT asserted: that the underlying series is uncorrelated with
        every channel. It is uncorrelated *in expectation*, but an AR(1) series
        against ~100 periods of AR-ish spend residual has a wide sampling
        distribution for that correlation (Display lands at r^2 = 0.14 on this
        seed). That series is never used — the loading zeroes it, and the sweep
        detaches the term entirely at strength 0 — so the honest claim is about
        the contribution, not the correlation. The *mean* delivered strength is
        calibrated, and is pinned by the neighbouring test.
        """
        from mmm_framework.validation.confounding_sensitivity import build_confounder

        mmm = self._model()
        u_scaled, lam, delivered = build_confounder(mmm, strength=0.0, random_seed=42)
        assert lam == pytest.approx(0.0)
        assert np.allclose(u_scaled, 0.0)
        assert float(np.mean(list(delivered.values()))) < 0.08

    def test_construction_is_deterministic(self):
        from mmm_framework.validation.confounding_sensitivity import build_confounder

        mmm = self._model()
        a, _, _ = build_confounder(mmm, strength=0.2, random_seed=7)
        b, _, _ = build_confounder(mmm, strength=0.2, random_seed=7)
        assert np.array_equal(a, b)


class TestLatentConfounderGraph:
    def _model(self):
        from mmm_framework.config import InferenceMethod, ModelConfig
        from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
        from mmm_framework.synth import dgp

        sc = dgp.build("clean", seed=3, n_weeks=60)
        cfg = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            use_parametric_adstock=True,
        )
        return BayesianMMM(sc.panel(), cfg, TrendConfig(type=TrendType.LINEAR))

    def test_the_default_graph_is_byte_identical(self):
        """Absent, not zero — the discipline every optional block here follows."""
        plain = self._model()
        assert "latent_confounder" not in set(plain.model.named_vars)
        assert "latent_confounder_component" not in set(plain.model.named_vars)

    def test_attaching_adds_exactly_two_named_variables(self):
        plain = self._model()
        base_vars = set(plain.model.named_vars)

        attached = self._model()
        attached.add_latent_confounder(np.linspace(-0.1, 0.1, attached.n_periods))
        new_vars = set(attached.model.named_vars)
        assert new_vars - base_vars == {
            "latent_confounder",
            "latent_confounder_component",
        }

    def test_detaching_restores_the_default_graph(self):
        mmm = self._model()
        mmm.add_latent_confounder(np.zeros(mmm.n_periods))
        assert "latent_confounder" in set(mmm.model.named_vars)
        mmm.add_latent_confounder(None)
        assert "latent_confounder" not in set(mmm.model.named_vars)

    def test_a_wrong_length_vector_is_refused_by_name(self):
        mmm = self._model()
        mmm.add_latent_confounder(np.zeros(mmm.n_periods + 5))
        with pytest.raises(ValueError, match="period axis"):
            _ = mmm.model

    def test_the_confounder_is_not_folded_into_media(self):
        """It is a baseline demand driver, not a media stream.

        Folding it into ``channel_contributions`` would double-count it in every
        ROI and estimand; adding it to ``X_controls`` would change the control
        coord and the shape of ``control_contributions``.
        """
        mmm = self._model()
        n_controls_before = mmm.n_controls
        mmm.add_latent_confounder(np.linspace(-0.1, 0.1, mmm.n_periods))
        model = mmm.model
        assert mmm.n_controls == n_controls_before
        contrib = model["channel_contributions"]
        assert "latent_confounder" not in [
            v.name for v in contrib.owner.inputs if getattr(v, "name", None)
        ]


@pytest.mark.slow
class TestConfounderSweep:
    def test_the_sweep_moves_coefficients_a_reweighting_cannot(self, confounded_fit):
        """The whole justification for paying for refits.

        A post-hoc convolution shifts and widens an estimand uniformly. Here the
        confounder is in the graph and competes for signal, so channels move by
        DIFFERENT amounts — and some can move up while others move down.
        """
        from mmm_framework.validation.confounding_sensitivity import (
            ConfounderSweepConfig,
            run_confounder_sweep,
        )

        _sc, mmm = confounded_fit
        res = run_confounder_sweep(
            mmm,
            ConfounderSweepConfig(
                strengths=(0.0, 0.25), signs=(1,), method="map", max_draws=80
            ),
        )
        assert res.status == "ok", res.reason
        zero = next(p for p in res.points if p.strength == 0.0)
        loaded = next(p for p in res.points if p.strength == 0.25)
        deltas = {
            ch: loaded.roi[ch] - zero.roi[ch] for ch in zero.roi if ch in loaded.roi
        }
        assert len(deltas) >= 3
        spread = max(deltas.values()) - min(deltas.values())
        assert (
            spread > 0.1
        ), f"channels moved uniformly, so nothing was re-fitted: {deltas}"

    def test_the_zero_point_reproduces_the_unconfounded_fit(self, confounded_fit):
        from mmm_framework.validation.confounding_sensitivity import (
            ConfounderSweepConfig,
            run_confounder_sweep,
        )

        _sc, mmm = confounded_fit
        res = run_confounder_sweep(
            mmm,
            ConfounderSweepConfig(
                strengths=(0.0,), signs=(1,), method="map", max_draws=80
            ),
        )
        zero = res.points[0]
        for ch, value in res.baseline_roi.items():
            assert zero.roi[ch] == pytest.approx(value, rel=1e-9)

    def test_delivered_strength_is_reported_so_absorption_is_visible(
        self, confounded_fit
    ):
        from mmm_framework.validation.confounding_sensitivity import (
            ConfounderSweepConfig,
            run_confounder_sweep,
        )

        _sc, mmm = confounded_fit
        res = run_confounder_sweep(
            mmm,
            ConfounderSweepConfig(
                strengths=(0.20,), signs=(1,), method="map", max_draws=80
            ),
        )
        point = res.points[0]
        assert point.delivered_r2_t
        assert float(np.mean(list(point.delivered_r2_t.values()))) == pytest.approx(
            0.20, abs=0.02
        )

    def test_to_dict_is_json_safe(self, confounded_fit):
        from mmm_framework.validation.confounding_sensitivity import (
            ConfounderSweepConfig,
            run_confounder_sweep,
        )

        _sc, mmm = confounded_fit
        res = run_confounder_sweep(
            mmm,
            ConfounderSweepConfig(
                strengths=(0.0, 0.15), signs=(1,), method="map", max_draws=80
            ),
        )
        json.dumps(res.to_dict())
        assert len(res.summary()) == len(res.baseline_roi)
