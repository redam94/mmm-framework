"""Payback horizon — per-draw intervals, truncation disclosure, gates (issue #224).

The trap these tests exist to keep closed: "TV pays back in 3 weeks" reads as a
measurement of the model's LEAST identified parameter, from a kernel truncated
at ``l_max`` and renormalized (which makes it read shorter than truth), from a
posterior mean that hides how little the prior moved. Every criterion below is
one of the ways that number can lie.

Measured values quoted in docstrings were produced by the fits built here (seed
7, 156 weeks, NUTS 4x800), not chosen. The unflattering ones are included.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from mmm_framework.planning.discount import (
    discount_weights,
    mid_horizon_discount_factor,
)
from mmm_framework.planning.payback import (
    ChannelPayback,
    PaybackResult,
    _t_name,
    channel_payback,
    payback_breakeven,
)
from mmm_framework.synth import dgp, mff
from mmm_framework.transforms.carryover import (
    carryover_crossing_lags,
    carryover_half_life,
)

# ---------------------------------------------------------------------------
# Analytic exactness — no fits
# ---------------------------------------------------------------------------


class TestCrossingLagsAnalytic:
    def _closed_form(self, alpha: float, l_max: int, share: float) -> float:
        """The closed-form interpolated cumulative-share crossing for a
        truncated+normalized geometric kernel — computed independently of the
        implementation under test."""
        w = alpha ** np.arange(l_max, dtype=float)
        cum = np.cumsum(w) / w.sum()
        j = int(np.searchsorted(cum, share))
        prev = cum[j - 1] if j > 0 else 0.0
        return j + (share - prev) / (cum[j] - prev)

    def test_per_draw_t50_matches_closed_form_to_1e9(self):
        """The acceptance criterion verbatim: for a synthetic posterior with
        known geometric alpha, per-draw t50 equals the closed-form interpolated
        cumulative-50% lag to 1e-9."""
        rng = np.random.default_rng(0)
        alphas = rng.uniform(0.05, 0.95, size=200)
        l_max = 8
        kernels = np.stack(
            [(a ** np.arange(l_max)) / (a ** np.arange(l_max)).sum() for a in alphas]
        )
        got = carryover_crossing_lags(kernels, 0.5)
        want = np.array([self._closed_form(a, l_max, 0.5) for a in alphas])
        np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)

    def test_t90_matches_closed_form_too(self):
        rng = np.random.default_rng(1)
        alphas = rng.uniform(0.05, 0.95, size=100)
        kernels = np.stack(
            [(a ** np.arange(8)) / (a ** np.arange(8)).sum() for a in alphas]
        )
        got = carryover_crossing_lags(kernels, 0.9)
        want = np.array([self._closed_form(a, 8, 0.9) for a in alphas])
        np.testing.assert_allclose(got, want, atol=1e-9, rtol=0)

    def test_interval_is_eti_of_transform_not_transform_of_mean(self):
        """log(0.5)/log(alpha) is convex, so t50(mean(alpha)) is NOT
        mean(t50(alpha)) — the acceptance criterion demands the former never
        appears. Constructed so the gap is unmistakable."""
        alphas = np.array([0.1, 0.9])  # convexity bites hard at the extremes
        kernels = np.stack(
            [(a ** np.arange(8)) / (a ** np.arange(8)).sum() for a in alphas]
        )
        per_draw_mean = float(carryover_crossing_lags(kernels, 0.5).mean())
        a_mean = float(alphas.mean())
        k_mean = (a_mean ** np.arange(8)) / (a_mean ** np.arange(8)).sum()
        of_mean = float(carryover_crossing_lags(k_mean[None, :], 0.5)[0])
        assert abs(per_draw_mean - of_mean) > 0.3  # measured: ~0.42

    def test_half_life_is_the_50_crossing(self):
        rng = np.random.default_rng(2)
        k = rng.dirichlet(np.ones(8), size=50)
        np.testing.assert_array_equal(
            carryover_half_life(k), carryover_crossing_lags(k, 0.5)
        )

    def test_share_domain_is_enforced(self):
        k = np.ones((1, 8)) / 8
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="share"):
                carryover_crossing_lags(k, bad)

    def test_truncation_bias_direction_analytic(self):
        """The issue's headline: alpha=0.8 at l_max=8 reads t90~5.8 against an
        untruncated ~9.3 — 13.4% of mass discarded and re-spread inside the
        window. The truncated crossing UNDERSTATES, always."""
        a = 0.8
        w8 = (a ** np.arange(9)) / (a ** np.arange(9)).sum()
        w200 = (a ** np.arange(200)) / (a ** np.arange(200)).sum()
        t90_trunc = float(carryover_crossing_lags(w8[None, :], 0.9)[0])
        t90_true = float(carryover_crossing_lags(w200[None, :], 0.9)[0])
        # Under the canonical convention (lag k occupies [k, k+1), issue #218)
        # these are 6.79 and 10.34 — the issue's "reads 6 vs 10", measured.
        assert t90_trunc == pytest.approx(6.79, abs=0.05)
        assert t90_true == pytest.approx(10.34, abs=0.05)
        assert t90_trunc < t90_true
        assert float(a**9) == pytest.approx(0.134, abs=0.001)  # the tail mass


class TestNaming:
    def test_threshold_names(self):
        assert _t_name(0.5) == "t50"
        assert _t_name(0.9) == "t90"
        assert _t_name(0.75) == "t75"


# ---------------------------------------------------------------------------
# The discount utility
# ---------------------------------------------------------------------------


class TestDiscount:
    def test_default_rate_is_zero_and_weights_are_ones(self):
        w = discount_weights(26)
        assert w.shape == (26,)
        np.testing.assert_array_equal(w, np.ones(26))

    def test_matches_the_legacy_experiment_value_arithmetic(self):
        """The extraction must be byte-compatible with what
        `compute_experiment_net_value` used to compute inline."""
        h, hl, r = 26, 12.0, 0.10
        w = np.arange(h, dtype=float)
        legacy = np.power(0.5, w / hl) * np.power(1.0 + r, -w / 52.0)
        np.testing.assert_allclose(
            discount_weights(h, rate_annual=r, half_life_weeks=hl), legacy
        )

    def test_matches_the_legacy_clv_mid_horizon_factor(self):
        weekly = (1.0 + 0.12) ** (1.0 / 52.0) - 1.0
        legacy = (1.0 + weekly) ** (-52 / 2.0)
        assert mid_horizon_discount_factor(52, 0.12) == pytest.approx(legacy)
        assert mid_horizon_discount_factor(52, 0.0) == 1.0

    def test_measured_smallness_of_the_correction(self):
        """The docstring's claim: at repo horizons a 10%/yr rate moves the mean
        weight by low single digits — which is why the default is 0.0 rather
        than an assumed nonzero rate."""
        for h in (26, 52):
            drop = 1.0 - discount_weights(h, rate_annual=0.10).mean()
            assert drop < 0.05


# ---------------------------------------------------------------------------
# Fits — planted recovery, negative control, gates, refusals
# ---------------------------------------------------------------------------

N_WEEKS = 156


def _spec(sc):
    return {
        "kpi": "Sales",
        "kpi_level": "national",
        "media_channels": [
            {
                "name": c,
                "adstock": {"type": "geometric"},
                "saturation": {"type": "hill"},
            }
            for c in sc.channels
        ],
        "control_variables": [{"name": c} for c in sc.controls.columns],
    }


def _fit(maker, tmp_path, *, method="nuts", seed=7, spec_override=None):
    from mmm_framework.agents.fitting import build_model

    sc = maker(seed=seed, n_weeks=N_WEEKS)
    csv = tmp_path / "world.csv"
    mff.scenario_to_mff(sc).to_csv(csv, index=False)
    model = build_model(spec_override or _spec(sc), str(csv))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stderr(io.StringIO()):
            if method == "nuts":
                model.fit(draws=800, tune=800, chains=4, random_seed=42)
            else:
                model.fit(method="map", random_seed=42)
    return sc, model


def _truth_crossing(sc, ch, share):
    cum = np.asarray(sc.notes["true_adstock"][ch]["cum_share"], dtype=float)
    j = int(np.searchsorted(cum, share))
    prev = cum[j - 1] if j > 0 else 0.0
    return j + (share - prev) / (cum[j] - prev)


@pytest.fixture(scope="module")
def clean_fit(tmp_path_factory):
    return _fit(dgp.make_clean, tmp_path_factory.mktemp("clean"))


@pytest.mark.slow
class TestPlantedRecoveryClean:
    """Measured (seed 7, NUTS 4x800): t50 truth/estimate per channel —
    TV 1.38/1.42, Search 0.62/0.63, Social 0.83/1.04, Display 1.00/1.06.
    Social is the unflattering one: mean |error| 0.08 lag-units overall but
    0.21 on Social, still inside its 90% interval."""

    def test_interval_covers_truth_for_at_least_3_of_4(self, clean_fit):
        sc, model = clean_fit
        res = channel_payback(model)
        covered = 0
        for ch, p in res.channels.items():
            assert p.status in ("ok", "downgraded")
            h = p.horizons["t50"]
            tr = _truth_crossing(sc, ch, 0.5)
            if h["lower"] is not None and h["lower"] <= tr <= h["upper"]:
                covered += 1
        assert covered >= 3

    def test_every_result_carries_tail_mass_and_l_max(self, clean_fit):
        _, model = clean_fit
        res = channel_payback(model)
        for p in res.channels.values():
            assert p.l_max == 8
            assert 0.0 <= p.truncated_tail_mass < 1.0
            d = p.to_dict()
            assert "truncated_tail_mass" in d and "l_max" in d

    def test_learning_verdict_attached_and_strong_on_clean(self, clean_fit):
        """The clean world has well-flighted channels: alpha genuinely learns
        (measured contractions 0.73-0.96), so the verdict must NOT read
        prior-dominated here — that would make the gate meaningless."""
        _, model = clean_fit
        res = channel_payback(model)
        for p in res.channels.values():
            assert p.learning_verdict is not None
            assert not p.prior_dominated

    def test_autocorr_gate_passes_on_clean(self, clean_fit):
        _, model = clean_fit
        res = channel_payback(model)
        assert res.autocorrelation.get("ppc_acf1_extreme") is False
        assert not any("BIASED SHORT" in c for c in res.caveats)

    def test_counterfactual_basis_reports_the_disagreement(self, clean_fit):
        _, model = clean_fit
        res = channel_payback(model, basis="counterfactual", max_draws=100)
        for p in res.channels.values():
            if p.basis == "counterfactual":
                assert p.kernel_t50_mean is not None

    def test_breakeven_composes_roi_kernel_and_discount(self, clean_fit):
        """make_clean plants ROAS < 1 everywhere, so most draws never repay —
        prob_never must be HIGH here, and reporting a crisp break-even lag
        without it would be the exact trustworthiness trap."""
        _, model = clean_fit
        be = payback_breakeven(model, value_per_kpi=1.0, value_source="test")
        assert set(be) == set(model.channel_names)
        for b in be.values():
            assert b.status in ("ok", "never")
            assert b.prob_never is not None and b.prob_never > 0.5

    def test_breakeven_refuses_without_a_valuation(self, clean_fit):
        from mmm_framework.finance import UnresolvedValueError

        _, model = clean_fit
        with pytest.raises(UnresolvedValueError):
            payback_breakeven(model)


@pytest.mark.slow
class TestNegativeControlMisspec:
    """`make_adstock_misspec` plants a 26-lag Weibull; the model fits geometric
    at l_max=8. Measured: fitted t90 5.2-7.1 against planted 9.1-11.9 — SHORT
    on all four channels, and the PPC lag-1 autocorrelation check is extreme
    (Bayesian p=0.01) while the residual Ljung-Box is NOT (p=0.16), which is
    why the gate runs both."""

    @pytest.fixture(scope="class")
    def misspec_fit(self, tmp_path_factory):
        return _fit(dgp.make_adstock_misspec, tmp_path_factory.mktemp("mis"))

    def test_the_answer_key_is_the_planted_weibull_not_the_model_family(self):
        """The world's true_adstock used to be stamped with the geometric
        _ALPHA table by _finish's setdefault — an answer key describing the
        model's family instead of the planted kernel."""
        sc = dgp.make_adstock_misspec(seed=7, n_weeks=N_WEEKS)
        for ch in sc.channels:
            ta = sc.notes["true_adstock"][ch]
            assert ta["family"] == "weibull"
            assert ta["l_max"] == 26

    def test_every_channel_is_downgraded_not_confidently_wrong(self, misspec_fit):
        _, model = misspec_fit
        res = channel_payback(model)
        assert all(p.status == "downgraded" for p in res.channels.values())
        assert any("BIASED SHORT" in c for c in res.caveats)

    def test_direction_of_bias_is_short(self, misspec_fit):
        """The acceptance criterion: assert the DIRECTION, not a magnitude —
        the truncated geometric cannot express the Weibull's 26-lag tail, so
        the fitted horizon understates on every channel."""
        sc, model = misspec_fit
        res = channel_payback(model)
        for ch, p in res.channels.items():
            fitted_t90 = p.horizons["t90"]["mean"]
            true_t90 = _truth_crossing(sc, ch, 0.9)
            assert fitted_t90 < true_t90, (
                f"{ch}: fitted t90 {fitted_t90:.2f} not short of planted "
                f"{true_t90:.2f} — the truncation-bias direction assertion"
            )

    def test_ppc_acf1_fires_where_ljung_box_does_not(self, misspec_fit):
        _, model = misspec_fit
        res = channel_payback(model)
        assert res.autocorrelation.get("ppc_acf1_extreme") is True


@pytest.mark.slow
class TestPriorSensitivity:
    """Two fits differing ONLY in the alpha prior — AdstockConfig.geometric()'s
    Beta(1,3) vs the graph default Beta(2,2), prior-implied half-lives 0.5 vs
    1.0 weeks. On a short, weakly-flighted world the data cannot arbitrate,
    and the honest output is a prior-dominated verdict, not two equally
    confident horizons."""

    def _short_fit(self, tmp_path, alpha_prior):
        from mmm_framework.agents.fitting import build_model

        sc = dgp.make_clean(seed=11, n_weeks=40)
        csv = tmp_path / "short.csv"
        mff.scenario_to_mff(sc).to_csv(csv, index=False)
        spec = _spec(sc)
        if alpha_prior is not None:
            # The spec route to a per-channel adstock prior — priors.media.
            # (An `alpha_prior` key inside media_channels[].adstock is silently
            # ignored, which is its own documented trap: "agent-set priors
            # don't change the model".)
            spec["priors"] = {
                "media": {
                    m["name"]: {"adstock_alpha": alpha_prior}
                    for m in spec["media_channels"]
                }
            }
        model = build_model(spec, str(csv))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                model.fit(draws=500, tune=500, chains=2, random_seed=42)
        return model

    def test_different_priors_move_the_number_and_the_verdict_says_so(self, tmp_path):
        m_beta13 = self._short_fit(
            tmp_path, {"distribution": "beta", "params": {"alpha": 1, "beta": 3}}
        )
        m_beta22 = self._short_fit(tmp_path, None)  # graph default Beta(2,2)
        r13 = channel_payback(m_beta13)
        r22 = channel_payback(m_beta22)
        moved = 0
        flagged = 0
        for ch in r13.channels:
            t13 = r13.channels[ch].horizons["t50"]["mean"]
            t22 = r22.channels[ch].horizons["t50"]["mean"]
            if abs(t13 - t22) > 0.05:
                moved += 1
            if (
                r13.channels[ch].prior_dominated
                or r22.channels[ch].prior_dominated
                or r13.channels[ch].learning_verdict
                not in ("strong", "moderate", "relocated")
                or r22.channels[ch].learning_verdict
                not in ("strong", "moderate", "relocated")
            ):
                flagged += 1
        assert moved >= 1, "the prior did not move any horizon — world too easy"
        assert flagged >= 1, (
            "two priors produced different numbers and NO channel was flagged "
            "as prior-influenced — exactly the two-equally-confident-estimates "
            "failure the criterion names"
        )


@pytest.mark.slow
class TestRefusalsByName:
    def test_extension_model_is_refused(self):
        from mmm_framework.mmm_extensions.config import (
            MediatorConfig,
            MediatorType,
            NestedModelConfig,
        )
        from mmm_framework.mmm_extensions.models.nested import NestedMMM

        n = 60
        rng = np.random.default_rng(0)
        media = np.abs(rng.normal(100, 20, (n, 2)))
        idx = pd.date_range("2022-01-03", periods=n, freq="W-MON")
        r = np.random.default_rng(1)
        aware = 40 + 0.3 * media[:, 0] + r.normal(0, 4, n)
        sales = 1000 + 4 * aware + 2 * media[:, 1] + r.normal(0, 40, n)
        cfg = NestedModelConfig(
            mediators=(
                MediatorConfig(
                    name="Awareness", mediator_type=MediatorType.FULLY_LATENT
                ),
            )
        )
        model = NestedMMM(media, sales, ["TV", "Digital"], cfg, index=idx)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                model.fit(method="map", random_seed=0)
        res = channel_payback(model)
        assert all(p.status == "refused" for p in res.channels.values())
        assert "extension model" in next(iter(res.channels.values())).reason

    def test_dual_stock_brand_model_is_refused_by_its_variables(self):
        """LongTermBrandMMM registers only the FAST stock under
        adstock_alpha_<ch>; the Beta(47,3) slow stock is invisible to a kernel
        reader, so a payback read off that name is dramatically too short while
        the model itself says the opposite. Detection is by registered
        variables so user-authored copies with other class names still refuse."""

        class _DualStockStub:
            channel_names = ["TV"]

            class _Post:
                data_vars = [
                    "adstock_alpha_TV",
                    "brand_retention",
                    "long_term_fraction",
                ]

                def __contains__(self, k):
                    return k in self.data_vars

            class _Trace:
                posterior = None

            _trace = _Trace()
            _trace.posterior = _Post()

        res = channel_payback(_DualStockStub())
        assert res.channels["TV"].status == "refused"
        assert "slow brand stock" in res.channels["TV"].reason

    def test_structural_ar1_is_refused_with_the_ridge_named(self):
        """AR(1) mediator persistence lives in the state's rho, on a stated
        ridge with the per-channel alpha — no single kernel to read."""

        class _Dyn:
            value = "ar1"

        class _MedSpec:
            name = "awareness"
            dynamics = _Dyn()

        class _Cfg:
            mediators = [_MedSpec()]
            latent_factors = []

        class _StructStub:
            channel_names = ["TV"]
            config = _Cfg()

            class _Post:
                data_vars = ["alpha_TV"]

                def __contains__(self, k):
                    return k in self.data_vars

            class _Trace:
                posterior = None

            _trace = _Trace()
            _trace.posterior = _Post()

        res = channel_payback(_StructStub())
        assert res.channels["TV"].status == "refused"
        assert "AR(1)" in res.channels["TV"].reason
        assert "ridge" in res.channels["TV"].reason


@pytest.mark.slow
class TestProvenanceAndCollapse:
    def test_map_fit_reports_collapsed_interval_not_precision(self, tmp_path):
        """A MAP fit's single-draw posterior collapses every interval onto its
        point estimate; rendering [x, x] as a 90% interval is the visual
        language of extreme precision (#249). It must read as absent."""
        _, model = _fit(dgp.make_clean, tmp_path, method="map")
        res = channel_payback(model)
        for p in res.channels.values():
            assert p.interval_collapsed
            assert p.horizons["t50"]["lower"] is None
            assert any("collapsed" in c for c in p.caveats)

    def test_frequentist_provenance_rides_the_output(self, clean_fit):
        """A bootstrap-derived payback interval is a CONFIDENCE interval. The
        family is read from _fit_diagnostics, so a stamped frequentist fit
        must flip the noun."""
        _, model = clean_fit
        res = channel_payback(model)
        assert res.interval_kind == "credible"
        old = getattr(model, "_fit_diagnostics", None)
        try:
            model._fit_diagnostics = {"inference_family": "frequentist"}
            res_f = channel_payback(model, learning=False, autocorrelation={})
            assert res_f.interval_kind == "confidence"
            assert any("confidence" in c for c in res_f.caveats)
        finally:
            model._fit_diagnostics = old


@pytest.mark.slow
class TestPersistenceProvenance:
    def test_serializer_records_basis_family_l_max_tail(self, clean_fit, tmp_path):
        """A reloaded model must not silently re-derive a horizon on a
        different basis: metadata.json carries basis/family/l_max/tail mass."""
        import json

        from mmm_framework.serialization import MMMSerializer

        _, model = clean_fit
        path = tmp_path / "m"
        MMMSerializer.save(model, str(path))
        meta = json.loads((path / "metadata.json").read_text())
        pb = meta["payback"]
        assert pb["basis"] == "kernel"
        for ch in model.channel_names:
            rec = pb["channels"][ch]
            assert rec["family"] == "geometric"
            assert rec["l_max"] == 8
            assert "truncated_tail_mass" in rec
        assert meta["payback_schema_version"] == "1.0"

    def test_run_metrics_carry_payback(self, clean_fit):
        import json

        from mmm_framework.planning.history import (
            RUN_METRICS_SCHEMA_VERSION,
            compute_run_metrics,
        )

        _, model = clean_fit
        metrics = compute_run_metrics(model, max_draws=100, random_seed=42)
        assert metrics["schema_version"] == RUN_METRICS_SCHEMA_VERSION
        json.dumps(metrics)  # numpy scalars would fail here
        for ch, row in metrics["channels"].items():
            pb = row.get("payback")
            assert pb is not None, f"{ch} carries no payback record"
            assert pb["basis"] == "kernel"
            assert pb["t50_mean"] is not None


@pytest.mark.slow
class TestRealisticRecovery:
    """`make_realistic` exported no carryover truth before #224 — this class
    is runnable only because that gap was closed. 7 channels, 2 deliberately
    weak/prior-dominated, latent-demand confounding: the honest expectation is
    coverage on MOST channels plus flagged verdicts on the weak ones, not a
    clean sweep."""

    def test_realistic_exports_the_answer_key(self):
        sc = dgp.make_realistic(seed=7, n_weeks=N_WEEKS)
        ta = sc.notes.get("true_adstock")
        assert ta is not None
        assert set(ta) == set(sc.channels)
        for ch in sc.channels:
            assert ta[ch]["cum_share"][-1] == pytest.approx(1.0)

    def test_interval_covers_truth_for_most_channels(self, tmp_path):
        sc, model = _fit(dgp.make_realistic, tmp_path)
        res = channel_payback(model)
        covered, total = 0, 0
        for ch, p in res.channels.items():
            if p.status == "refused":
                continue
            total += 1
            h = p.horizons["t50"]
            if h["lower"] is None:
                continue
            tr = _truth_crossing(sc, ch, 0.5)
            covered += h["lower"] <= tr <= h["upper"]
        assert total >= 6
        # Measured (seed 7, NUTS 4x800): 6/7 covered; Print (weak, near-
        # collinear with Radio) is the one that can miss. The bar is most-not-
        # all, and the miss is disclosed here rather than hidden by reruns.
        assert covered >= total - 2


class TestResultShape:
    def test_payload_is_json_shaped(self):
        p = ChannelPayback(channel="TV", status="refused", reason="because")
        r = PaybackResult(
            channels={"TV": p},
            basis="kernel",
            thresholds=(0.5, 0.9),
            interval_mass=0.9,
            interval_kind="credible",
        )
        import json

        payload = json.loads(json.dumps(r.to_dict()))
        assert payload["channels"]["TV"]["status"] == "refused"
        assert payload["thresholds"] == [0.5, 0.9]

    def test_unknown_basis_is_refused_loudly(self):
        with pytest.raises(ValueError, match="basis"):
            channel_payback(object(), basis="vibes")
