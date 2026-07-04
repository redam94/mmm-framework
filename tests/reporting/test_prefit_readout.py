"""Tests for the pre-fit "Model Design Readout" (pre-registration document).

Covers the three new modules end-to-end without any MCMC:

- ``reporting/charts/prior.py`` — the pre-fit chart kit on pure array inputs
  (prior-predictive fan, replicate-stat histograms, prior densities, implied
  response bands, SBC rank-histogram / ECDF-diff, including their empty-input
  degradations).
- ``reporting/prefit.py`` — templated + LLM-enriched insights
  (:func:`build_prefit_insights`), the :class:`PrefitReadoutGenerator` shell
  rendered from canned facts (masthead, nav, change record, SBC verdict table,
  HTML-escaping), and section gating (missing SBC / missing curves).
- ``reporting/helpers/prefit.py`` — prior enumeration, assumptions,
  prior-predictive facts and implied response curves computed from a real
  *unfitted* :class:`BayesianMMM` (prior sampling only, no fit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    ControlVariableConfig,
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM
from mmm_framework.model.trend_config import TrendConfig, TrendType
from mmm_framework.reporting.charts.prior import (
    create_prior_adstock_band,
    create_prior_density_chart,
    create_prior_predictive_fan,
    create_prior_saturation_band,
    create_prior_stat_distribution,
    create_sbc_ecdf_diff,
    create_sbc_rank_histogram,
)
from mmm_framework.reporting.config import ReportConfig
from mmm_framework.reporting.helpers.prefit import (
    enumerate_model_priors,
    model_assumptions,
)
from mmm_framework.reporting.prefit import (
    PREFIT_INSIGHT_SLOTS,
    PrefitReadoutGenerator,
    build_prefit_insights,
    prefit_facts,
)

# All nine section ids the full readout renders, in document order.
_SECTION_IDS = (
    "purpose",
    "spec",
    "assumptions",
    "priors",
    "response",
    "prior-predictive",
    "sbc",
    "revisions",
    "signoff",
)

_XSS = "<script>alert(1)</script>"


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def report_config() -> ReportConfig:
    return ReportConfig()


@pytest.fixture(scope="module")
def simple_panel() -> PanelDataset:
    """52-week national panel with 2 channels + 1 control (test_approx_fit twin)."""
    periods = pd.date_range("2020-01-06", periods=52, freq="W-MON")
    n = len(periods)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=["Price"],
    )
    rng = np.random.default_rng(42)
    y = pd.Series(1000 + rng.standard_normal(n) * 100, name="Sales")
    X_media = pd.DataFrame(
        {
            "TV": np.abs(rng.standard_normal(n) * 50 + 100),
            "Digital": np.abs(rng.standard_normal(n) * 30 + 80),
        }
    )
    X_controls = pd.DataFrame({"Price": 10 + rng.standard_normal(n) * 0.5})
    cfg = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    return PanelDataset(
        y=y,
        X_media=X_media,
        X_controls=X_controls,
        coords=coords,
        index=periods,
        config=cfg,
    )


@pytest.fixture(scope="module")
def unfitted_model(simple_panel) -> BayesianMMM:
    """A configured-but-NOT-fitted BayesianMMM (the readout's target state)."""
    model_config = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_chains=1,
        n_draws=100,
        n_tune=100,
        target_accept=0.8,
    )
    return BayesianMMM(simple_panel, model_config, TrendConfig(type=TrendType.LINEAR))


@pytest.fixture(scope="module")
def real_facts(unfitted_model) -> dict:
    """prefit_facts from the real unfitted model — prior sampling only, shared."""
    return prefit_facts(unfitted_model, n_prior_samples=100)


# ─────────────────────────────────────────────────────────────────────────────
# Canned-facts builders (no model)
# ─────────────────────────────────────────────────────────────────────────────
def _canned_ppc(n: int = 20) -> dict:
    rng = np.random.default_rng(0)
    observed = 1000 + rng.normal(0, 50, n)
    return {
        "kpi_label": "Sales",
        "dates": [f"2024-{1 + i // 4:02d}-{1 + (i % 4) * 7:02d}" for i in range(n)],
        "observed": observed,
        "bands": {
            "p05": observed - 400,
            "p25": observed - 150,
            "p50": observed + rng.normal(0, 10, n),
            "p75": observed + 150,
            "p95": observed + 400,
        },
        "coverage_90": 0.9,
        "frac_negative": 0.01,
        "rep_means": 1000 + rng.normal(0, 120, 200),
        "rep_sds": np.abs(rng.normal(90, 30, 200)),
        "obs_mean": 1000.0,
        "obs_sd": 50.0,
        "n_draws": 200,
        "var_name": "y_obs",
    }


def _canned_curve() -> dict:
    x = np.linspace(0.0, 1.0, 40)
    med = x / (x + 0.3)
    lags = np.arange(9)
    w = 0.6**lags
    w = w / w.sum()
    return {
        "sat_family": "logistic",
        "adstock_family": "geometric",
        "saturation": {"x": x, "median": med, "lower": med * 0.6, "upper": med * 1.3},
        "adstock": {"lags": lags, "median": w, "lower": w * 0.5, "upper": w * 1.4},
    }


def _sbc_param(
    name: str = "beta_TV",
    calibrated: bool = True,
    with_ranks: bool = False,
) -> dict:
    rng = np.random.default_rng(7)
    n_sims, n_bins, L = 40, 10, 100
    counts = rng.multinomial(n_sims, np.ones(n_bins) / n_bins)
    out = {
        "name": name,
        "L": L,
        "n_sims": n_sims,
        "n_bins": n_bins,
        "bin_counts": [int(c) for c in counts],
        "chi2_stat": 4.2,
        "chi2_pvalue": 0.02 if not calibrated else 0.61,
        "shape": "∪ overconfident" if not calibrated else "flat",
        "bias_z": 0.4,
        "dispersion_z": -0.2,
        "miscalibration": 0.8 if not calibrated else 0.05,
        "calibrated": calibrated,
    }
    if with_ranks:
        out["int_ranks"] = [int(r) for r in rng.integers(0, L + 1, n_sims)]
    return out


def _canned_sbc(calibrated: bool = True, flagged_name: str = "beta_TV") -> dict:
    return {
        "all_calibrated": calibrated,
        "n_sims_requested": 40,
        "n_sims_effective": 40,
        "L": 100,
        "n_bins": 10,
        "sampler": "nuts",
        "params": [
            _sbc_param(flagged_name, calibrated=calibrated, with_ranks=True),
            _sbc_param("sigma", calibrated=True),
        ],
        "caveats": ["Reduced-data SBC: 40 sims is a smoke check, not a verdict."],
    }


def _canned_facts(
    *,
    ppc: dict | str | None = "default",
    curves: dict | str | None = "default",
    sbc: dict | None = None,
    revisions: list[dict] | None = None,
    assumption_channels: list[str] | None = None,
) -> dict:
    """Minimal facts dict matching prefit_facts' return shape."""
    rng = np.random.default_rng(3)
    priors = [
        {
            "group": "Media effects",
            "name": "beta_TV",
            "family": "Gamma",
            "dims": "",
            "mean": 1.5,
            "sd": 1.0,
            "lower": 0.2,
            "upper": 3.6,
            "calibrated": True,
        },
        {
            "group": "Carryover (adstock)",
            "name": "adstock_alpha_TV",
            "family": "Beta",
            "dims": "",
            "mean": 0.4,
            "sd": 0.2,
            "lower": 0.05,
            "upper": 0.8,
            "calibrated": False,
        },
        {
            "group": "Observation noise",
            "name": "sigma",
            "family": "Half-Normal",
            "dims": "",
            "mean": 0.5,
            "sd": 0.3,
            "lower": 0.05,
            "upper": 1.1,
            "calibrated": False,
        },
    ]
    assumptions = [
        {
            "topic": "Observation model",
            "setting": "normal likelihood, identity link",
            "detail": "The KPI is modeled as this distribution.",
            "channels": [],
        },
        {
            "topic": "Carryover (adstock)",
            "setting": "geometric (l_max=8)",
            "detail": "Media effect persists beyond the exposure period.",
            "channels": assumption_channels or ["TV", "Digital"],
        },
    ]
    return {
        "meta": {
            "kpi": "Sales",
            "channels": ["TV", "Digital"],
            "controls": ["Price"],
            "n_obs": 52,
            "n_geos": 0,
            "n_free_params": len(priors),
            "n_periods": 52,
            "date_start": "2020-01-06",
            "date_end": "2020-12-28",
            "likelihood": "normal",
            "link": "identity",
        },
        "assumptions": assumptions,
        "priors": priors,
        "ppc": _canned_ppc() if ppc == "default" else ppc,
        "curves": {"TV": _canned_curve()} if curves == "default" else curves,
        "densities": [{"name": "beta_TV", "samples": rng.gamma(2.0, 1.0, 300)}],
        "sbc": sbc,
        "revisions": list(revisions or []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Charts — pure array inputs
# ─────────────────────────────────────────────────────────────────────────────
class TestPriorCharts:
    def _assert_chart(self, html: str, div_id: str) -> None:
        assert isinstance(html, str) and html.strip()
        assert f'id="{div_id}"' in html
        assert "Plotly.newPlot" in html

    def test_prior_predictive_fan(self, report_config):
        ppc = _canned_ppc()
        html = create_prior_predictive_fan(
            ppc["dates"],
            ppc["observed"],
            ppc["bands"],
            report_config,
            div_id="fanTest",
            kpi_label="Sales",
        )
        self._assert_chart(html, "fanTest")
        assert "Sales" in html

    def test_prior_stat_distribution(self, report_config):
        rng = np.random.default_rng(1)
        html = create_prior_stat_distribution(
            rng.normal(1000, 100, 300),
            1010.0,
            report_config,
            div_id="statTest",
            stat_label="replicate mean",
        )
        self._assert_chart(html, "statTest")
        assert "observed" in html

    def test_prior_density_chart(self, report_config):
        rng = np.random.default_rng(2)
        html = create_prior_density_chart(
            "beta_TV", rng.gamma(2.0, 1.0, 400), report_config, div_id="densTest"
        )
        self._assert_chart(html, "densTest")
        assert "beta_TV" in html

    def test_prior_density_chart_degenerate_constant(self, report_config):
        """Zero-variance samples must not raise — histogram fallback kicks in."""
        html = create_prior_density_chart(
            "const_param",
            np.full(100, 2.0),
            report_config,
            div_id="densConstTest",
        )
        self._assert_chart(html, "densConstTest")
        assert "histogram" in html

    def test_prior_saturation_band(self, report_config):
        html = create_prior_saturation_band(
            "TV", _canned_curve()["saturation"], report_config, div_id="satTest"
        )
        self._assert_chart(html, "satTest")

    def test_prior_adstock_band(self, report_config):
        html = create_prior_adstock_band(
            "TV", _canned_curve()["adstock"], report_config, div_id="adTest"
        )
        self._assert_chart(html, "adTest")

    def test_sbc_rank_histogram(self, report_config):
        html = create_sbc_rank_histogram(
            _sbc_param("beta_TV"), report_config, div_id="sbcHistTest"
        )
        self._assert_chart(html, "sbcHistTest")
        assert "beta_TV" in html

    def test_sbc_rank_histogram_empty_counts_returns_empty(self, report_config):
        assert create_sbc_rank_histogram({"bin_counts": []}, report_config) == ""
        assert create_sbc_rank_histogram({}, report_config) == ""

    def test_sbc_ecdf_diff(self, report_config):
        html = create_sbc_ecdf_diff(
            _sbc_param("beta_TV", with_ranks=True), report_config, div_id="ecdfTest"
        )
        self._assert_chart(html, "ecdfTest")

    def test_sbc_ecdf_diff_missing_ranks_returns_empty(self, report_config):
        assert create_sbc_ecdf_diff(_sbc_param("beta_TV"), report_config) == ""
        assert create_sbc_ecdf_diff({"int_ranks": []}, report_config) == ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. Insights
# ─────────────────────────────────────────────────────────────────────────────
class _FakeReply:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content):
        self._content = content
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _FakeReply(self._content)


class _RaisingLLM:
    def invoke(self, messages):
        raise RuntimeError("LLM unavailable")


class TestPrefitInsights:
    def test_all_slots_filled_without_llm(self):
        insights = build_prefit_insights(_canned_facts(), llm=None)
        for slot in PREFIT_INSIGHT_SLOTS:
            assert slot in insights, f"missing slot {slot}"
            assert insights[slot].strip(), f"empty slot {slot}"

    def test_all_slots_filled_on_sparse_facts(self):
        facts = _canned_facts(ppc=None, curves={}, sbc=None, revisions=None)
        insights = build_prefit_insights(facts, llm=None)
        for slot in PREFIT_INSIGHT_SLOTS:
            assert insights[slot].strip(), f"empty slot {slot}"

    def test_sbc_gloss_varies_with_verdict(self):
        gloss_none = build_prefit_insights(_canned_facts(sbc=None))["sbc_gloss"]
        gloss_ok = build_prefit_insights(_canned_facts(sbc=_canned_sbc(True)))[
            "sbc_gloss"
        ]
        gloss_bad = build_prefit_insights(
            _canned_facts(sbc=_canned_sbc(False, flagged_name="adstock_alpha_TV"))
        )["sbc_gloss"]

        assert "not been run" in gloss_none
        assert gloss_none != gloss_ok != gloss_bad and gloss_none != gloss_bad
        # The miscalibration gloss names the flagged parameter.
        assert "adstock_alpha_TV" in gloss_bad
        assert "adstock_alpha_TV" not in gloss_ok

    def test_revisions_gloss_varies(self):
        empty = build_prefit_insights(_canned_facts(revisions=[]))["revisions_gloss"]
        revs = [{"date": "2026-06-01", "author": "MR", "change": "x", "rationale": "y"}]
        nonempty = build_prefit_insights(_canned_facts(revisions=revs))[
            "revisions_gloss"
        ]
        assert empty != nonempty
        assert "initial specification" in empty
        assert "revision 1" in nonempty

    def test_llm_enrichment_replaces_labelled_slots_only(self):
        facts = _canned_facts()
        templated = build_prefit_insights(facts, llm=None)
        llm = _FakeLLM(
            "STANDFIRST: enriched standfirst.\nPRIORS: enriched priors gloss."
        )
        enriched = build_prefit_insights(facts, llm=llm)

        assert llm.calls == 1
        assert enriched["standfirst"] == "enriched standfirst."
        assert enriched["priors_gloss"] == "enriched priors gloss."
        # Every other slot keeps the templated fallback.
        for slot in PREFIT_INSIGHT_SLOTS:
            if slot in ("standfirst", "priors_gloss"):
                continue
            assert enriched[slot] == templated[slot], f"slot {slot} changed"

    def test_llm_failure_keeps_templated_text(self):
        facts = _canned_facts()
        templated = build_prefit_insights(facts, llm=None)
        degraded = build_prefit_insights(facts, llm=_RaisingLLM())
        assert degraded == templated


# ─────────────────────────────────────────────────────────────────────────────
# 3. Generator from canned facts
# ─────────────────────────────────────────────────────────────────────────────
class TestGeneratorFromCannedFacts:
    def test_full_document_structure(self):
        revisions = [
            {
                "date": "2026-06-01",
                "author": "MR",
                "change": "Tightened the TV adstock prior",
                "rationale": "Prior predictive band was implausibly wide",
            }
        ]
        facts = _canned_facts(
            sbc=_canned_sbc(False, flagged_name="beta_TV"),
            revisions=revisions,
            assumption_channels=["TV", _XSS],
        )
        html = PrefitReadoutGenerator(facts=facts).generate_report()

        # Masthead eyebrow.
        assert "Model design readout" in html
        # All 9 sections in the nav AND as anchors.
        for sid in _SECTION_IDS:
            assert f'href="#{sid}"' in html, f"nav missing #{sid}"
            assert f'id="{sid}"' in html, f"section missing id={sid}"
        # Change record content.
        assert "Tightened the TV adstock prior" in html
        assert "Prior predictive band was implausibly wide" in html
        # SBC verdict table.
        assert "χ² p-value" in html
        assert "beta_TV" in html
        assert "Dispersion z" in html
        # XSS: the raw channel name never appears unescaped.
        assert "<script>alert" not in html
        assert "&lt;script&gt;alert" in html

    def test_sbc_none_still_renders_sbc_section(self):
        facts = _canned_facts(sbc=None)
        html = PrefitReadoutGenerator(facts=facts).generate_report()
        assert 'id="sbc"' in html
        assert 'href="#sbc"' in html
        assert "not been run" in html

    def test_empty_curves_drop_response_section(self):
        facts = _canned_facts(curves={})
        html = PrefitReadoutGenerator(facts=facts).generate_report()
        assert 'href="#response"' not in html
        assert 'id="response"' not in html
        # The remaining 8 sections survive.
        for sid in _SECTION_IDS:
            if sid == "response":
                continue
            assert f'id="{sid}"' in html

    def test_requires_model_or_facts(self):
        with pytest.raises(ValueError, match="model or precomputed facts"):
            PrefitReadoutGenerator()


# ─────────────────────────────────────────────────────────────────────────────
# 4+5. Real unfitted model end-to-end (prior sampling only — no fit)
# ─────────────────────────────────────────────────────────────────────────────
class TestPrefitRealModel:
    def test_priors_enumerated_with_groups(self, real_facts):
        priors = real_facts["priors"]
        assert len(priors) >= 8
        groups = {r["group"] for r in priors}
        for expected in (
            "Media effects",
            "Carryover (adstock)",
            "Saturation",
            "Observation noise",
        ):
            assert expected in groups, f"missing prior group {expected}"
        # Prior draws annotated the rows with finite empirical summaries.
        beta_rows = [r for r in priors if r["name"].startswith("beta_")]
        assert beta_rows
        for r in beta_rows:
            assert r["mean"] is not None and np.isfinite(r["mean"])
            assert r["lower"] is not None and r["upper"] is not None
            assert r["lower"] <= r["upper"]

    def test_assumptions_topics(self, real_facts):
        topics = {a["topic"] for a in real_facts["assumptions"]}
        assert "Observation model" in topics
        assert "Carryover (adstock)" in topics

    def test_prior_predictive_facts(self, real_facts):
        ppc = real_facts["ppc"]
        assert ppc is not None
        assert len(ppc["dates"]) == 52
        assert np.isfinite(ppc["coverage_90"])
        assert 0.0 <= ppc["coverage_90"] <= 1.0
        assert ppc["n_draws"] > 0
        for key in ("p05", "p25", "p50", "p75", "p95"):
            assert len(ppc["bands"][key]) == 52

    def test_prior_response_curves_shapes(self, real_facts):
        curves = real_facts["curves"]
        for ch in ("TV", "Digital"):
            assert ch in curves, f"no curves for {ch}"
            entry = curves[ch]
            assert "saturation" in entry and "adstock" in entry

            sat_med = np.asarray(entry["saturation"]["median"], dtype=float)
            assert np.all(np.diff(sat_med) >= -1e-9), f"{ch} saturation not monotone"

            ad_med = np.asarray(entry["adstock"]["median"], dtype=float)
            assert np.all(np.diff(ad_med) <= 1e-9), f"{ch} adstock not decaying"
            assert ad_med[0] == pytest.approx(np.max(ad_med))

    def test_generator_renders_from_real_model_facts(self, real_facts):
        html = PrefitReadoutGenerator(facts=real_facts).generate_report()
        assert "priorPredictiveFan" in html
        assert "Model design readout" in html
        for sid in _SECTION_IDS:
            assert f'id="{sid}"' in html

    def test_enumerate_model_priors_unit(self, unfitted_model):
        rows = enumerate_model_priors(unfitted_model)
        by_name = {r.name: r for r in rows}
        assert "beta_TV" in by_name
        assert by_name["beta_TV"].family == "Gamma"
        assert by_name["beta_TV"].group == "Media effects"
        # Sorted so the Media effects group leads the table.
        assert rows[0].group == "Media effects"

    def test_model_assumptions_direct(self, unfitted_model):
        rows = model_assumptions(unfitted_model)
        adstock_rows = [r for r in rows if r.topic == "Carryover (adstock)"]
        assert adstock_rows
        # Both channels appear across the adstock assumption rows.
        seen = {ch for r in adstock_rows for ch in r.channels}
        assert seen == {"TV", "Digital"}


# ─────────────────────────────────────────────────────────────────────────────
# Prior estimands + components in time + traces + default SBC (follow-up)
# ─────────────────────────────────────────────────────────────────────────────
class TestPriorEstimandsAndComponents:
    """The prior must be inspectable where decisions live: the estimands, the
    structural components over time (original scale), and single prior draws."""

    def test_real_facts_include_components(self, real_facts):
        comps = real_facts["components"]
        # The core graph registers all four component deterministics.
        assert set(comps) == {"trend", "seasonality", "controls", "media"}
        for key, comp in comps.items():
            bands = comp["bands"]
            assert len(comp["dates"]) == 52
            for b in ("lower", "median", "upper"):
                arr = np.asarray(bands[b], dtype=float)
                assert arr.shape == (52,) and np.all(np.isfinite(arr))
            assert np.all(
                np.asarray(bands["lower"]) <= np.asarray(bands["upper"]) + 1e-12
            )
            assert comp["traces"] is not None and comp["traces"].shape[1] == 52
            assert np.isfinite(comp["abs_scale"]) and comp["abs_scale"] >= 0

    def test_real_facts_include_prior_estimands(self, real_facts):
        est = real_facts["estimands"]
        rows = est["channels"]
        assert {r["channel"] for r in rows} == {"TV", "Digital"}
        for r in rows:
            assert r["label"] == "ROI" and r["reference"] == 1.0
            assert np.isfinite(r["mean"])
            assert r["lower"] <= r["mean"] <= r["upper"]
            assert 0.0 <= r["p_above_reference"] <= 1.0
            assert r["draws"].size > 0 and np.all(np.isfinite(r["draws"]))
        blended = est["blended"]
        assert blended["lower"] <= blended["mean"] <= blended["upper"]
        assert not blended["partial"]  # both channels are spend-measured
        share = est["marketing_share"]
        assert 0.0 <= share["lower"] <= share["mean"] <= share["upper"]

    def test_real_facts_ppc_traces_and_scale_z(self, real_facts):
        ppc = real_facts["ppc"]
        assert ppc["traces"] is not None
        assert ppc["traces"].shape[1] == 52
        assert np.isfinite(ppc["scale_z_abs_mean"])

    def test_real_render_includes_new_sections(self, real_facts):
        html = PrefitReadoutGenerator(facts=real_facts).generate_report()
        for marker in (
            'id="components"',
            'id="prior-estimands"',
            "priorComponent_0",
            "priorEstimand_0",
            "Prior draws",  # spaghetti legend entry on the fan
            "break-even",  # reference line on the estimand densities
        ):
            assert marker in html, f"missing {marker}"

    def test_canned_facts_without_new_keys_drop_sections(self):
        facts = _canned_facts()
        facts.pop("components", None)
        facts.pop("estimands", None)
        html = PrefitReadoutGenerator(facts=facts).generate_report()
        assert 'id="components"' not in html
        assert 'id="prior-estimands"' not in html

    def test_component_chart_and_density_reference(self, report_config):
        from mmm_framework.reporting.charts.prior import (
            create_prior_component_chart,
            create_prior_density_chart,
            create_prior_predictive_fan,
        )

        comp = {
            "dates": [f"2024-01-{d:02d}" for d in range(1, 11)],
            "bands": {
                "lower": np.linspace(-1, -2, 10),
                "median": np.zeros(10),
                "upper": np.linspace(1, 2, 10),
            },
            "traces": np.random.default_rng(0).normal(0, 1, (4, 10)),
        }
        html = create_prior_component_chart(
            "Baseline trend", comp, report_config, div_id="compX"
        )
        assert "compX" in html and "Plotly.newPlot" in html

        dens = create_prior_density_chart(
            "TV",
            np.random.default_rng(1).gamma(2, 1, 300),
            report_config,
            div_id="densRef",
            reference=1.0,
            reference_label="break-even",
        )
        assert "densRef" in html or "densRef" in dens
        assert "break-even" in dens

        ppc = _canned_ppc()
        fan = create_prior_predictive_fan(
            ppc["dates"],
            ppc["observed"],
            ppc["bands"],
            report_config,
            div_id="fanTr",
            sample_traces=np.random.default_rng(2).normal(1000, 50, (3, 20)),
        )
        assert "Prior draws" in fan

    def test_insight_slots_cover_new_sections(self, real_facts):
        from mmm_framework.reporting import PREFIT_INSIGHT_SLOTS, build_prefit_insights

        ins = build_prefit_insights(real_facts)
        assert set(PREFIT_INSIGHT_SLOTS) <= set(ins)
        assert "estimands_gloss" in ins and ins["estimands_gloss"]
        assert "components_gloss" in ins and ins["components_gloss"]
        # Grounded: the estimand gloss quotes the blended prior return.
        assert (
            f"{real_facts['estimands']['blended']['mean']:.2f}"
            in ins["estimands_gloss"]
        )
        # Sparse facts flip both glosses to the unavailable wording.
        sparse = dict(real_facts, components={}, estimands={})
        ins2 = build_prefit_insights(sparse)
        assert ins2["estimands_gloss"] != ins["estimands_gloss"]
        assert ins2["components_gloss"] != ins["components_gloss"]


class TestDefaultSbcRun:
    """SBC runs by default when the generator gets a model and no stored result."""

    class _FakeParam:
        def __init__(self):
            self.int_ranks = np.arange(10)

        def to_dashboard(self, *, max_ranks: int = 0):
            out = {
                "name": "beta_TV",
                "L": 50,
                "n_sims": 10,
                "n_bins": 10,
                "bin_counts": [1] * 10,
                "chi2_pvalue": 0.9,
                "shape": "uniform",
                "miscalibration": 0.01,
                "calibrated": True,
            }
            if max_ranks:
                out["int_ranks"] = [int(r) for r in self.int_ranks[:max_ranks]]
            return out

    class _FakeSbc:
        def __init__(self):
            self.params = [TestDefaultSbcRun._FakeParam()]

        def to_dashboard(self):
            return {
                "all_calibrated": True,
                "n_sims_effective": 10,
                "n_sims_requested": 10,
                "L": 50,
                "n_bins": 10,
                "sampler": "numpyro",
                "seed": 0,
                "alpha": 0.05,
                "elapsed_s": 1.0,
                "n_failed_fits": 0,
                "caveats": [],
                "params": [p.to_dashboard() for p in self.params],
            }

    def test_run_sbc_default_invokes_engine_with_ranks(
        self, unfitted_model, monkeypatch
    ):
        calls: dict = {}

        def fake_run(model, **kwargs):
            calls["kwargs"] = kwargs
            return TestDefaultSbcRun._FakeSbc()

        import mmm_framework.diagnostics.sbc as sbc_mod

        monkeypatch.setattr(sbc_mod, "run_mmm_sbc", fake_run)
        gen = PrefitReadoutGenerator(
            unfitted_model, n_prior_samples=50, sbc_kwargs={"n_sims": 7}
        )
        assert calls["kwargs"]["n_sims"] == 7  # override merged over defaults
        assert calls["kwargs"]["L"] == 50
        sbc = gen.facts["sbc"]
        assert sbc is not None and sbc["all_calibrated"]
        # Serialized WITH ranks so the ECDF-difference panel renders.
        assert sbc["params"][0]["int_ranks"] == list(range(10))
        html = gen.generate_report()
        assert "sbcEcdf_0" in html

    def test_run_sbc_false_skips(self, unfitted_model, monkeypatch):
        import mmm_framework.diagnostics.sbc as sbc_mod

        def boom(*a, **k):  # pragma: no cover - must not be called
            raise AssertionError("SBC must not run when run_sbc=False")

        monkeypatch.setattr(sbc_mod, "run_mmm_sbc", boom)
        gen = PrefitReadoutGenerator(unfitted_model, n_prior_samples=50, run_sbc=False)
        assert gen.facts["sbc"] is None or not gen.facts["sbc"]

    def test_sbc_failure_degrades_gracefully(self, unfitted_model, monkeypatch):
        import mmm_framework.diagnostics.sbc as sbc_mod

        def boom(*a, **k):
            raise RuntimeError("sampler exploded")

        monkeypatch.setattr(sbc_mod, "run_mmm_sbc", boom)
        gen = PrefitReadoutGenerator(unfitted_model, n_prior_samples=50)
        assert not gen.facts.get("sbc")
        html = gen.generate_report()
        assert 'id="sbc"' in html  # the not-yet-run section still renders

    def test_facts_only_path_never_runs_sbc(self, monkeypatch):
        import mmm_framework.diagnostics.sbc as sbc_mod

        def boom(*a, **k):  # pragma: no cover
            raise AssertionError("SBC must not run without a model")

        monkeypatch.setattr(sbc_mod, "run_mmm_sbc", boom)
        gen = PrefitReadoutGenerator(facts=_canned_facts())
        assert gen.generate_report()
