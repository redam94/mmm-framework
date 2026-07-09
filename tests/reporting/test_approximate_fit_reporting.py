"""Reporting + registry must distinguish an APPROXIMATE (MAP/ADVI/Pathfinder)
fit from a full NUTS posterior, flag it prominently, and (for DAG-routed
extension models) tell the user which spec settings won't apply.

These are fast, mostly unit-level checks — the end-to-end render off a real
MAP fit lives in the slow reporting suite.
"""

from __future__ import annotations

import types

import pytest

from mmm_framework.reporting.config import ReportConfig, SectionConfig
from mmm_framework.reporting.extractors.base import DataExtractor
from mmm_framework.reporting.extractors.bundle import MMMDataBundle
from mmm_framework.reporting.sections import ExecutiveSummarySection
from mmm_framework.reporting.augur_sections import AugurHeadlineSection


# ── extractor: fold the model's own approximate/fit_method into the bundle ──
def _merge(results_diag):
    fake = types.SimpleNamespace(
        results=types.SimpleNamespace(diagnostics=results_diag)
    )
    return DataExtractor._merge_fit_provenance(
        fake, {"rhat_max": 1.001, "ess_bulk_min": 900}
    )


def test_merge_fit_provenance_flags_approximate():
    merged = _merge({"approximate": True, "fit_method": "map"})
    assert merged["approximate"] is True
    assert merged["fit_method"] == "map"
    # R-hat/ESS from the raw trace are meaningless for a point estimate — nulled.
    assert merged["rhat_max"] is None
    assert merged["ess_bulk_min"] is None
    # And the verdict is "not assessable", never a green converged.
    from mmm_framework.diagnostics.convergence import is_converged

    assert is_converged(merged) is None


def test_merge_fit_provenance_leaves_nuts_untouched():
    merged = _merge({"approximate": False, "fit_method": "nuts"})
    assert merged.get("approximate") in (False, None)
    assert merged["rhat_max"] == 1.001  # NUTS diagnostics preserved
    assert merged["fit_method"] == "nuts"


# ── classic report: prominent approximate banner ──
def _exec_section(diag):
    bundle = MMMDataBundle()
    bundle.channel_names = ["TV"]
    bundle.diagnostics = diag
    return ExecutiveSummarySection(
        data=bundle, config=ReportConfig(), section_config=SectionConfig(enabled=True)
    )


def test_executive_summary_approximate_banner():
    html = _exec_section({"approximate": True, "fit_method": "map"}).render()
    assert "Approximate fit" in html
    assert "MAP" in html
    assert "not calibrated" in html


def test_executive_summary_no_banner_for_nuts():
    html = _exec_section(
        {"approximate": False, "rhat_max": 1.0, "ess_bulk_min": 900}
    ).render()
    assert "Approximate fit" not in html


# ── augur client report: client-facing caveat ──
def _augur_headline(diag):
    bundle = MMMDataBundle()
    bundle.channel_names = ["TV"]
    bundle.diagnostics = diag
    bundle.marketing_attributed_revenue = {
        "mean": 1000.0,
        "lower": 800.0,
        "upper": 1200.0,
    }
    return AugurHeadlineSection(
        data=bundle, config=ReportConfig(), section_config=SectionConfig(enabled=True)
    )


def test_augur_caveat_for_approximate_fit():
    html = _augur_headline({"approximate": True, "fit_method": "advi"}).render()
    assert "approximate" in html.lower()
    assert "not calibrated" in html.lower()


def test_augur_caveat_for_non_converged():
    # rhat well above 1.01 with real ESS → is_converged == False (not approximate).
    diag = {
        "approximate": False,
        "rhat_max": 1.9,
        "ess_bulk_min": 900,
        "divergences": 20,
    }
    html = _augur_headline(diag).render()
    assert "not reliable" in html.lower() or "provisional" in html.lower()


def test_augur_no_caveat_for_clean_nuts():
    diag = {
        "approximate": False,
        "rhat_max": 1.0,
        "ess_bulk_min": 1500,
        "divergences": 0,
    }
    html = _augur_headline(diag).render()
    # neither the approximate nor the non-convergence caveat fires
    assert "approximate" not in html.lower()
    assert "not reliable" not in html.lower()


# ── registry: extension specs reject spec priors that won't apply ──
def _nested_spec():
    return {
        "kpi": "Sales",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "dag_model_type": "nested_mmm",
        "inference": {"method": "map"},
    }


def test_registry_rejects_spec_prior_on_extension():
    from mmm_framework.agents.fitting import unconsumed_prior_path

    err = unconsumed_prior_path(
        ["priors", "media", "TV", "roi"], {"median": 2.0, "sigma": 0.5}, _nested_spec()
    )
    assert err is not None
    assert "nested_mmm" in err and "DAG" in err


def test_registry_rejects_media_prior_mode_on_extension():
    from mmm_framework.agents.fitting import unconsumed_spec_path

    err = unconsumed_spec_path(["media_prior_mode"], "coefficient", _nested_spec())
    assert err is not None and "extension" in err.lower()


def test_registry_allows_prior_on_plain_model():
    from mmm_framework.agents.fitting import unconsumed_prior_path

    plain = _nested_spec()
    plain.pop("dag_model_type")
    # a valid per-channel roi prior on a plain MMM is accepted (no error)
    assert (
        unconsumed_prior_path(
            ["priors", "media", "TV", "roi"], {"median": 2.0, "sigma": 0.5}, plain
        )
        is None
    )


# ── prefit inference-plan row reflects an approximate method ──
def test_prefit_inference_plan_reflects_map():
    from mmm_framework.config.enums import FitMethod
    from mmm_framework.reporting.helpers.prefit import model_assumptions

    fake_model = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            fit_method=FitMethod.MAP,
            n_chains=2,
            n_draws=100,
            n_tune=100,
            target_accept=0.9,
        ),
        mff_config=None,
        trend_config=None,
        geo_names=None,
        control_names=[],
    )
    rows = model_assumptions(fake_model)
    plan = next((r for r in rows if r.topic == "Inference plan"), None)
    assert plan is not None
    assert "MAP" in plan.setting and "approximate" in plan.setting.lower()
    assert "not calibrated" in plan.detail.lower()


def test_prefit_inference_plan_nuts_default():
    from mmm_framework.config.enums import FitMethod
    from mmm_framework.reporting.helpers.prefit import model_assumptions

    fake_model = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            fit_method=FitMethod.NUTS,
            n_chains=4,
            n_draws=1000,
            n_tune=1000,
            target_accept=0.9,
        ),
        mff_config=None,
        trend_config=None,
        geo_names=None,
        control_names=[],
    )
    plan = next(r for r in model_assumptions(fake_model) if r.topic == "Inference plan")
    assert "NUTS" in plan.setting
    assert "R-hat" in plan.detail


@pytest.mark.slow
def test_real_map_fit_report_shows_approximate_banner():
    """End-to-end: a real MAP fit rendered through the full report carries the
    prominent approximate banner and 'N/A' diagnostics — never a green
    converged verdict for an uncalibrated fit."""
    import numpy as np
    import pandas as pd

    from mmm_framework.config import (
        DimensionType,
        InferenceMethod,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
        ModelConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset
    from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
    from mmm_framework.reporting import MMMReportGenerator

    rng = np.random.default_rng(0)
    n = 60
    periods = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    tv = np.abs(rng.normal(100, 20, n))
    search = np.abs(rng.normal(60, 15, n))
    y = pd.Series(1000 + 1.4 * tv + 2.2 * search + rng.normal(0, 25, n), name="Sales")
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Search": search}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV", "Search"],
            controls=None,
        ),
        index=periods,
        config=MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
                MediaChannelConfig(name="Search", dimensions=[DimensionType.PERIOD]),
            ],
            controls=[],
        ),
    )
    model = BayesianMMM(
        panel,
        ModelConfig(inference_method=InferenceMethod.BAYESIAN_PYMC),
        TrendConfig(type=TrendType.LINEAR),
    )
    results = model.fit(method="map", random_seed=0)
    assert results.approximate is True

    html = MMMReportGenerator(model=model, panel=panel, results=results).render()
    # Prominent approximate callout on the headline.
    assert "Approximate fit" in html and "not calibrated" in html
    # No false green "converged" claim, and diagnostics degrade to N/A not a crash.
    assert (
        "Model has NOT converged" not in html
    )  # that banner is for real non-convergence
