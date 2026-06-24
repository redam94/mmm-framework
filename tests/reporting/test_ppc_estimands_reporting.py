"""End-to-end checks for the estimand-results + posterior-predictive reporting.

A real (tiny) BayesianMMM is fit, then the BayesianMMMExtractor must populate
``bundle.estimands`` (mean + CI for the model's declared/default estimands) and
``bundle.posterior_predictive`` (goodness-of-fit summary), and the generated
report must carry both the EstimandsSection and the PosteriorPredictiveSection.

These exercise the real model surface (``evaluate_estimands`` /
``predict``) -- a seam a hand-built bundle would never catch -- so they are
marked slow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor


@pytest.fixture(scope="module")
def fitted_model_and_panel():
    periods = pd.date_range("2021-01-04", periods=48, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(11)
    t = np.arange(n)
    tv = np.abs(rng.normal(120, 30, n))
    search = np.abs(rng.normal(60, 20, n))
    y = pd.Series(
        1000
        + 8.0 * t
        + 50.0 * np.sin(2 * np.pi * t / 52)
        + 1.4 * tv
        + 2.2 * search
        + rng.normal(0, 25, n),
        name="Sales",
    )
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Search"],
        controls=None,
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Search", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Search": search}),
        X_controls=None,
        coords=coords,
        index=periods,
        config=config,
    )
    model = BayesianMMM(
        panel,
        ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=2,
            n_draws=150,
            n_tune=150,
        ),
        TrendConfig(type=TrendType.LINEAR),
    )
    model.fit(random_seed=0)
    return model, panel


@pytest.mark.slow
def test_extractor_populates_estimands(fitted_model_and_panel):
    model, panel = fitted_model_and_panel
    bundle = BayesianMMMExtractor(model, panel=panel).extract()

    assert bundle.estimands, "expected default estimands to be realized"
    for key, v in bundle.estimands.items():
        assert v["mean"] is not None
        assert "kind" in v and "hdi_prob" in v
        # CI bounds present and ordered when finite.
        if v.get("lower") is not None and v.get("upper") is not None:
            assert v["lower"] <= v["upper"]
    # At least one ROI estimand per the MMM capability defaults.
    assert any(
        "roi" in v["kind"] or "roas" in v["kind"] for v in bundle.estimands.values()
    )


@pytest.mark.slow
def test_extractor_populates_posterior_predictive(fitted_model_and_panel):
    model, panel = fitted_model_and_panel
    bundle = BayesianMMMExtractor(model, panel=panel).extract()

    pp = bundle.posterior_predictive
    assert pp is not None
    n = len(np.asarray(pp["observed"]))
    assert len(np.asarray(pp["pred_mean"])) == n
    assert np.asarray(pp["samples"]).ndim == 2
    assert np.asarray(pp["samples"]).shape[1] == n

    # Calibration curve is monotone-ish in nominal and bounded in [0, 1].
    assert pp["coverage"]
    for pt in pp["coverage"]:
        assert 0.0 <= pt["empirical"] <= 1.0
    # Posterior-predictive p-values for the four summary statistics.
    assert set(pp["bayes_p"]) == {"mean", "std", "min", "max"}
    for p in pp["bayes_p"].values():
        assert 0.0 <= p <= 1.0
    # A well-fit synthetic model should track the data reasonably.
    assert pp["r2"] is not None and pp["r2"] > 0.5


@pytest.mark.slow
def test_report_includes_estimand_and_ppc_sections(fitted_model_and_panel):
    model, panel = fitted_model_and_panel
    html = MMMReportGenerator(model=model, panel=panel).render()

    assert 'id="estimands"' in html
    assert 'id="posterior-predictive"' in html
    # The PPC section embeds its four goodness-of-fit charts.
    assert "ppcObservedVsPredicted" in html
    assert "ppcDensityOverlay" in html
    assert "ppcCalibration" in html
    assert "ppcResiduals" in html


# =============================================================================
# Fast extractor unit tests (no model fit) — finiteness + estimand mapping
# =============================================================================


class _FakePred:
    def __init__(self, y_rep, mean, lo, hi):
        self.y_pred_samples = y_rep
        self.y_pred_mean = mean
        self.y_pred_hdi_low = lo
        self.y_pred_hdi_high = hi


class _FakeResult:
    def __init__(self, mean, lo, hi, kind="roi", status="ok"):
        self.mean = mean
        self.hdi_low = lo
        self.hdi_high = hi
        self.kind = kind
        self.status = status
        self.units = ""
        self.hdi_prob = 0.94
        self.extra = {"prob_positive": 0.9}


class _FakeModel:
    """Minimal model surface the BayesianMMMExtractor PPC/estimand paths touch."""

    def __init__(self, y, pred, estimands):
        self.y = y
        self.y_mean = 0.0
        self.y_std = 1.0
        self.channel_names = ["TV"]
        self._trace = object()
        self._results = None
        self._pred = pred
        self._estimands = estimands

    def predict(self, return_original_scale=True, hdi_prob=0.8):
        return self._pred

    def evaluate_estimands(self):
        return self._estimands


def _extractor(model):
    from mmm_framework.reporting.extractors.bundle import MMMDataBundle

    return BayesianMMMExtractor(model), MMMDataBundle()


def test_ppc_extraction_populates_on_finite_arrays():
    rng = np.random.default_rng(1)
    n = 20
    y = rng.normal(100, 10, n)
    y_rep = rng.normal(100, 10, (60, n))
    pred = _FakePred(
        y_rep,
        y_rep.mean(axis=0),
        np.percentile(y_rep, 10, axis=0),
        np.percentile(y_rep, 90, axis=0),
    )
    model = _FakeModel(y, pred, {})
    ext, bundle = _extractor(model)
    bundle = ext._extract_posterior_predictive(bundle)
    assert bundle.posterior_predictive is not None
    assert set(bundle.posterior_predictive["bayes_p"]) == {"mean", "std", "min", "max"}


def test_ppc_extraction_bails_on_nonfinite():
    rng = np.random.default_rng(2)
    n = 20
    y = rng.normal(100, 10, n)
    y_rep = rng.normal(100, 10, (60, n))
    y_rep[3, 5] = np.nan  # a single non-finite replicate value
    pred = _FakePred(y_rep, y_rep.mean(axis=0), None, None)
    model = _FakeModel(y, pred, {})
    ext, bundle = _extractor(model)
    bundle = ext._extract_posterior_predictive(bundle)
    # Non-finite replicate -> the whole PPC view is dropped (no NaN in report).
    assert bundle.posterior_predictive is None


def test_estimand_extraction_skips_nan_mean():
    model = _FakeModel(
        np.zeros(5),
        _FakePred(np.zeros((2, 5)), np.zeros(5), None, None),
        {
            "roi:TV": _FakeResult(1.8, 1.2, 2.4),
            "roi:Bad": _FakeResult(float("nan"), 1.0, 2.0),
            "roi:Unsup": _FakeResult(None, None, None, status="unsupported"),
        },
    )
    ext, bundle = _extractor(model)
    bundle = ext._extract_estimands(bundle)
    assert bundle.estimands is not None
    assert "roi:TV" in bundle.estimands
    assert "roi:Bad" not in bundle.estimands  # NaN mean dropped
    assert "roi:Unsup" not in bundle.estimands  # unsupported dropped
    assert bundle.estimands["roi:TV"]["prob_positive"] == 0.9


# =============================================================================
# Extended models (NestedMMM) — graph-sampled posterior-predictive (slow)
# =============================================================================


@pytest.fixture(scope="module")
def fitted_nested_model():
    from mmm_framework.mmm_extensions import NestedMMM
    from mmm_framework.mmm_extensions.builders import (
        MediatorConfigBuilder,
        NestedModelConfigBuilder,
    )

    channels = ["tv", "digital", "social"]
    rng = np.random.default_rng(13)
    periods = pd.date_range("2021-01-04", periods=52, freq="W-MON")
    X_media = np.abs(rng.normal(100, 40, (52, len(channels))))
    y = 1000 + 1.5 * X_media[:, 0] + 1.0 * X_media[:, 1] + rng.normal(0, 60, 52)

    mediator = (
        MediatorConfigBuilder("brand_awareness")
        .partially_observed(observation_noise=0.15)
        .with_positive_media_effect(sigma=1.0)
        .build()
    )
    config = (
        NestedModelConfigBuilder()
        .add_mediator(mediator)
        .map_channels_to_mediator("brand_awareness", ["tv", "digital"])
        .build()
    )
    model = NestedMMM(
        X_media=X_media, y=y, channel_names=channels, config=config, index=periods
    )
    model.fit(draws=150, tune=150, chains=2, random_seed=0)
    return model


@pytest.mark.slow
def test_extended_extractor_populates_ppc(fitted_nested_model):
    from mmm_framework.reporting.extractors.extended import ExtendedMMMExtractor

    bundle = ExtendedMMMExtractor(fitted_nested_model).extract()
    pp = bundle.posterior_predictive
    assert pp is not None, "extended PPC should populate via graph sampling"
    n = len(np.asarray(pp["observed"]))
    assert len(np.asarray(pp["pred_mean"])) == n
    assert np.asarray(pp["samples"]).ndim == 2
    assert np.asarray(pp["samples"]).shape[1] == n
    assert set(pp["bayes_p"]) == {"mean", "std", "min", "max"}
    for c in pp["coverage"]:
        assert 0.0 <= c["empirical"] <= 1.0
    # The 90% predictive interval should cover the bulk of observations.
    cov90 = next(c["empirical"] for c in pp["coverage"] if c["nominal"] == 0.9)
    assert cov90 > 0.6


@pytest.mark.slow
def test_extended_report_includes_ppc_section(fitted_nested_model):
    # Full report render (exercises every section, incl. the mediator pathway
    # chart whose channels fall outside the default color palette → the hsl
    # fallback path in _hex_to_rgb).
    html = MMMReportGenerator(model=fitted_nested_model).render()
    assert 'id="posterior-predictive"' in html
    assert "ppcObservedVsPredicted" in html
    assert "ppcResiduals" in html
