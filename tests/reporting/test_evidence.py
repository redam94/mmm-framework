"""Tests for the evidence tier + identifiability gate (issue #102).

Covers the pure ``reporting/evidence.py`` helper (tier logic, collinearity,
chip rendering), the extractor integration (``_extract_channel_evidence``
stamping the bundle), and end-to-end rendering across the classic report, the
augur readout, and the interactive report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.reporting.evidence import (
    ChannelEvidence,
    EvidenceTier,
    channel_evidence,
    collinearity_from_matrix,
    evidence_chip_html,
    evidence_legend_html,
)


# ---------------------------------------------------------------------------
# Tier logic
# ---------------------------------------------------------------------------
class TestChannelEvidenceTiers:
    def _learning(self):
        return pd.DataFrame(
            [
                {
                    "parameter": "beta_TV",
                    "contraction": 0.02,
                    "verdict": "prior-dominated",
                },
                {"parameter": "beta_Search", "contraction": 0.75, "verdict": "strong"},
                {"parameter": "beta_Radio", "contraction": 0.55, "verdict": "moderate"},
            ]
        )

    def test_experiment_channel_is_validated(self):
        ev = channel_evidence(
            ["TV", "Search"],
            experiment_channels={"Search"},
            learning=self._learning(),
        )
        assert ev["Search"].tier is EvidenceTier.EXPERIMENT_VALIDATED
        assert ev["Search"].experiment is True

    def test_prior_dominated_from_verdict(self):
        ev = channel_evidence(["TV"], learning=self._learning())
        assert ev["TV"].tier is EvidenceTier.PRIOR_DOMINATED
        assert ev["TV"].gated is True

    def test_prior_dominated_from_low_contraction_without_verdict(self):
        learn = pd.DataFrame(
            [{"parameter": "beta_TV", "contraction": 0.03, "verdict": None}]
        )
        ev = channel_evidence(["TV"], learning=learn)
        assert ev["TV"].tier is EvidenceTier.PRIOR_DOMINATED

    def test_relocated_is_not_prior_dominated(self):
        # A relocated parameter has low contraction but moved a lot — the
        # evidence dominated the LOCATION, so it is model-identified.
        learn = pd.DataFrame(
            [{"parameter": "beta_TV", "contraction": -0.2, "verdict": "relocated"}]
        )
        ev = channel_evidence(["TV"], learning=learn)
        assert ev["TV"].tier is EvidenceTier.MODEL_IDENTIFIED

    def test_model_identified_default(self):
        ev = channel_evidence(["Search"], learning=self._learning())
        assert ev["Search"].tier is EvidenceTier.MODEL_IDENTIFIED
        assert ev["Search"].gated is False

    def test_roi_mode_parameter_names_match(self):
        # ROI-parameterized default priors expose ``roi_<ch>`` instead of beta.
        learn = pd.DataFrame(
            [{"parameter": "roi_TV", "contraction": 0.01, "verdict": "prior-dominated"}]
        )
        ev = channel_evidence(["TV"], learning=learn)
        assert ev["TV"].tier is EvidenceTier.PRIOR_DOMINATED

    def test_per_geo_parameters_take_worst(self):
        # Vector beta_<ch>[g] — the least-learned geo dominates the tier.
        learn = pd.DataFrame(
            [
                {"parameter": "beta_TV[0]", "contraction": 0.8, "verdict": "strong"},
                {
                    "parameter": "beta_TV[1]",
                    "contraction": 0.02,
                    "verdict": "prior-dominated",
                },
            ]
        )
        ev = channel_evidence(["TV"], learning=learn)
        assert ev["TV"].tier is EvidenceTier.PRIOR_DOMINATED

    def test_no_learning_defaults_to_model_identified(self):
        ev = channel_evidence(["TV", "Search"])
        assert all(e.tier is EvidenceTier.MODEL_IDENTIFIED for e in ev.values())


# ---------------------------------------------------------------------------
# Collinearity / identifiability
# ---------------------------------------------------------------------------
class TestCollinearity:
    def test_collinear_pair_flagged(self):
        rng = np.random.RandomState(0)
        base = rng.randn(80, 3)
        base[:, 2] = base[:, 1] * 0.99 + 0.01 * rng.randn(80)  # col2 ≈ col1
        coll = collinearity_from_matrix(base, ["A", "B", "C"])
        assert coll["B"]["vif"] > 5.0
        assert coll["C"]["vif"] > 5.0
        assert "C" in coll["B"]["collinear_with"] or "B" in coll["C"]["collinear_with"]
        assert coll["A"]["vif"] < 5.0

    def test_independent_channels_not_flagged(self):
        rng = np.random.RandomState(1)
        mat = rng.randn(80, 3)
        coll = collinearity_from_matrix(mat, ["A", "B", "C"])
        assert all(coll[c]["vif"] < 5.0 for c in ["A", "B", "C"])
        assert all(not coll[c]["collinear_with"] for c in ["A", "B", "C"])

    def test_single_channel_no_inflation(self):
        coll = collinearity_from_matrix(np.random.randn(50, 1), ["A"])
        assert coll["A"]["vif"] == 1.0

    def test_collinearity_flows_into_identifiability(self):
        rng = np.random.RandomState(2)
        mat = rng.randn(80, 2)
        mat[:, 1] = mat[:, 0] * 0.995 + 0.005 * rng.randn(80)
        coll = collinearity_from_matrix(mat, ["Search", "Shopping"])
        ev = channel_evidence(["Search", "Shopping"], collinearity=coll)
        assert ev["Search"].identified is False
        assert ev["Search"].caveat() is not None
        assert "Shopping" in ev["Search"].caveat()


# ---------------------------------------------------------------------------
# ChannelEvidence behavior
# ---------------------------------------------------------------------------
class TestChannelEvidence:
    def test_experiment_validated_never_gated_by_collinearity(self):
        ev = ChannelEvidence(
            channel="Search",
            tier=EvidenceTier.EXPERIMENT_VALIDATED,
            identified=False,
            collinear_with=["Shopping"],
        )
        assert ev.gated is False
        assert ev.caveat() is None  # experiment measured it directly

    def test_to_dict_round_trips_fields(self):
        ev = ChannelEvidence(
            channel="TV",
            tier=EvidenceTier.PRIOR_DOMINATED,
            contraction=0.02,
            learning_verdict="prior-dominated",
        )
        d = ev.to_dict()
        assert d["tier"] == "prior-dominated"
        assert d["gated"] is True
        assert d["short_label"] == "Prior-driven"
        assert d["gloss"]


# ---------------------------------------------------------------------------
# Chip rendering
# ---------------------------------------------------------------------------
class TestChipRendering:
    def test_classic_chip_from_object(self):
        ev = ChannelEvidence(channel="TV", tier=EvidenceTier.MODEL_IDENTIFIED)
        html = evidence_chip_html(ev, theme="classic")
        assert "evidence-chip ev-model" in html
        assert "Modeled" in html

    def test_augur_chip_from_dict(self):
        d = ChannelEvidence(channel="TV", tier=EvidenceTier.PRIOR_DOMINATED).to_dict()
        html = evidence_chip_html(d, theme="augur")
        assert "tier-chip t-reduce" in html
        assert "Prior-driven" in html

    def test_caveat_chip_rendered_when_not_identified(self):
        ev = ChannelEvidence(
            channel="Search",
            tier=EvidenceTier.MODEL_IDENTIFIED,
            identified=False,
            collinear_with=["Shopping"],
        )
        html = evidence_chip_html(ev, theme="classic")
        assert "not separately identified" in html

    def test_no_caveat_chip_for_experiment_validated_collinear(self):
        ev = ChannelEvidence(
            channel="Search",
            tier=EvidenceTier.EXPERIMENT_VALIDATED,
            identified=False,
            collinear_with=["Shopping"],
        )
        html = evidence_chip_html(ev, theme="classic")
        assert "not separately identified" not in html

    def test_none_renders_empty(self):
        assert evidence_chip_html(None) == ""

    def test_legend_lists_all_three_tiers(self):
        html = evidence_legend_html(theme="classic")
        assert "Validated" in html and "Modeled" in html and "Prior-driven" in html


# ---------------------------------------------------------------------------
# Extractor integration (fake model → bundle)
# ---------------------------------------------------------------------------
class _FakeExperiment:
    def __init__(self, channel):
        self.channel = channel


class _FakeModel:
    """Minimal model exposing exactly what ``_extract_channel_evidence`` reads."""

    def __init__(self, channels, learning, experiments, media):
        self.channel_names = channels
        self._trace = object()  # truthy marker for "fitted"
        self.experiments = experiments
        self.X_media_raw = media
        self._learning = learning

    def compute_parameter_learning(self, prior_samples=400, random_seed=0):
        return self._learning


def _fake_extractor(model, bundle):
    from mmm_framework.reporting.extractors.mixins import EstimandPPCMixin

    class _Ex(EstimandPPCMixin):
        def __init__(self, m):
            self.mmm = m
            self.model = m

    ex = _Ex(model)
    return ex._extract_channel_evidence(bundle)


class TestExtractorIntegration:
    def _bundle_and_model(self):
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        channels = ["TV", "Search", "Shopping"]
        rng = np.random.RandomState(3)
        media = rng.rand(80, 3)
        media[:, 2] = media[:, 1] * 0.99 + 0.01 * rng.rand(80)  # Search~Shopping
        learning = pd.DataFrame(
            [
                {
                    "parameter": "beta_TV",
                    "contraction": 0.01,
                    "verdict": "prior-dominated",
                },
                {"parameter": "beta_Search", "contraction": 0.7, "verdict": "strong"},
                {"parameter": "beta_Shopping", "contraction": 0.6, "verdict": "strong"},
            ]
        )
        model = _FakeModel(channels, learning, [_FakeExperiment("Search")], media)
        bundle = MMMDataBundle(
            channel_names=channels,
            channel_roi={
                ch: {"mean": 2.0, "lower": 1.5, "upper": 2.5} for ch in channels
            },
            estimands={
                "contribution_roi:TV": {"mean": 2.0, "lower": 1.5, "upper": 2.5},
                "marginal_roas:Search": {"mean": 3.0, "lower": 2.5, "upper": 3.5},
                "blended_roi": {"mean": 2.2, "lower": 1.9, "upper": 2.6},
            },
        )
        return bundle, model

    def test_bundle_channel_evidence_populated(self):
        bundle, model = self._bundle_and_model()
        bundle = _fake_extractor(model, bundle)
        assert bundle.channel_evidence is not None
        assert bundle.channel_evidence["TV"]["tier"] == "prior-dominated"
        assert bundle.channel_evidence["Search"]["tier"] == "experiment-validated"

    def test_evidence_stamped_on_channel_roi(self):
        bundle, model = self._bundle_and_model()
        bundle = _fake_extractor(model, bundle)
        assert bundle.channel_roi["TV"]["evidence"]["tier"] == "prior-dominated"

    def test_evidence_stamped_on_per_channel_estimands_only(self):
        bundle, model = self._bundle_and_model()
        bundle = _fake_extractor(model, bundle)
        assert (
            bundle.estimands["contribution_roi:TV"]["evidence"]["tier"]
            == "prior-dominated"
        )
        # A non-channel estimand key gets no evidence.
        assert "evidence" not in bundle.estimands["blended_roi"]

    def test_collinear_channels_not_identified(self):
        bundle, model = self._bundle_and_model()
        bundle = _fake_extractor(model, bundle)
        assert bundle.channel_evidence["Shopping"]["identified"] is False

    def test_missing_channel_names_is_noop(self):
        from mmm_framework.reporting.extractors.bundle import MMMDataBundle

        model = _FakeModel([], pd.DataFrame(), [], None)
        bundle = _fake_extractor(model, MMMDataBundle())
        assert bundle.channel_evidence is None


# ---------------------------------------------------------------------------
# End-to-end rendering
# ---------------------------------------------------------------------------
def _bundle_with_evidence():
    from mmm_framework.reporting.extractors.bundle import MMMDataBundle

    channels = ["TV", "Search", "Shopping"]
    ev = {
        ch: e.to_dict()
        for ch, e in channel_evidence(
            channels,
            experiment_channels={"Search"},
            learning=pd.DataFrame(
                [
                    {
                        "parameter": "beta_TV",
                        "contraction": 0.01,
                        "verdict": "prior-dominated",
                    },
                    {
                        "parameter": "beta_Search",
                        "contraction": 0.7,
                        "verdict": "strong",
                    },
                    {
                        "parameter": "beta_Shopping",
                        "contraction": 0.6,
                        "verdict": "strong",
                    },
                ]
            ),
            collinearity={
                "Search": {"vif": 40.0, "collinear_with": ["Shopping"]},
                "Shopping": {"vif": 40.0, "collinear_with": ["Search"]},
            },
        ).items()
    }
    roi = {
        ch: {
            "mean": 2.0,
            "lower": 1.5,
            "upper": 2.5,
            "is_monetary": True,
            "reference": 1.0,
            "value_units": "ROI",
            "evidence": ev[ch],
        }
        for ch in channels
    }
    return MMMDataBundle(
        channel_names=channels,
        channel_roi=roi,
        channel_spend={ch: 100.0 for ch in channels},
        current_spend={ch: 100.0 for ch in channels},
        channel_evidence=ev,
    )


class TestClassicRender:
    def test_classic_report_shows_chips_and_legend(self):
        from mmm_framework.reporting import MMMReportGenerator, ReportConfig

        html = MMMReportGenerator(
            data=_bundle_with_evidence(), config=ReportConfig()
        ).render()
        assert "evidence-chip ev-prior" in html  # TV prior-dominated
        assert "evidence-chip ev-experiment" in html  # Search validated
        assert "not separately identified" in html  # Shopping collinear
        assert "How to read the evidence column" in html


class TestAugurRender:
    def test_augur_report_shows_evidence_column(self):
        import dataclasses

        from mmm_framework.reporting import MMMReportGenerator, ReportConfig

        cfg = ReportConfig()
        cfg = dataclasses.replace(cfg, shell="augur")
        html = MMMReportGenerator(data=_bundle_with_evidence(), config=cfg).render()
        assert "tier-chip t-reduce" in html  # prior-dominated chip
        assert "not separately identified" in html
        # The evidence legend explains the provenance tiers.
        assert "evidence tier" in html


# ---------------------------------------------------------------------------
# Real-fit end-to-end (slow) — the experiment-validated tier only exists on a
# genuine fit with an in-graph experiment calibration.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_real_fit_experiment_validated_tier():
    from mmm_framework.calibration.likelihood import (
        ExperimentEstimand,
        ExperimentMeasurement,
    )
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
    from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

    periods = pd.date_range("2021-01-04", periods=48, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(11)
    t = np.arange(n)
    tv = np.abs(rng.normal(120, 30, n))
    search = np.abs(rng.normal(60, 20, n))
    y = pd.Series(
        1000
        + 8 * t
        + 50 * np.sin(2 * np.pi * t / 52)
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
    cfg = MFFConfig(
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
        config=cfg,
    )
    model = BayesianMMM(
        panel,
        ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=2,
            n_draws=120,
            n_tune=120,
        ),
        TrendConfig(type=TrendType.LINEAR),
    )
    model.add_experiment_calibration(
        [
            ExperimentMeasurement(
                channel="Search",
                test_period=(periods[10], periods[20]),
                value=2.2,
                se=0.3,
                estimand=ExperimentEstimand.ROAS,
            )
        ]
    )
    results = model.fit(random_seed=0)
    bundle = BayesianMMMExtractor(model, results).extract()

    assert bundle.channel_evidence is not None
    assert bundle.channel_evidence["Search"]["tier"] == "experiment-validated"
    assert bundle.channel_evidence["Search"]["experiment"] is True
    # TV had no experiment → model-identified or prior-dominated (data-derived).
    assert bundle.channel_evidence["TV"]["tier"] in (
        "model-identified",
        "prior-dominated",
    )
    # Stamped onto channel_roi so the report renders it.
    assert bundle.channel_roi["Search"]["evidence"]["tier"] == "experiment-validated"
