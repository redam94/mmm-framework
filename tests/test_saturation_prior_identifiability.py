"""The saturation prior, and what it can and cannot be asked to do (#207, #203).

Saturation is not identified from the sales likelihood. That is settled — Jin et
al. (Google, 2017) call the Hill parameters "essentially unidentifiable"; Dew et
al. (2024) prove predictive fit cannot arbitrate between observationally
equivalent response curves. When a parameter is unidentified **the prior is the
answer**, so these tests are about whether the prior is defensible, not whether
the fit recovers a truth.

The defect this fixes: media reaches saturation normalized by the channel's
training maximum, so ``sat_lam`` *is* the elbow position — half-saturation at
``ln(2)/lam`` in units of that maximum. The historical ``Exponential(lam=0.5)``
was never reparameterized after that normalization, leaving its mode at "this
channel does not saturate at all" and 29.3% of its mass beyond any spend we
observed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mmm_framework.config import (
    AdstockConfig,
    ControlVariableConfig,
    DimensionType,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
    PriorConfig,
    SaturationConfig,
)
from mmm_framework.config.enums import SaturationType
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.diagnostics.saturation import (
    ELBOW_MASS_UNANCHORED,
    saturation_learning,
    saturation_prior_report,
    warn_if_saturation_prior_is_unanchored,
)
from mmm_framework.model import BayesianMMM
from mmm_framework.model.base import (
    DEFAULT_LOGISTIC_ELBOW_FRACTION,
    DEFAULT_LOGISTIC_LAM_SIGMA,
)
from mmm_framework.model.trend_config import TrendConfig, TrendType

CHANNELS = ["TV", "Digital"]

#: The pre-1.3 default. `PriorType` has no Exponential member, but Gamma(1, rate)
#: IS the Exponential — which is also the documented way to restore it.
LEGACY_LAM_PRIOR = PriorConfig(distribution="Gamma", params={"alpha": 1.0, "beta": 0.5})


def _panel(n_periods: int = 104) -> PanelDataset:
    periods = pd.date_range("2020-01-06", periods=n_periods, freq="W-MON")
    rng = np.random.default_rng(42)
    n = n_periods
    tv = np.abs(rng.standard_normal(n) * 50 + 100)
    tv[rng.random(n) < 0.25] = 0.0
    mff = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name=c, dimensions=[DimensionType.PERIOD])
            for c in CHANNELS
        ],
        controls=[
            ControlVariableConfig(name="Price", dimensions=[DimensionType.PERIOD])
        ],
    )
    return PanelDataset(
        y=pd.Series(1000 + rng.standard_normal(n) * 100, name="Sales"),
        X_media=pd.DataFrame(
            {"TV": tv, "Digital": np.abs(rng.standard_normal(n) * 30 + 80)}
        ),
        X_controls=pd.DataFrame({"Price": 10 + rng.standard_normal(n) * 0.5}),
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=CHANNELS,
            controls=["Price"],
        ),
        index=periods,
        config=mff,
    )


def _model(sat: SaturationConfig, *, model_config=None, adstock=None) -> BayesianMMM:
    panel = _panel()
    base = panel.config
    panel.config = MFFConfig(
        kpi=base.kpi,
        media_channels=[
            MediaChannelConfig(
                name=c,
                dimensions=[DimensionType.PERIOD],
                adstock=adstock or AdstockConfig.geometric(),
                saturation=sat,
            )
            for c in CHANNELS
        ],
        controls=base.controls,
    )
    return BayesianMMM(
        panel, model_config or ModelConfig(), TrendConfig(type=TrendType.LINEAR)
    )


def _families(model) -> dict[str, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = model.model
    return {
        rv.name: rv.owner.op.__class__.__name__.replace("RV", "")
        for rv in graph.free_RVs
    }


class TestTheDefaultPriorIsStatedInDataUnits:
    def test_default_logistic_prior_is_lognormal_on_the_elbow(self):
        fam = _families(_model(SaturationConfig.logistic()))
        assert fam["sat_lam_TV"] == "LogNormal"

    def test_the_median_elbow_sits_at_the_configured_fraction_of_max_spend(self):
        """`sat_lam` IS the elbow: half-saturation is ln(2)/lam on normalized media."""
        mu = float(np.log(np.log(2.0) / DEFAULT_LOGISTIC_ELBOW_FRACTION))
        median_lam = float(np.exp(mu))
        assert np.log(2.0) / median_lam == pytest.approx(
            DEFAULT_LOGISTIC_ELBOW_FRACTION
        )

    def test_far_less_prior_mass_lands_beyond_observed_spend(self):
        """The defect: an elbow beyond max spend is unlearnable from any data.

        Old default `Exponential(0.5)`: P(lam < ln2) = 29.3%.
        New default: ~4%.
        """
        legacy = float(stats.expon(scale=1 / 0.5).cdf(np.log(2.0)))
        mu = float(np.log(np.log(2.0) / DEFAULT_LOGISTIC_ELBOW_FRACTION))
        new = float(
            stats.lognorm(s=DEFAULT_LOGISTIC_LAM_SIGMA, scale=np.exp(mu)).cdf(
                np.log(2.0)
            )
        )
        assert legacy == pytest.approx(0.293, abs=0.002)
        assert new < 0.06
        assert new < legacy / 4

    def test_the_old_prior_is_still_reachable_as_documented(self):
        """The escape hatch must actually work — PriorType has no Exponential,
        so the docs name Gamma(1, rate), which is the same distribution."""
        fam = _families(_model(SaturationConfig.logistic(lam_prior=LEGACY_LAM_PRIOR)))
        assert fam["sat_lam_TV"] == "Gamma"

    def test_gamma_one_really_is_the_exponential(self):
        g = stats.gamma(a=1.0, scale=1 / 0.5)
        e = stats.expon(scale=1 / 0.5)
        xs = np.linspace(0.01, 12, 50)
        np.testing.assert_allclose(g.cdf(xs), e.cdf(xs), rtol=1e-12)


class TestHillAnchoring:
    def test_anchoring_is_opt_in_and_off_by_default(self):
        fam = _families(_model(SaturationConfig.hill()))
        assert fam["sat_half_TV"] == "Beta"
        assert "sat_half_raw_TV" not in fam

    def test_anchoring_confines_the_elbow_to_observed_spend(self):
        model = _model(
            SaturationConfig(type=SaturationType.HILL, anchor_kappa_to_data=True)
        )
        fam = _families(model)
        # The anchored elbow is a Deterministic over a raw Beta.
        assert "sat_half_raw_TV" in fam
        report = saturation_prior_report(model, draws=1500)
        row = report.set_index("channel").loc["TV"]
        assert row["elbow_q95"] <= 1.0
        assert row["verdict"] == "anchored"

    def test_an_explicit_kappa_prior_still_wins(self):
        model = _model(
            SaturationConfig(
                type=SaturationType.HILL,
                anchor_kappa_to_data=True,
                kappa_prior=PriorConfig(
                    distribution="Beta", params={"alpha": 5.0, "beta": 5.0}
                ),
            )
        )
        fam = _families(model)
        assert fam["sat_half_TV"] == "Beta"
        assert "sat_half_raw_TV" not in fam


class TestTheDiagnosticSaysWhenThePriorIsTheAnswer:
    def test_the_new_default_reports_anchored(self):
        report = saturation_prior_report(
            _model(SaturationConfig.logistic()), draws=3000
        )
        assert set(report["channel"]) == set(CHANNELS)
        assert (report["verdict"] == "anchored").all(), report

    def test_the_old_default_reports_unanchored_and_warns(self):
        """This is the check that would have caught #207 before it shipped."""
        model = _model(SaturationConfig.logistic(lam_prior=LEGACY_LAM_PRIOR))
        report = saturation_prior_report(model, draws=4000)
        assert (report["mass_beyond_support"] > ELBOW_MASS_UNANCHORED).all(), report
        assert (report["verdict"] == "unanchored").all()

        with pytest.warns(UserWarning, match="BEYOND"):
            warn_if_saturation_prior_is_unanchored(model, draws=2000)

    def test_the_anchored_default_does_not_warn(self):
        """Safe to call unconditionally: silent when there is nothing to say."""
        model = _model(SaturationConfig.logistic())
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_if_saturation_prior_is_unanchored(model, draws=2000)

    def test_families_without_an_elbow_are_omitted_not_guessed_at(self):
        """`root` has no asymptote, so it has no half-saturation point."""
        report = saturation_prior_report(_model(SaturationConfig.root()), draws=500)
        assert report.empty

    def test_learning_requires_a_fitted_model(self):
        with pytest.raises(ValueError, match="not fitted"):
            saturation_learning(_model(SaturationConfig.logistic()))


@pytest.mark.slow
class TestPostFitLearning:
    def test_contraction_is_reported_for_the_saturation_block(self):
        model = _model(SaturationConfig.logistic())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(method="map", random_seed=0, progressbar=False)
        frame = saturation_learning(model, draws=800)
        assert set(frame["parameter"]) == {"sat_lam_TV", "sat_lam_Digital"}
        assert "contraction" in frame.columns
        assert "verdict" in frame.columns


@pytest.mark.slow
class TestMapBoundaryDiagnostic:
    """#203 — an interval transform saturating under find_MAP.

    `find_MAP` optimizes in the unconstrained space; float64 `sigmoid` returns
    exactly 1.0 past ~37, so a [0,1]-bounded parameter can land exactly on its
    boundary where the Beta gradient `-2/(1-a)` divides by zero. The raw failure
    is a bare ZeroDivisionError from a JIT-compiled Composite op.

    The #207 prior change incidentally stopped this particular configuration from
    crashing — the optimizer takes a different path — but the failure MODE is
    unchanged, which is why the diagnostic is tested against the old prior rather
    than deleted along with the repro.
    """

    def test_the_message_names_the_mechanism_and_the_ways_out(self):
        built = _model(SaturationConfig.logistic(lam_prior=LEGACY_LAM_PRIOR))
        model = BayesianMMM(
            built.panel,
            ModelConfig(media_prior_mode="roi"),
            TrendConfig(type=TrendType.NONE),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ZeroDivisionError) as exc:
                model.fit(method="map", random_seed=0, progressbar=False)
        message = str(exc.value)
        assert "boundary" in message
        assert "adstock_alpha_TV" in message
        assert "advi" in message
        assert "issues/203" in message

    def test_the_anchored_default_no_longer_hits_it_on_this_configuration(self):
        built = _model(SaturationConfig.logistic())
        model = BayesianMMM(
            built.panel,
            ModelConfig(media_prior_mode="roi"),
            TrendConfig(type=TrendType.NONE),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(method="map", random_seed=0, progressbar=False)
        assert model._trace is not None
