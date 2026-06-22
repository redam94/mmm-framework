"""End-to-end proof of Phase 2 on the awareness garden model.

Demonstrates the per-model config schema + pluggable likelihood working together:
a bespoke ``CONFIG_SCHEMA`` (``number_of_trials`` etc.) drives the model, and a
``binomial`` likelihood (the model writes its own ``pm.Binomial``) turns the KPI
into a survey count — fit end-to-end, with the ``awareness_lift`` estimand (from
Phase 1) reported on top. The default Normal mode is also exercised.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    DimensionType,
    KPIConfig,
    LikelihoodConfig,
    LikelihoodFamily,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import TrendConfig
from mmm_framework.model.trend_config import TrendType

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/garden_models")
    ),
)
from awareness_structural_mmm import (
    AwarenessParams,
    AwarenessStructuralMMM,
)  # noqa: E402

N_TRIALS = 1000


def _awareness_panel(binomial: bool):
    """A national awareness series driven by two channels. For the binomial KPI,
    ``y`` is an integer count of aware respondents out of ``N_TRIALS``."""
    periods = pd.date_range("2021-01-04", periods=40, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(11)
    tv = np.abs(rng.normal(100, 25, n))
    digital = np.abs(rng.normal(80, 20, n))
    # Latent aware-rate driven by (normalized) media.
    logit = -0.4 + 1.3 * (tv / tv.max()) + 0.7 * (digital / digital.max())
    rate = 1.0 / (1.0 + np.exp(-logit))
    if binomial:
        y = pd.Series(rng.binomial(N_TRIALS, rate), name="Awareness").astype(float)
    else:
        y = pd.Series(100 * rate + rng.normal(0, 2, n), name="Awareness")
    config = MFFConfig(
        kpi=KPIConfig(name="Awareness", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    return PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": digital}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV", "Digital"],
            controls=None,
        ),
        index=periods,
        config=config,
    )


def _model(binomial: bool, **param_overrides):
    likelihood = (
        LikelihoodConfig.binomial(n_trials=N_TRIALS)
        if binomial
        else LikelihoodConfig.normal()
    )
    return AwarenessStructuralMMM(
        _awareness_panel(binomial),
        ModelConfig(likelihood=likelihood),
        TrendConfig(type=TrendType.NONE),
        model_params=param_overrides or None,
    )


class TestConfigSchema:
    def test_defaults_tracked(self):
        mmm = _model(binomial=False)
        assert isinstance(mmm.model_params, AwarenessParams)
        assert mmm.model_params.retention_prior_alpha == 6.0
        assert mmm.model_params.number_of_trials == 1000

    def test_overrides_validated(self):
        mmm = _model(binomial=False, retention_prior_alpha=9.0, number_of_trials=500)
        assert mmm.model_params.retention_prior_alpha == 9.0
        assert mmm.model_params.number_of_trials == 500

    def test_bad_param_rejected(self):
        with pytest.raises(Exception):  # gt=0 validator
            _model(binomial=False, number_of_trials=-1)


class TestNormalMode:
    def test_builds_normal_and_standardizes(self):
        mmm = _model(binomial=False)
        ops = {type(v.owner.op).__name__ for v in mmm.model.observed_RVs}
        assert any("Normal" in o for o in ops)
        assert mmm._standardizes_y and mmm.y_std > 1.0


class TestBinomialMode:
    def test_builds_binomial_and_keeps_natural_scale(self):
        mmm = _model(binomial=True)
        # Binomial family -> y is NOT standardized (raw counts; y_std==1).
        assert not mmm._standardizes_y
        assert mmm.y_std == 1.0
        assert np.allclose(mmm.y, mmm.y_raw)
        rv = next(v for v in mmm.model.observed_RVs if v.name == "y_obs")
        assert "Binomial" in type(rv.owner.op).__name__
        assert "awareness_rate" in {d.name for d in mmm.model.deterministics}

    @pytest.mark.slow
    def test_fit_and_awareness_lift_estimand(self):
        # The full proof: config + binomial likelihood + Phase-1 estimand, fit.
        mmm = _model(binomial=True, number_of_trials=N_TRIALS)
        mmm.fit(method="map", random_seed=0)
        assert mmm._trace is not None
        assert "awareness_retention" in mmm._trace.posterior

        # awareness_lift zeros the media block per channel -> expands to
        # "awareness_lift:<channel>"; each should evaluate to a finite mean lift.
        results = mmm.evaluate_estimands(["awareness_lift"])
        lifts = {k: v for k, v in results.items() if k.startswith("awareness_lift")}
        assert lifts, f"no awareness_lift results; got {list(results)}"
        for k, v in lifts.items():
            assert v.status == "ok", f"{k}: {v.reason}"
            assert np.isfinite(v.mean), k

    @pytest.mark.slow
    def test_default_estimands_resolve_from_names(self):
        # The model declares DEFAULT_ESTIMANDS as names; they resolve + evaluate.
        mmm = _model(binomial=True)
        mmm.fit(method="map", random_seed=0)
        results = mmm.evaluate_estimands()  # None -> DEFAULT_ESTIMANDS
        assert any(k.startswith("awareness_lift") for k in results)
        assert any(k.startswith("contribution_roi") for k in results)
