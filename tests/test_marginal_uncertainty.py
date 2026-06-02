"""Tests for marginal-ROAS uncertainty (P1-3, critique.md §3.9).

The headline marginal-efficiency number must carry a credible interval, not be a
bare point estimate. Fast tests cover the result container's new optional fields;
the slow test fits a small model and checks the posterior interval is real,
ordered, brackets the point estimate, and survives a zero-spend channel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.analysis import MarginalAnalysisResult
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


class TestMarginalResultFields:
    def test_hdi_fields_default_to_none(self):
        # Backward compatibility: existing positional/keyword construction works
        # and the new interval fields are optional.
        r = MarginalAnalysisResult(
            channel="TV",
            current_spend=1000.0,
            spend_increase=100.0,
            spend_increase_pct=10.0,
            marginal_contribution=50.0,
            marginal_roas=0.5,
        )
        assert r.marginal_roas_hdi_low is None
        assert r.marginal_roas_hdi_high is None
        assert r.marginal_contribution_hdi_low is None
        assert r.marginal_contribution_hdi_high is None
        assert r.hdi_prob is None

    def test_hdi_fields_settable(self):
        r = MarginalAnalysisResult(
            channel="TV",
            current_spend=1000.0,
            spend_increase=100.0,
            spend_increase_pct=10.0,
            marginal_contribution=50.0,
            marginal_roas=0.5,
            marginal_roas_hdi_low=0.2,
            marginal_roas_hdi_high=0.8,
            hdi_prob=0.94,
        )
        assert r.marginal_roas_hdi_low == 0.2
        assert r.marginal_roas_hdi_high == 0.8
        assert r.hdi_prob == 0.94


@pytest.fixture
def fitted_model():
    periods = pd.date_range("2021-01-04", periods=40, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(5)
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV", "Digital"],
        controls=None,
    )
    t = np.arange(n)
    tv = np.abs(rng.normal(100, 25, n))
    digital = np.abs(rng.normal(80, 20, n))
    y = pd.Series(
        1000
        + 10.0 * t
        + 50.0 * np.sin(2 * np.pi * t / 52)
        + 1.0 * tv
        + 0.5 * digital
        + rng.normal(0, 20, n),
        name="Sales",
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": digital}),
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
            target_accept=0.85,
        ),
        TrendConfig(type=TrendType.LINEAR),
    )
    model.fit(random_seed=0)
    return model


@pytest.mark.slow
class TestMarginalUncertaintyIntegration:
    def test_hdi_columns_present_and_ordered(self, fitted_model):
        df = fitted_model.compute_marginal_contributions(
            spend_increase_pct=10.0, hdi_prob=0.9, random_seed=0
        )
        for col in (
            "Marginal ROAS HDI Low",
            "Marginal ROAS HDI High",
            "Marginal Contribution HDI Low",
            "Marginal Contribution HDI High",
        ):
            assert col in df.columns

        for _, row in df.iterrows():
            assert row["Marginal ROAS HDI Low"] <= row["Marginal ROAS HDI High"]
            # The point estimate must lie inside its own credible interval.
            assert (
                row["Marginal ROAS HDI Low"]
                <= row["Marginal ROAS"]
                <= row["Marginal ROAS HDI High"]
            )
            assert (
                row["Marginal Contribution HDI Low"]
                <= row["Marginal Contribution"]
                <= row["Marginal Contribution HDI High"]
            )
            # A real interval, not a collapsed point.
            assert row["Marginal ROAS HDI High"] > row["Marginal ROAS HDI Low"]

    def test_compute_uncertainty_false_is_backward_compatible(self, fitted_model):
        df = fitted_model.compute_marginal_contributions(
            spend_increase_pct=10.0, compute_uncertainty=False, random_seed=0
        )
        assert "Marginal ROAS" in df.columns
        assert "Marginal ROAS HDI Low" not in df.columns

    def test_zero_spend_channel_does_not_poison_hdi(self, fitted_model):
        # Force one channel to zero spend so spend_increase == 0; the per-draw
        # division must be guarded (no inf/nan in the HDI).
        fitted_model.X_media_raw[:, fitted_model.channel_names.index("Digital")] = 0.0
        df = fitted_model.compute_marginal_contributions(
            spend_increase_pct=10.0, random_seed=0
        )
        digital = df[df["Channel"] == "Digital"].iloc[0]
        assert np.isfinite(digital["Marginal ROAS HDI Low"])
        assert np.isfinite(digital["Marginal ROAS HDI High"])
        assert digital["Marginal ROAS HDI Low"] == 0.0
        assert digital["Marginal ROAS HDI High"] == 0.0
