"""Tests for P2-4: the extractor that auto-populates the CausalAssumptionsSection.

Two seams matter and are verified here against REAL objects (not hand-built
dicts): (1) UnobservedConfoundingSensitivity.to_dict() must serialize the
``is_fragile`` property so the section renders "Fragile"; (2) the BayesianMMM
extractor must surface the model's resolved confounder roles + robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    CausalControlRole,
    ControlVariableConfig,
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
from mmm_framework.reporting.extractors.bundle import MMMDataBundle
from mmm_framework.validation.results import (
    ChannelRobustness,
    UnobservedConfoundingSensitivity,
)


def _render(bundle) -> str:
    return MMMReportGenerator(data=bundle).render()


class TestRobustnessSeam:
    def test_real_to_dict_serializes_is_fragile_and_renders(self):
        # Build REAL ChannelRobustness objects (is_fragile is a *property*) and
        # round-trip through the actual to_dict() the extractor uses -- this is
        # the seam a synthetic-dict test would never catch.
        sens = UnobservedConfoundingSensitivity(
            channels=[
                ChannelRobustness(
                    channel="TV",
                    estimate=2.0,
                    std_error=0.1,
                    t_value=20.0,
                    dof=40,
                    partial_r2=0.90,
                    robustness_value=0.50,
                    robustness_value_half=0.40,
                ),
                ChannelRobustness(
                    channel="Weak",
                    estimate=0.05,
                    std_error=0.5,
                    t_value=0.1,
                    dof=40,
                    partial_r2=0.01,
                    robustness_value=0.02,  # < 0.10 threshold -> is_fragile
                    robustness_value_half=0.01,
                ),
            ],
            dof=40,
            q=1.0,
            caveat="OLS-analogy robustness value.",
        )
        rob = sens.to_dict()
        # The property must have been serialized as a key.
        assert rob["channels"][1]["is_fragile"] is True
        assert rob["channels"][0]["is_fragile"] is False

        bundle = MMMDataBundle(
            channel_names=["TV", "Weak"],
            causal_assumptions={"robustness": rob},
        )
        html = _render(bundle)
        assert "Robustness to Unobserved Confounding" in html
        assert "Fragile" in html  # Weak channel
        assert "Robust" in html  # TV channel


@pytest.fixture
def fitted_model_with_confounder():
    periods = pd.date_range("2021-01-04", periods=40, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(9)
    t = np.arange(n)
    tv = np.abs(rng.normal(100, 25, n))
    demand = rng.normal(0, 1, n)  # the confounder
    y = pd.Series(
        1000
        + 10.0 * t
        + 60.0 * np.sin(2 * np.pi * t / 52)
        + 1.2 * tv
        + 8.0 * demand
        + rng.normal(0, 20, n),
        name="Sales",
    )
    coords = PanelCoordinates(
        periods=periods,
        geographies=None,
        products=None,
        channels=["TV"],
        controls=["Demand"],
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
        ],
        controls=[
            ControlVariableConfig(
                name="Demand",
                dimensions=[DimensionType.PERIOD],
                causal_role=CausalControlRole.CONFOUNDER,
            )
        ],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv}),
        X_controls=pd.DataFrame({"Demand": demand}),
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
    return model


class TestExtractorEndToEnd:
    def test_unfitted_strategy_only(self):
        # No trace -> no robustness table, but the identification strategy text
        # (honest framing) is still produced.
        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        periods = pd.date_range("2021-01-04", periods=20, freq="W-MON")
        n = len(periods)
        rng = np.random.default_rng(2)
        coords = PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV"],
            controls=None,
        )
        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
            ],
            controls=[],
        )
        panel = PanelDataset(
            y=pd.Series(100 + rng.normal(0, 5, n), name="Sales"),
            X_media=pd.DataFrame({"TV": np.abs(rng.normal(100, 30, n))}),
            X_controls=None,
            coords=coords,
            index=periods,
            config=config,
        )
        model = BayesianMMM(
            panel, ModelConfig(inference_method=InferenceMethod.BAYESIAN_PYMC)
        )
        bundle = BayesianMMMExtractor(model).extract()
        ca = bundle.causal_assumptions
        assert ca is not None
        # No confounders designated -> the honest "rests entirely on no
        # unobserved confounding" framing.
        assert "no-unobserved-confounding assumption" in ca["identification_strategy"]
        assert "robustness" not in ca  # unfitted -> no table

    @pytest.mark.slow
    def test_fitted_model_populates_confounders_and_robustness(
        self, fitted_model_with_confounder
    ):
        from mmm_framework.reporting.extractors.bayesian import BayesianMMMExtractor

        bundle = BayesianMMMExtractor(fitted_model_with_confounder).extract()
        ca = bundle.causal_assumptions
        assert ca is not None
        # Resolved CONFOUNDER role surfaced as a designated confounder.
        assert ca["assumed_confounders"] == ["Demand"]
        assert "Demand" in ca["identification_strategy"]
        # Robustness table populated from a real fitted trace, with the
        # is_fragile key present for every channel.
        assert "robustness" in ca
        chans = ca["robustness"]["channels"]
        assert chans and all("is_fragile" in c for c in chans)
        assert {c["channel"] for c in chans} == {"TV"}

        html = _render(bundle)
        assert "Robustness to Unobserved Confounding" in html
        assert "Assumed Confounders" in html and "Demand" in html
