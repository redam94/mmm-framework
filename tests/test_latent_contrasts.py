"""Latent-variable contrasts in the estimand engine: a named deterministic
re-evaluated under an intervention vs a baseline (via the model's
``sample_latent_under``), reduced and combined by the contrast op. Unit-tested
with a stub model, plus an end-to-end check on the awareness garden model whose
``media_total`` goodwill stock responds to media interventions."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from mmm_framework.estimands.evaluate import EstimandEvaluator
from mmm_framework.estimands.spec import (
    Contrast,
    Estimand,
    LatentVar,
    Observed,
    ZeroInput,
)


def _latent_contrast_estimand(var="L", target="TV", reduce="mean", op="difference"):
    return Estimand(
        name="latent_lift",
        kind="latent_lift",
        numerator=Contrast(
            quantity=LatentVar(name=var),
            intervention=Observed(),
            baseline=ZeroInput(target=target),
            op=op,
            reduce=reduce,
        ),
        denominator=None,
        required_capabilities=[f"HAS_LATENT:{var}"],
        units="native",
    )


class _StubModel:
    """A model whose latent ``L`` is 1.0 per obs under the factual world and 0.0
    when a channel is zeroed — so a difference/mean contrast equals 1.0."""

    n_obs = 8

    def __init__(self):
        post = xr.Dataset(
            {"L": (("chain", "draw", "obs"), np.ones((2, 20, self.n_obs)))}
        )
        self._trace = SimpleNamespace(posterior=post)
        self.trace = self._trace
        self.channel_names = ["TV", "Digital"]

    def _get_time_mask(self, tp):
        return np.ones(self.n_obs, dtype=bool)

    def sample_latent_under(self, var_name, intervention=None, random_seed=None):
        kind = getattr(intervention, "type", "observed")
        fill = 0.0 if kind in ("zero_input", "scale_input", "set_input") else 1.0
        return np.full((40, self.n_obs), fill)  # (n_samples, n_obs)


class TestLatentContrastUnit:
    def test_difference_mean(self):
        model = _StubModel()
        res = EstimandEvaluator(model).evaluate([_latent_contrast_estimand()])
        r = res["latent_lift:TV"] if "latent_lift:TV" in res else res["latent_lift"]
        assert r.status == "ok"
        assert abs(r.mean - 1.0) < 1e-9  # (1.0 - 0.0) averaged over obs

    def test_higher_dim_latent_unsupported(self):
        model = _StubModel()

        def _sample(var_name, intervention=None, random_seed=None):
            return np.ones((40, model.n_obs, 3))  # (n_samples, n_obs, k) -> 3-D

        model.sample_latent_under = _sample
        res = EstimandEvaluator(model).evaluate([_latent_contrast_estimand()])
        r = next(iter(res.values()))
        assert r.status == "unsupported"
        assert "trailing shape" in (r.reason or "")

    def test_no_sample_latent_under_degrades(self):
        model = _StubModel()
        model.sample_latent_under = None  # model can't realize latent contrasts
        res = EstimandEvaluator(model).evaluate([_latent_contrast_estimand()])
        r = next(iter(res.values()))
        assert r.status == "unsupported"
        assert "sample_latent_under" in (r.reason or "")


@pytest.mark.slow
def test_awareness_latent_contrast_end_to_end():
    """The awareness model's latent goodwill stock (`media_total`) under
    media-on vs a channel zeroed — a real latent contrast through the engine."""
    import pandas as pd

    sys.path.insert(
        0,
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../examples/garden_models")
        ),
    )
    from awareness_structural_mmm import AwarenessStructuralMMM

    from mmm_framework.config import (
        DimensionType,
        KPIConfig,
        MediaChannelConfig,
        MFFConfig,
        ModelConfig,
    )
    from mmm_framework.data_loader import PanelCoordinates, PanelDataset
    from mmm_framework.model import TrendConfig
    from mmm_framework.model.trend_config import TrendType

    periods = pd.date_range("2021-01-04", periods=40, freq="W-MON")
    n = len(periods)
    rng = np.random.default_rng(5)
    t = np.arange(n)
    tv = np.abs(rng.normal(100, 25, n))
    dig = np.abs(rng.normal(80, 20, n))
    y = pd.Series(
        1000 + 10 * t + 1.0 * tv + 0.5 * dig + rng.normal(0, 20, n), name="Aware"
    )
    cfg = MFFConfig(
        kpi=KPIConfig(name="Aware", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD]),
            MediaChannelConfig(name="Digital", dimensions=[DimensionType.PERIOD]),
        ],
        controls=[],
    )
    panel = PanelDataset(
        y=y,
        X_media=pd.DataFrame({"TV": tv, "Digital": dig}),
        X_controls=None,
        coords=PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV", "Digital"],
            controls=None,
        ),
        index=periods,
        config=cfg,
    )
    mmm = AwarenessStructuralMMM(panel, ModelConfig(), TrendConfig(type=TrendType.NONE))
    mmm.fit(method="map", random_seed=0)

    # media_total is the goodwill stock (n_obs,) -> a valid latent for a contrast.
    assert (
        "media_total" in mmm.model_capabilities()
        or "HAS_LATENT:media_total" in mmm.model_capabilities()
    )
    res = mmm.evaluate_estimands(
        [_latent_contrast_estimand(var="media_total", target="TV")]
    )
    r = res["latent_lift:TV"] if "latent_lift:TV" in res else res["latent_lift"]
    assert r.status == "ok"
    assert np.isfinite(r.mean)
