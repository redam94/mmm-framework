"""Per-cell adstock for geo / geo x product panels.

A panel model stacks ``n_periods x n_cells`` observations into one vector
(period-major from the MFF loader). Adstock — both the legacy two-point blend
precompute and the parametric in-graph kernels — must convolve along each
cross-section cell's *own* time axis. Before the per-cell fix, the convolution
ran straight down the stacked vector, so one geography's spend carried over
into *other geographies'* rows (with period-major ordering, an observation's
"lag 1" is a different geo at the same week).

The contract tested here: an impulse of spend in one cell produces zero media
contribution in every other cell, in every adstock family, and the panel-aware
convolution agrees exactly with the trusted 1-D kernel applied per cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pytest

from mmm_framework.config import (
    AdstockConfig,
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model import BayesianMMM, TrendConfig, TrendType
from mmm_framework.transforms import geometric_adstock_2d
from mmm_framework.transforms.adstock_pt import (
    adstock_weights_pt,
    apply_adstock_panel_pt,
    apply_adstock_pt,
)

# ---------------------------------------------------------------------------
# transform level
# ---------------------------------------------------------------------------


def _eval(t):
    return pytensor.function([], t)()


@pytest.mark.parametrize("kind", ["geometric", "delayed", "weibull"])
@pytest.mark.parametrize("ordering", ["period_major", "cell_major"])
def test_panel_convolution_matches_per_cell_1d(kind, ordering):
    """Panel-aware convolution == trusted 1-D kernel applied to each cell."""
    rng = np.random.default_rng(0)
    n_periods, n_cells, l_max = 40, 3, 8
    series = rng.exponential(1.0, size=(n_periods, n_cells))

    if ordering == "period_major":
        time_idx = np.repeat(np.arange(n_periods), n_cells)
        cell_idx = np.tile(np.arange(n_cells), n_periods)
    else:
        time_idx = np.tile(np.arange(n_periods), n_cells)
        cell_idx = np.repeat(np.arange(n_cells), n_periods)
    x_stacked = series[time_idx, cell_idx]

    weights = adstock_weights_pt(
        kind, l_max, alpha=0.6, theta=2.0, shape=2.0, scale=3.0
    )
    got = _eval(
        apply_adstock_panel_pt(
            pt.as_tensor_variable(x_stacked),
            weights,
            l_max,
            time_idx=time_idx,
            cell_idx=cell_idx,
            n_periods=n_periods,
            n_cells=n_cells,
        )
    )
    want = np.empty_like(x_stacked)
    for k in range(n_cells):
        want[cell_idx == k] = _eval(
            apply_adstock_pt(pt.as_tensor_variable(series[:, k]), weights, l_max)
        )
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_panel_convolution_single_cell_matches_1d():
    """With one cell the panel path reduces to the historical 1-D result."""
    rng = np.random.default_rng(1)
    n, l_max = 60, 8
    x = rng.exponential(1.0, n)
    weights = adstock_weights_pt("geometric", l_max, alpha=0.5)
    got = _eval(
        apply_adstock_panel_pt(
            pt.as_tensor_variable(x),
            weights,
            l_max,
            time_idx=np.arange(n),
            cell_idx=np.zeros(n, dtype=int),
            n_periods=n,
            n_cells=1,
        )
    )
    want = _eval(apply_adstock_pt(pt.as_tensor_variable(x), weights, l_max))
    np.testing.assert_allclose(got, want, rtol=1e-10)


# ---------------------------------------------------------------------------
# model level
# ---------------------------------------------------------------------------


def _impulse_panel(geos, products=None, n_periods=30, impulse_cell=0, impulse_week=5):
    """A panel where TV spends once, in one cell only; everyone else is dark."""
    periods = pd.date_range("2023-01-02", periods=n_periods, freq="W-MON")
    if products:
        index = pd.MultiIndex.from_product(
            [periods, geos, products], names=["Period", "Geography", "Product"]
        )
        kpi_dims = [
            DimensionType.PERIOD,
            DimensionType.GEOGRAPHY,
            DimensionType.PRODUCT,
        ]
    else:
        index = pd.MultiIndex.from_product(
            [periods, geos], names=["Period", "Geography"]
        )
        kpi_dims = [DimensionType.PERIOD, DimensionType.GEOGRAPHY]

    n_cells = len(geos) * (len(products) if products else 1)
    cell_of = np.tile(np.arange(n_cells), n_periods)
    week_of = np.repeat(np.arange(n_periods), n_cells)
    tv = np.where((cell_of == impulse_cell) & (week_of == impulse_week), 100.0, 0.0)

    y = pd.Series(
        np.random.default_rng(0).normal(500.0, 5.0, len(index)),
        index=index,
        name="Sales",
    )
    X_media = pd.DataFrame({"TV": tv}, index=index)
    coords = PanelCoordinates(
        periods=periods,
        geographies=list(geos),
        products=list(products) if products else None,
        channels=["TV"],
        controls=[],
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=kpi_dims),
        media_channels=[MediaChannelConfig(name="TV", dimensions=kpi_dims)],
    )
    panel = PanelDataset(
        y=y, X_media=X_media, X_controls=None, coords=coords, index=index, config=config
    )
    dark = cell_of != impulse_cell
    return panel, dark


def _prior_contributions(panel, *, parametric, adstock=None):
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_draws=10,
        n_tune=10,
        n_chains=1,
        use_parametric_adstock=parametric,
    )
    if adstock is not None:
        for mc in panel.config.media_channels:
            mc.adstock = adstock
    mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))

    import pymc as pm

    with mmm.model:
        prior = pm.sample_prior_predictive(draws=25, random_seed=1)
    cc = prior.prior["channel_contributions"].values  # (chain, draw, obs, channel)
    return mmm, np.abs(cc).reshape(-1, cc.shape[2]).max(axis=0)


@pytest.mark.parametrize(
    "adstock",
    [
        AdstockConfig.geometric(),
        AdstockConfig.delayed(),
        AdstockConfig.weibull(l_max=12),
    ],
    ids=["geometric", "delayed", "weibull"],
)
def test_geo_impulse_does_not_bleed_parametric(adstock):
    """A geo that never spends gets exactly zero media contribution (in-graph)."""
    panel, dark = _impulse_panel(["East", "West"])
    _, max_abs = _prior_contributions(panel, parametric=True, adstock=adstock)
    assert max_abs[dark].max() < 1e-12, "carryover bled into a never-spending geo"
    assert max_abs[~dark].max() > 1e-6, "impulse geo should carry a contribution"


def test_geo_impulse_does_not_bleed_legacy_blend():
    """Legacy two-point-blend path: precomputed adstock is per-geo too."""
    panel, dark = _impulse_panel(["East", "West"])
    cfg = ModelConfig(
        inference_method=InferenceMethod.BAYESIAN_PYMC,
        n_draws=10,
        n_tune=10,
        n_chains=1,
        use_parametric_adstock=False,
    )
    mmm = BayesianMMM(panel, cfg, TrendConfig(type=TrendType.LINEAR))
    for alpha, X in mmm.X_media_adstocked.items():
        assert np.abs(X[dark]).max() < 1e-12, f"alpha={alpha}: bleed into dark geo"
        assert np.abs(X[~dark]).max() > 1e-6


def test_geo_product_impulse_isolated_to_one_cell():
    """geo x product: the impulse stays inside its (geo, product) cell."""
    panel, dark = _impulse_panel(["East", "West"], products=["Core", "Premium"])
    mmm, max_abs = _prior_contributions(panel, parametric=True)
    assert mmm.n_cells == 4
    assert max_abs[dark].max() < 1e-12
    assert max_abs[~dark].max() > 1e-6


def test_national_per_cell_helper_is_identity_with_one_cell():
    """National models: per-cell helper returns exactly geometric_adstock_2d."""
    periods = pd.date_range("2023-01-02", periods=40, freq="W-MON")
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"TV": rng.exponential(50.0, 40)}, index=periods)
    y = pd.Series(rng.normal(500.0, 5.0, 40), index=periods, name="Sales")
    coords = PanelCoordinates(
        periods=periods, geographies=None, products=None, channels=["TV"], controls=[]
    )
    config = MFFConfig(
        kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
        media_channels=[
            MediaChannelConfig(name="TV", dimensions=[DimensionType.PERIOD])
        ],
    )
    panel = PanelDataset(
        y=y, X_media=X, X_controls=None, coords=coords, index=periods, config=config
    )
    mmm = BayesianMMM(
        panel,
        ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_draws=10,
            n_tune=10,
            n_chains=1,
        ),
        TrendConfig(type=TrendType.LINEAR),
    )
    assert mmm.n_cells == 1
    raw = X.to_numpy(float)
    np.testing.assert_array_equal(
        mmm._geometric_adstock_per_cell(raw, 0.5), geometric_adstock_2d(raw, 0.5)
    )
