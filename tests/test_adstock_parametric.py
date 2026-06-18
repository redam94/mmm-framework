"""Tests for parametric adstock kernels (geometric / delayed / Weibull).

Covers the NumPy reference, the PyTensor in-graph version (which must match the
reference exactly), and the core BayesianMMM parametric adstock path driven by
``ModelConfig.use_parametric_adstock`` + per-channel ``AdstockConfig``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm_framework.config import (
    AdstockConfig,
    ControlVariableConfig,
    DimensionType,
    InferenceMethod,
    KPIConfig,
    MediaChannelConfig,
    MFFConfig,
    ModelConfig,
)
from mmm_framework.data_loader import PanelCoordinates, PanelDataset
from mmm_framework.model.base import BayesianMMM
from mmm_framework.transforms import adstock_weights, apply_adstock, parametric_adstock


# =============================================================================
# NumPy reference kernels
# =============================================================================


class TestAdstockWeights:
    def test_geometric_matches_alpha_powers(self):
        w = adstock_weights("geometric", 6, alpha=0.5, normalize=False)
        assert np.allclose(w, 0.5 ** np.arange(6))
        assert np.argmax(w) == 0  # peak at lag 0

    def test_geometric_monotonic_decay(self):
        w = adstock_weights("geometric", 8, alpha=0.7, normalize=True)
        assert np.all(np.diff(w) <= 0)

    def test_delayed_peaks_at_theta(self):
        for theta in (1, 3, 5):
            w = adstock_weights("delayed", 9, alpha=0.6, theta=theta)
            assert np.argmax(w) == theta

    def test_weibull_shape_controls_peak(self):
        w_delayed = adstock_weights("weibull", 12, shape=2.5, scale=3.0)
        w_front = adstock_weights("weibull", 12, shape=0.7, scale=2.0)
        assert np.isfinite(w_delayed).all()
        assert np.isfinite(w_front).all()
        assert np.argmax(w_delayed) > 0  # delayed peak
        assert np.argmax(w_front) == 0  # front-loaded

    def test_normalize_sums_to_one(self):
        for kind, kw in [
            ("geometric", dict(alpha=0.6)),
            ("delayed", dict(alpha=0.6, theta=2)),
            ("weibull", dict(shape=2.0, scale=2.5)),
        ]:
            w = adstock_weights(kind, 10, normalize=True, **kw)
            assert abs(w.sum() - 1.0) < 1e-9

    def test_none_is_unit_impulse(self):
        w = adstock_weights("none", 5)
        assert w[0] == 1.0
        assert np.all(w[1:] == 0.0)


class TestApplyAdstock:
    def test_pulse_response_is_kernel(self):
        x = np.array([100.0, 0, 0, 0, 0, 0])
        y = parametric_adstock(x, "geometric", l_max=6, alpha=0.5, normalize=False)
        assert np.allclose(y, 100 * 0.5 ** np.arange(6))

    def test_none_passthrough(self):
        x = np.array([3.0, 1, 4, 1, 5])
        y = parametric_adstock(x, "none", l_max=4)
        assert np.allclose(y, x)

    def test_causal_no_future_leakage(self):
        # A spike at t=3 must not affect outputs before t=3.
        x = np.zeros(8)
        x[3] = 1.0
        w = adstock_weights("delayed", 5, alpha=0.6, theta=2)
        y = apply_adstock(x, w)
        assert np.allclose(y[:3], 0.0)


# =============================================================================
# PyTensor kernels must match the NumPy reference exactly
# =============================================================================


class TestPyTensorMatchesNumpy:
    @pytest.mark.parametrize(
        "kind,kw",
        [
            ("geometric", dict(alpha=0.55)),
            ("delayed", dict(alpha=0.55, theta=2)),
            ("weibull", dict(shape=2.2, scale=3.0)),
            ("none", {}),
        ],
    )
    def test_pt_equals_np(self, kind, kw):
        import pytensor.tensor as pt

        from mmm_framework.transforms.adstock_pt import parametric_adstock_pt

        x = np.array([3.0, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 0])
        xt = pt.dvector("x")
        y_pt = parametric_adstock_pt(xt, kind, l_max=8, normalize=True, **kw).eval(
            {xt: x}
        )
        y_np = parametric_adstock(x, kind, l_max=8, normalize=True, **kw)
        assert np.allclose(y_pt, y_np, atol=1e-10)


# =============================================================================
# Core model parametric adstock path
# =============================================================================


@pytest.fixture
def panel_with_adstock():
    """Factory: national panel whose channels use the given AdstockConfigs."""

    def _make(tv_adstock: AdstockConfig, digital_adstock: AdstockConfig):
        periods = pd.date_range("2020-01-06", periods=40, freq="W-MON")
        n = len(periods)
        coords = PanelCoordinates(
            periods=periods,
            geographies=None,
            products=None,
            channels=["TV", "Digital"],
            controls=["Price"],
        )
        rng = np.random.default_rng(7)
        y = pd.Series(1000 + rng.standard_normal(n) * 80, name="Sales")
        X_media = pd.DataFrame(
            {
                "TV": np.abs(rng.standard_normal(n) * 50 + 100),
                "Digital": np.abs(rng.standard_normal(n) * 30 + 80),
            }
        )
        X_controls = pd.DataFrame({"Price": 10 + rng.standard_normal(n) * 0.5})
        config = MFFConfig(
            kpi=KPIConfig(name="Sales", dimensions=[DimensionType.PERIOD]),
            media_channels=[
                MediaChannelConfig(
                    name="TV", dimensions=[DimensionType.PERIOD], adstock=tv_adstock
                ),
                MediaChannelConfig(
                    name="Digital",
                    dimensions=[DimensionType.PERIOD],
                    adstock=digital_adstock,
                ),
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
            config=config,
        )

    return _make


def _free_rv_names(model):
    return {v.name for v in model.free_RVs}


class TestCoreParametricAdstock:
    def test_default_is_parametric(self, panel_with_adstock):
        panel = panel_with_adstock(AdstockConfig.geometric(), AdstockConfig.geometric())
        model = BayesianMMM(panel, ModelConfig()).model
        names = _free_rv_names(model)
        # Since 2026-06 the default estimates the continuous kernel in-graph.
        assert "adstock_alpha_TV" in names
        assert "adstock_TV" not in names

    def test_legacy_optout_uses_two_point_mix(self, panel_with_adstock):
        panel = panel_with_adstock(AdstockConfig.geometric(), AdstockConfig.geometric())
        model = BayesianMMM(panel, ModelConfig(use_parametric_adstock=False)).model
        names = _free_rv_names(model)
        # Legacy path samples the Beta mix and no continuous alpha.
        assert "adstock_TV" in names
        assert "adstock_alpha_TV" not in names

    def test_geometric_estimates_continuous_alpha(self, panel_with_adstock):
        panel = panel_with_adstock(
            AdstockConfig.geometric(l_max=6), AdstockConfig.geometric(l_max=6)
        )
        model = BayesianMMM(panel, ModelConfig(use_parametric_adstock=True)).model
        names = _free_rv_names(model)
        assert "adstock_alpha_TV" in names
        assert "adstock_alpha_Digital" in names
        assert "adstock_TV" not in names  # no Beta mix in parametric mode

    def test_delayed_adds_theta(self, panel_with_adstock):
        panel = panel_with_adstock(
            AdstockConfig.delayed(l_max=8), AdstockConfig.geometric(l_max=6)
        )
        model = BayesianMMM(panel, ModelConfig(use_parametric_adstock=True)).model
        names = _free_rv_names(model)
        assert "adstock_alpha_TV" in names
        assert "adstock_theta_TV" in names
        assert "adstock_theta_Digital" not in names  # geometric channel

    def test_weibull_adds_shape_and_scale(self, panel_with_adstock):
        panel = panel_with_adstock(
            AdstockConfig.weibull(l_max=10), AdstockConfig.geometric(l_max=6)
        )
        model = BayesianMMM(panel, ModelConfig(use_parametric_adstock=True)).model
        names = _free_rv_names(model)
        assert "adstock_shape_TV" in names
        assert "adstock_scale_TV" in names

    @pytest.mark.slow
    def test_parametric_model_samples(self, panel_with_adstock):
        panel = panel_with_adstock(
            AdstockConfig.delayed(l_max=6), AdstockConfig.weibull(l_max=8)
        )
        config = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            n_chains=1,
            n_draws=30,
            n_tune=30,
            use_parametric_adstock=True,
        )
        mmm = BayesianMMM(panel, config)
        results = mmm.fit(draws=30, tune=30, chains=1, progressbar=False)
        assert results is not None

    @pytest.mark.slow
    def test_delayed_recovers_delayed_peak(self):
        """The point of the feature: recover a peak geometric cannot represent.

        Generate the target from a known *delayed* kernel (peak at lag 3) and
        confirm the fitted delayed channel places its carryover peak away from
        lag 0 and near the true delay — something a geometric kernel (which
        always peaks at lag 0) structurally cannot do.
        """
        true_theta = 3
        l_max = 8
        periods = pd.date_range("2020-01-06", periods=120, freq="W-MON")
        n = len(periods)
        rng = np.random.default_rng(123)

        spend = np.abs(rng.standard_normal(n) * 40 + 60)
        spend_norm = spend / spend.max()
        w_true = adstock_weights("delayed", l_max, alpha=0.8, theta=true_theta)
        adstocked = apply_adstock(spend_norm, w_true)
        y = pd.Series(1000 + 300 * adstocked + rng.standard_normal(n) * 5, name="Sales")

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
                MediaChannelConfig(
                    name="TV",
                    dimensions=[DimensionType.PERIOD],
                    adstock=AdstockConfig.delayed(l_max=l_max),
                )
            ],
            controls=[],
        )
        panel = PanelDataset(
            y=y,
            X_media=pd.DataFrame({"TV": spend}),
            X_controls=None,
            coords=coords,
            index=periods,
            config=config,
        )

        model_config = ModelConfig(
            inference_method=InferenceMethod.BAYESIAN_PYMC,
            use_parametric_adstock=True,
        )
        mmm = BayesianMMM(panel, model_config)
        mmm.fit(draws=400, tune=400, chains=2, target_accept=0.9, progressbar=False)

        curves = mmm.compute_adstock_curves()
        assert curves is not None and "TV" in curves
        peak_lag = int(np.argmax(curves["TV"]))
        # The recovered carryover must peak away from lag 0, near the true delay.
        assert peak_lag >= 1, f"expected a delayed peak, got lag {peak_lag}"
        assert (
            abs(peak_lag - true_theta) <= 1
        ), f"recovered peak lag {peak_lag} not near true theta {true_theta}"
