"""ROI / marginal-ROI spend-response zones.

``compute_response_zones`` defines a channel's **breakthrough / optimal /
saturation** spend ranges on **marginal-ROI break-even bands** — NOT on percent
of maximum response. These tests pin (a) the analytic saturation derivative used
for marginal ROI, (b) the zone-boundary crossing logic, and (c) the end-to-end
behaviour on a fitted model (curve shapes, zone ordering, the ROI=response/spend
identity, current-zone classification, and break-even monotonicity).
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.reporting.helpers.saturation import (
    _apply_saturation,
    _apply_saturation_derivative,
    _largest_spend_at_or_above,
    compute_response_zones,
)

# ---------------------------------------------------------------------------
# fast: analytic marginal-ROI math (no model)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"type": "exponential", "lam": np.array([0.5, 1.5, 3.0])},
        {
            "type": "hill",
            "kappa": np.array([0.3, 0.5, 0.8]),
            "slope": np.array([1.2, 2.0, 3.0]),
        },
        {"type": "michaelis_menten", "kappa": np.array([0.2, 0.5, 1.0])},
        {"type": "tanh", "kappa": np.array([0.3, 0.6, 1.0])},
        {"type": "logistic", "lam": np.array([2.0, 4.0, 8.0])},
    ],
)
def test_saturation_derivative_matches_numerical(params):
    """The analytic derivative used for mROI matches central-difference numerical
    differentiation of the saturation function, for every saturation form."""
    eps = 1e-6
    for x in (0.05, 0.2, 0.5, 1.0, 1.8):
        num = (
            _apply_saturation(x + eps, params) - _apply_saturation(x - eps, params)
        ) / (2 * eps)
        ana = _apply_saturation_derivative(x, params)
        np.testing.assert_allclose(ana, num, rtol=2e-4, atol=1e-6)


def test_largest_spend_crossing():
    grid = np.linspace(0, 10, 101)
    y = 3.0 - 0.3 * grid  # decreasing; crosses 1.0 at s = 20/3 ≈ 6.667
    s = _largest_spend_at_or_above(grid, y, 1.0)
    assert abs(s - 20.0 / 3.0) < 0.1
    # never below threshold -> right edge of the grid
    assert _largest_spend_at_or_above(grid, y, -5.0) == grid[-1]
    # always below threshold -> left edge (zone empty)
    assert _largest_spend_at_or_above(grid, y, 100.0) == grid[0]


# ---------------------------------------------------------------------------
# slow: end-to-end on a fitted model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model():
    from mmm_framework.config import ModelConfig
    from mmm_framework.model import BayesianMMM, TrendConfig
    from mmm_framework.model.trend_config import TrendType
    from mmm_framework.synth import dgp

    panel = dgp.build("clean", seed=0, n_weeks=104).panel()
    mmm = BayesianMMM(
        panel,
        ModelConfig(use_parametric_adstock=True),
        TrendConfig(type=TrendType.LINEAR),
    )
    mmm.fit(
        draws=300,
        tune=600,
        chains=2,
        target_accept=0.9,
        random_seed=3,
        progressbar=False,
    )
    return mmm


def _elasticity_at(z, s):
    roi = float(np.interp(s, z.spend_grid, z.roi_mean))
    mroi = float(np.interp(s, z.spend_grid, z.mroi_mean))
    return mroi / roi if roi > 1e-9 else 0.0


@pytest.mark.slow
class TestResponseZones:
    # use a wide spend range so the concave curve actually saturates and all three
    # zones appear (within only ~2× current spend these channels are still linear).
    def test_structure_and_zone_ordering(self, fitted_model):
        zones = compute_response_zones(fitted_model, spend_multiplier=12, hdi_prob=0.8)
        assert len(zones) >= 1
        assert set(zones) <= set(fitted_model.channel_names)
        for ch, z in zones.items():
            n = len(z.spend_grid)
            assert z.roi_mean.shape == (n,) and z.mroi_mean.shape == (n,)
            # concave (logistic/exp) saturation ⇒ marginal ROI is non-increasing
            assert z.mroi_mean[0] >= z.mroi_mean[-1] - 1e-9
            # zones partition the spend axis in order, within the grid
            bt, op, sat = z.breakthrough_range, z.optimal_range, z.saturation_range
            assert bt[0] == 0.0
            assert bt[1] <= op[1] + 1e-6
            assert op[0] == bt[1] and sat[0] == op[1]
            assert abs(sat[1] - z.spend_grid[-1]) < 1e-6
            # average ROI is exactly response/spend at an interior grid point
            i = n // 2
            assert abs(z.roi_mean[i] - z.response_mean[i] / z.spend_grid[i]) < 1e-6 * (
                abs(z.roi_mean[i]) + 1.0
            )
            # current-zone follows the current elasticity e = mROI/ROI (thresholds 0.8 / 0.5)
            e = z.current_mroi / z.current_roi if z.current_roi > 1e-9 else 0.0
            if e >= 0.8:
                assert (
                    z.current_zone == "breakthrough" and z.recommendation == "increase"
                )
            elif e >= 0.5:
                assert z.current_zone == "optimal" and z.recommendation == "hold"
            else:
                assert z.current_zone == "saturation" and z.recommendation == "reduce"

    def test_optimal_point_sits_in_the_efficient_elasticity_band(self, fitted_model):
        # the optimal operating point's elasticity (mROI/ROI) is in the optimal band
        zones = compute_response_zones(fitted_model, spend_multiplier=12, hdi_prob=0.8)
        checked = 0
        for z in zones.values():
            if z.optimal_spend is None:
                continue
            checked += 1
            e = _elasticity_at(z, z.optimal_spend)
            assert 0.45 <= e <= 0.85, (
                z.channel,
                e,
            )  # within the optimal band (small tol)
            assert (
                abs(z.headroom_to_optimal - (z.optimal_spend - z.current_spend)) < 1e-6
            )
        assert checked >= 1, "at least one channel should saturate in range"

    def test_zones_ordered_by_elasticity(self, fitted_model):
        # saturation (high spend) is where the next dollar earns far less than the
        # average; breakthrough (low spend) is the near-linear regime.
        zones = compute_response_zones(fitted_model, spend_multiplier=12, hdi_prob=0.8)
        checked = 0
        for z in zones.values():
            bt, sat = z.breakthrough_range, z.saturation_range
            if bt[1] <= bt[0] or sat[1] <= sat[0]:
                continue
            checked += 1
            bt_mid = 0.5 * (bt[0] + bt[1])
            sat_mid = 0.5 * (sat[0] + sat[1])
            assert _elasticity_at(z, bt_mid) > _elasticity_at(z, sat_mid)
            # saturation: marginal ROI is well below average ROI
            assert _elasticity_at(z, sat_mid) < 0.55
        assert checked >= 1
