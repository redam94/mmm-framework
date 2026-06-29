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


@pytest.mark.slow
class TestResponseZones:
    def test_structure_and_zone_ordering(self, fitted_model):
        zones = compute_response_zones(fitted_model, break_even=1.0, band=0.15)
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
            # current-zone classification follows current marginal ROI
            t_hi, t_lo = 1.15, 0.85
            if z.current_mroi >= t_hi:
                assert (
                    z.current_zone == "breakthrough" and z.recommendation == "increase"
                )
            elif z.current_mroi >= t_lo:
                assert z.current_zone == "optimal" and z.recommendation == "hold"
            else:
                assert z.current_zone == "saturation" and z.recommendation == "reduce"

    def test_optimal_spend_sits_at_break_even(self, fitted_model):
        # The synthetic KPI is an index (not dollars), so a *revenue* break-even of
        # 1.0 may sit outside the channel's mROI range. Pick a target inside that
        # range so the curve is guaranteed to cross it, then assert the optimal
        # spend is exactly where mROI == that target (the property under test).
        base = compute_response_zones(fitted_model, band=0.15)
        ch = next(iter(base))
        z0 = base[ch]
        target = float(
            0.5 * (z0.mroi_mean[0] + z0.mroi_mean[-1])
        )  # midpoint of mROI range
        z = compute_response_zones(fitted_model, [ch], break_even=target, band=0.15)[ch]
        assert z.optimal_spend is not None
        mroi_at = float(np.interp(z.optimal_spend, z.spend_grid, z.mroi_mean))
        assert abs(mroi_at - target) < 0.05 * abs(target) + 1e-6, (mroi_at, target)
        # headroom is the signed distance to the optimum
        assert abs(z.headroom_to_optimal - (z.optimal_spend - z.current_spend)) < 1e-6

    def test_higher_break_even_shrinks_optimal_spend(self, fitted_model):
        base = compute_response_zones(fitted_model, band=0.15)
        ch = next(iter(base))
        z0 = base[ch]
        hi_m, lo_m = float(z0.mroi_mean[0]), float(
            z0.mroi_mean[-1]
        )  # max @ low spend, min @ high
        lo_t = 0.4 * hi_m + 0.6 * lo_m  # lower target  -> higher optimal spend
        hi_t = 0.6 * hi_m + 0.4 * lo_m  # higher target -> lower optimal spend
        lo = compute_response_zones(fitted_model, [ch], break_even=lo_t)[ch]
        hi = compute_response_zones(fitted_model, [ch], break_even=hi_t)[ch]
        assert lo.optimal_spend is not None and hi.optimal_spend is not None
        # a higher break-even target demands higher mROI ⇒ lower optimal spend
        assert hi.optimal_spend <= lo.optimal_spend + 1e-6
