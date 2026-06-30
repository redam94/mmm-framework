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
    _adaptive_max_spend,
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


def _point_elasticity(x, params):
    f = _apply_saturation(x, params)
    df = _apply_saturation_derivative(x, params)
    return x * df / f if f > 1e-12 else 1.0


@pytest.mark.parametrize(
    "params",
    [
        {"type": "michaelis_menten", "kappa": np.array([0.5])},
        {"type": "exponential", "lam": np.array([1.5])},
        {"type": "hill", "kappa": np.array([0.6]), "slope": np.array([1.5])},
        {"type": "tanh", "kappa": np.array([0.5])},
    ],
)
def test_adaptive_max_spend_reaches_saturation(params):
    """The adaptive ceiling extends far enough that BOTH zone boundaries
    (optimal onset e=0.8 and saturation onset e=0.5) fall strictly inside the
    grid — so the deep-dive chart always shows all three zones, not just
    'breakthrough'. This is the regression guard for the collapsed-zone bug."""
    scale, sat_e = 1.0, 0.5
    # current spend tiny enough that neither floor nor cap binds
    ms = _adaptive_max_spend(
        params, scale, current_spend=0.05, saturation_elasticity=sat_e
    )
    xmax = ms / scale
    # elasticity is monotone-decreasing here; both boundaries must be in range
    assert _point_elasticity(xmax, params) < sat_e, ("did not reach saturation", ms)
    # ...and we didn't over-extend into the dead-flat tail (just past the onset)
    assert _point_elasticity(xmax, params) > sat_e * 0.4, ("over-extended", ms)


def test_adaptive_max_spend_floor_and_cap():
    """A channel that saturates very early floors at current×min_mult (still shows
    headroom past current); a near-linear channel caps at current×max_mult instead
    of running the axis to infinity."""
    mm = {"type": "michaelis_menten", "kappa": np.array([0.5])}
    # saturates early relative to a large current spend -> floored at 2.5×
    floored = _adaptive_max_spend(
        mm, 1.0, current_spend=100.0, saturation_elasticity=0.5
    )
    assert abs(floored - 100.0 * 2.5) < 1e-6
    # near-linear (huge half-saturation) never crosses the target -> capped at 60×
    linear = {"type": "michaelis_menten", "kappa": np.array([1e9])}
    capped = _adaptive_max_spend(
        linear, 1.0, current_spend=1.0, saturation_elasticity=0.5
    )
    assert abs(capped - 1.0 * 60.0) < 1e-6


@pytest.mark.parametrize(
    "params, has_breakthrough",
    [
        # CONCAVE forms have NO breakthrough level (marginal return highest at $0)
        ({"type": "exponential", "lam": np.array([1.0])}, False),
        ({"type": "michaelis_menten", "kappa": np.array([0.5])}, False),
        ({"type": "tanh", "kappa": np.array([0.5])}, False),
        ({"type": "hill", "kappa": np.array([0.5]), "slope": np.array([0.8])}, False),
        ({"type": "hill", "kappa": np.array([0.5]), "slope": np.array([1.0])}, False),
        # S-SHAPED forms have a convex take-off region -> breakthrough exists
        ({"type": "hill", "kappa": np.array([0.5]), "slope": np.array([2.0])}, True),
        ({"type": "hill", "kappa": np.array([0.5]), "slope": np.array([3.0])}, True),
        ({"type": "logistic", "lam": np.array([4.0])}, True),
    ],
)
def test_convex_region_end_is_shape_robust(params, has_breakthrough):
    """The breakthrough zone is the convex (increasing-marginal-return) region.
    It must be present ONLY for genuinely S-shaped curves and EMPTY for concave
    ones — the calculation has to be robust to the saturation shape."""
    from mmm_framework.reporting.helpers.saturation import _convex_region_end

    bt_end = _convex_region_end(params, scale=1.0, max_spend=10.0)
    if has_breakthrough:
        assert bt_end > 0.0, (params, bt_end)
    else:
        assert bt_end == 0.0, (params, bt_end)


def _zones_ns(mroi, roi, *, current_spend, current_roi, current_mroi, break_even=1.0):
    import types

    grid = np.linspace(0.0, 10.0, len(mroi))
    return types.SimpleNamespace(
        spend_grid=grid,
        mroi_mean=np.asarray(mroi, float),
        roi_mean=np.asarray(roi, float),
        current_spend=current_spend,
        current_roi=current_roi,
        current_mroi=current_mroi,
        break_even=break_even,
    )


def test_roi_axis_top_excludes_concave_toe_spike():
    """A concave curve's avg/marginal ROI peak at the toe (s→0); that degenerate
    spike must NOT set the ROI-axis scale, or it squashes the whole chart."""
    from mmm_framework.reporting.deck.charts import _roi_axis_top

    mroi = np.concatenate([[1000.0], np.linspace(5.0, 0.5, 49)])  # huge at the toe
    z = _zones_ns(mroi, mroi, current_spend=2.0, current_roi=4.0, current_mroi=4.0)
    top = _roi_axis_top(z)
    assert top < 50.0, top  # the toe spike (left of current) is excluded
    assert top >= 1.0 * 1.5  # break-even reference still in frame


def test_roi_axis_top_handles_infinite_toe():
    """Hill slope<1 marginal ROI literally diverges at s→0; the axis top must stay
    finite and reasonable (scaled to the decision region)."""
    from mmm_framework.reporting.deck.charts import _roi_axis_top

    mroi = np.linspace(5.0, 0.5, 50)
    mroi[0] = np.inf
    z = _zones_ns(mroi, mroi, current_spend=2.0, current_roi=4.0, current_mroi=4.0)
    top = _roi_axis_top(z)
    assert np.isfinite(top) and top < 50.0, top


def test_roi_axis_top_keeps_scurve_interior_peak():
    """An S-shaped curve's marginal-ROI peak is in the INTERIOR (its breakthrough
    region) — it must be kept in frame, not clipped."""
    from mmm_framework.reporting.deck.charts import _roi_axis_top

    grid = np.linspace(0.0, 10.0, 50)
    mroi = np.exp(-((grid - 3.0) ** 2) / 2.0) * 100.0  # peak ~100 at the interior
    roi = np.minimum(mroi, 30.0)
    z = _zones_ns(mroi, roi, current_spend=1.0, current_roi=10.0, current_mroi=10.0)
    top = _roi_axis_top(z)
    # the interior marginal-ROI peak is kept in frame (not clipped to ~current)
    assert top >= float(np.nanmax(mroi)) * 0.999, top


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
    # use a wide spend range so the concave curve actually saturates and the
    # optimal/saturation zones appear (within only ~2× current spend these channels
    # are still linear). The clean DGP uses EXPONENTIAL (concave) saturation, so
    # there is NO breakthrough zone — the calculation is shape-aware.
    def test_structure_and_zone_ordering(self, fitted_model):
        zones = compute_response_zones(fitted_model, spend_multiplier=12, hdi_prob=0.8)
        assert len(zones) >= 1
        assert set(zones) <= set(fitted_model.channel_names)
        for ch, z in zones.items():
            n = len(z.spend_grid)
            assert z.roi_mean.shape == (n,) and z.mroi_mean.shape == (n,)
            # concave (exponential) saturation ⇒ marginal ROI is non-increasing
            assert z.mroi_mean[0] >= z.mroi_mean[-1] - 1e-9
            # zones partition the spend axis in order, within the grid
            bt, op, sat = z.breakthrough_range, z.optimal_range, z.saturation_range
            assert bt[0] == 0.0
            assert bt[1] <= op[1] + 1e-6
            assert op[0] == bt[1] and sat[0] == op[1]
            assert abs(sat[1] - z.spend_grid[-1]) < 1e-6
            # concave curve ⇒ breakthrough zone is EMPTY (no take-off threshold)
            assert bt == (0.0, 0.0), (ch, bt)
            assert op[1] > op[0], (ch, "optimal zone should be non-empty", op)
            # average ROI is exactly response/spend at an interior grid point
            i = n // 2
            assert abs(z.roi_mean[i] - z.response_mean[i] / z.spend_grid[i]) < 1e-6 * (
                abs(z.roi_mean[i]) + 1.0
            )
            # recommendation is position-based and self-consistent (shape-robust):
            # 'increase' ⇒ under the operating point, 'reduce' ⇒ in saturation.
            assert z.current_zone in ("breakthrough", "optimal", "saturation")
            assert z.recommendation in ("increase", "hold", "reduce")
            if z.recommendation == "increase":
                assert z.optimal_spend is not None
                assert z.current_spend < z.optimal_spend
            if z.recommendation == "reduce":
                assert z.current_spend >= z.saturation_range[0] - 1e-6

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
        # the optimal zone is more efficient (higher elasticity) than saturation,
        # where the next dollar earns far less than the average. (The clean DGP is
        # concave ⇒ no breakthrough zone, so we compare optimal vs saturation.)
        zones = compute_response_zones(fitted_model, spend_multiplier=12, hdi_prob=0.8)
        checked = 0
        for z in zones.values():
            op, sat = z.optimal_range, z.saturation_range
            if op[1] <= op[0] or sat[1] <= sat[0]:
                continue
            checked += 1
            op_mid = 0.5 * (op[0] + op[1])
            sat_mid = 0.5 * (sat[0] + sat[1])
            assert _elasticity_at(z, op_mid) > _elasticity_at(z, sat_mid)
            # saturation: marginal ROI is well below average ROI
            assert _elasticity_at(z, sat_mid) < 0.55
        assert checked >= 1
