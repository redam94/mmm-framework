"""Tests for the decision layer (mmm_framework.planning): budget optimization
and experiment-design recommendation.

The allocator and scorer are pure numpy/pandas over sampled response curves, so
they are tested here against synthetic curves with known optima — no model fit
needed. The model-integration path (sample_channel_contributions → ops) is
covered by the slow end-to-end test at the bottom.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning import (
    ResponseCurves,
    optimize_budget,
    recommend_experiments,
)
from mmm_framework.planning.budget import _eval_allocation, _greedy_allocate


def _sqrt_curves(
    base_spend: np.ndarray,
    coefs: np.ndarray,
    multipliers: np.ndarray,
    n_draws: int = 1,
    noise: float = 0.0,
    seed: int = 0,
) -> ResponseCurves:
    """Concave (sqrt) response curves: contribution_c(s) = coef_c * sqrt(s).
    With sqrt curves the optimum allocates spend proportional to coef^2."""
    rng = np.random.default_rng(seed)
    spend = base_spend[:, None] * multipliers[None, :]
    mean = coefs[:, None] * np.sqrt(spend)  # (C, G)
    draws = np.stack(
        [mean * (1.0 + noise * rng.standard_normal(mean.shape)) for _ in range(n_draws)]
    )
    return ResponseCurves(
        channel_names=[f"ch{i}" for i in range(len(base_spend))],
        multipliers=multipliers,
        base_spend=base_spend,
        contributions=draws,
    )


MULTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])


class TestGreedyAllocator:
    def test_conserves_budget_and_respects_bounds(self):
        curves = _sqrt_curves(np.array([100.0, 100.0]), np.array([1.0, 2.0]), MULTS)
        spend_grid = curves.spend_grid
        alloc = _greedy_allocate(
            curves.mean_curves(),
            spend_grid,
            total_budget=200.0,
            lo_spend=np.array([20.0, 20.0]),
            hi_spend=np.array([150.0, 150.0]),
            n_steps=500,
        )
        assert alloc.sum() == pytest.approx(200.0, rel=1e-6)
        assert np.all(alloc >= 20.0 - 1e-9) and np.all(alloc <= 150.0 + 1e-6)

    def test_finds_known_sqrt_optimum(self):
        # coef^2 ratio 1:4 -> optimal split 20%/80% of budget (interior optimum).
        # A fine grid isolates the allocator's math from interpolation error
        # (on the default 10-point grid the surrogate optimum shifts a few pts).
        fine = np.linspace(0.0, 2.0, 81)
        curves = _sqrt_curves(np.array([100.0, 100.0]), np.array([1.0, 2.0]), fine)
        alloc = _greedy_allocate(
            curves.mean_curves(),
            curves.spend_grid,
            total_budget=200.0,
            lo_spend=np.zeros(2),
            hi_spend=np.array([200.0, 200.0]),
            n_steps=2000,
        )
        assert alloc[0] / 200.0 == pytest.approx(0.2, abs=0.02)
        assert alloc[1] / 200.0 == pytest.approx(0.8, abs=0.02)

    def test_caps_bind(self):
        # ch1 dominates but is capped at 1.2x -> remainder flows to ch0
        curves = _sqrt_curves(np.array([100.0, 100.0]), np.array([0.5, 5.0]), MULTS)
        alloc = _greedy_allocate(
            curves.mean_curves(),
            curves.spend_grid,
            total_budget=200.0,
            lo_spend=np.zeros(2),
            hi_spend=np.array([200.0, 120.0]),
            n_steps=1000,
        )
        assert alloc[1] == pytest.approx(120.0, rel=0.01)
        assert alloc[0] == pytest.approx(80.0, rel=0.01)

    def test_eval_allocation_interpolates(self):
        curves = _sqrt_curves(np.array([100.0]), np.array([2.0]), MULTS)
        val = _eval_allocation(
            np.array([100.0]), curves.mean_curves(), curves.spend_grid
        )
        assert val == pytest.approx(2.0 * np.sqrt(100.0), rel=1e-6)


class TestOptimizeBudget:
    def test_reallocation_beats_current_when_rois_differ(self):
        curves = _sqrt_curves(
            np.array([100.0, 100.0]),
            np.array([1.0, 3.0]),
            MULTS,
            n_draws=50,
            noise=0.05,
        )
        res = optimize_budget(curves=curves, n_steps=500)
        assert res.total_budget == pytest.approx(200.0)
        t = res.table.set_index("channel")
        # money moves toward the higher-return channel
        assert t.loc["ch1", "optimal_spend"] > t.loc["ch0", "optimal_spend"]
        assert res.expected_uplift > 0
        assert res.prob_positive_uplift > 0.9
        # shares sum to ~100%
        assert t["optimal_share_pct"].sum() == pytest.approx(100.0, abs=1.0)

    def test_budget_change_pct_scales_total(self):
        curves = _sqrt_curves(np.array([100.0, 100.0]), np.array([1.0, 1.0]), MULTS)
        res = optimize_budget(curves=curves, budget_change_pct=-10, n_steps=200)
        assert res.total_budget == pytest.approx(180.0)

    def test_symmetric_channels_split_evenly_and_stably(self):
        curves = _sqrt_curves(
            np.array([100.0, 100.0]), np.array([2.0, 2.0]), MULTS, n_draws=30
        )
        res = optimize_budget(curves=curves, n_steps=1000)
        t = res.table.set_index("channel")
        assert t.loc["ch0", "optimal_share_pct"] == pytest.approx(50.0, abs=3.0)
        # identical noiseless draws -> no allocation instability
        assert t["allocation_instability"].max() < 1.0


class TestRecommendExperiments:
    def _fake_mmm(self):
        class _M:
            has_geo = False

            class mff_config:
                media_channels: list = []

        return _M()

    def test_prioritizes_uncertain_high_spend_channel(self):
        # ch0: large spend, very noisy ROAS; ch1: small spend, precise ROAS
        rng = np.random.default_rng(1)
        base = np.array([300.0, 50.0])
        mean = np.array([1.5, 1.5])[:, None] * np.sqrt(base[:, None] * MULTS[None, :])
        draws = []
        for _ in range(80):
            wiggle = np.array([1 + 0.6 * rng.standard_normal(), 1.0])
            draws.append(mean * wiggle[:, None])
        curves = ResponseCurves(
            channel_names=["big_noisy", "small_precise"],
            multipliers=MULTS,
            base_spend=base,
            contributions=np.stack(draws),
        )
        table, designs = recommend_experiments(self._fake_mmm(), curves=curves, top_k=1)
        assert table.iloc[0]["channel"] == "big_noisy"
        assert designs[0]["channel"] == "big_noisy"
        # national model -> spend-pulse design; snippet targets the channel
        assert "spend pulse" in designs[0]["design_type"]
        assert "'big_noisy'" in designs[0]["calibration_snippet"]
        assert "ExperimentMeasurement" in designs[0]["calibration_snippet"]
        assert designs[0]["min_duration_periods"] >= 12  # default l_max 8 + 4

    def test_table_covers_all_channels_sorted(self):
        curves = _sqrt_curves(
            np.array([100.0, 100.0, 100.0]),
            np.array([1.0, 2.0, 3.0]),
            MULTS,
            n_draws=20,
            noise=0.1,
        )
        table, designs = recommend_experiments(self._fake_mmm(), curves=curves, top_k=2)
        assert len(table) == 3
        assert list(table["priority"]) == sorted(table["priority"], reverse=True)
        assert len(designs) == 2


@pytest.mark.slow
def test_planning_end_to_end_with_fitted_model(tmp_path):
    """Full path: fit a tiny model, then run both model ops against it."""
    import os
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
    )
    from ex_model_workflow import generate_synthetic_mff

    from mmm_framework.agents import model_ops as M
    from mmm_framework.agents.fitting import build_model

    df = generate_synthetic_mff(n_weeks=60)
    path = str(tmp_path / "mff.csv")
    df.to_csv(path, index=False)
    spec = {
        "kpi": "Sales",
        "kpi_level": "national",
        "time_granularity": "weekly",
        "media_channels": [{"name": "TV"}, {"name": "Digital"}],
        "control_variables": [],
        "inference": {"chains": 2, "draws": 100, "tune": 100},
    }
    mmm = build_model(spec, path)
    mmm.fit(random_seed=1)

    res = M.OPS["optimize_budget"](mmm, None, max_draws=50)
    assert not res.get("error"), res.get("error")
    alloc = res["dashboard"]["budget_optimization"]["allocation"]
    assert {a["channel"] for a in alloc} == {"TV", "Digital"}
    total = sum(a["optimal_spend"] for a in alloc)
    assert total == pytest.approx(
        res["dashboard"]["budget_optimization"]["total_budget"], rel=0.01
    )
    assert res["tables"], "expected a dashboard table"

    res2 = M.OPS["experiment_design"](mmm, None, top_k=2, max_draws=50)
    assert not res2.get("error"), res2.get("error")
    designs = res2["dashboard"]["experiment_design"]["designs"]
    assert len(designs) == 2
    assert all("calibration_snippet" in d for d in designs)
