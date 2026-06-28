"""Tests for the Planner backend: geo/DMA allocation (B4), the forward flighting
calendar (B6), and the plan_budget / plan_scenario model-ops the Planner FE drives.

The allocator math is exercised against a fake fitted model whose per-(geo,
channel) response is a known concave (sqrt) curve, so optima are predictable
without a real MCMC fit. One slow end-to-end test runs the op against a real fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.agents import model_ops as M
from mmm_framework.planning import (
    build_flighting_schedule,
    combine_geo_curves,
    compute_response_curves_per_geo,
    optimize_budget_by_geo,
)
from mmm_framework.planning.flighting import _pattern_weights


class FakeGeoMMM:
    """Additive geo panel: contribution(obs, ch) = coef[geo, ch] * sqrt(spend)."""

    def __init__(self):
        self.channel_names = ["TV", "Search"]
        self.geo_names = ["North", "South"]
        self.has_geo = True
        self.n_geos = 2
        # 2 geos × 3 periods = 6 obs
        self.geo_idx = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        self.X_media_raw = np.full((6, 2), 100.0)
        # North favors TV, South favors Search (mirror image) → national is
        # symmetric, but per-geo reveals the heterogeneity.
        self._coef = np.array([[3.0, 1.0], [1.0, 3.0]])

    def sample_channel_contributions(
        self, X_media=None, max_draws=None, random_seed=None
    ):
        X = self.X_media_raw if X_media is None else X_media
        n_obs, C = X.shape
        out = np.zeros((4, n_obs, C))
        for i in range(n_obs):
            g = int(self.geo_idx[i])
            for c in range(C):
                out[:, i, c] = self._coef[g, c] * np.sqrt(max(X[i, c], 0.0))
        return out

    def what_if_scenario(
        self, spend_changes, time_period=None, max_draws=200, random_seed=None
    ):
        return {
            "baseline_outcome": 1000.0,
            "scenario_outcome": 1100.0,
            "outcome_change": 100.0,
            "outcome_change_pct": 10.0,
            "spend_changes": {
                ch: {
                    "original": 100.0,
                    "scenario": 100.0 * (1 + v),
                    "change": 100.0 * v,
                    "change_pct": 100.0 * v,
                }
                for ch, v in spend_changes.items()
            },
            "outcome_change_hdi": [50.0, 150.0],
            "prob_positive": 0.9,
            "n_draws": 50,
            "hdi_prob": 0.94,
        }


class TestFlighting:
    def test_even_conserves_and_is_flat(self):
        fl = build_flighting_schedule({"TV": 130.0}, 13, pattern="even")
        assert len(fl["schedule"]) == 13
        assert sum(r["total"] for r in fl["schedule"]) == pytest.approx(130.0)
        vals = [r["TV"] for r in fl["schedule"]]
        assert max(vals) - min(vals) < 1e-9

    def test_front_loaded_decreases(self):
        w = _pattern_weights("front_loaded", 5)
        assert w[0] > w[-1]
        assert w.sum() == pytest.approx(1.0)

    def test_back_loaded_increases(self):
        w = _pattern_weights("back_loaded", 5)
        assert w[0] < w[-1]

    def test_pulsed_has_off_periods(self):
        fl = build_flighting_schedule(
            {"TV": 100.0}, 6, pattern="pulsed", pulse_on=1, pulse_off=1
        )
        spend = [r["TV"] for r in fl["schedule"]]
        assert spend[0] > 0 and spend[1] == 0  # on, off, on, off...
        assert sum(spend) == pytest.approx(100.0)

    def test_conserves_per_channel_budget(self):
        fl = build_flighting_schedule(
            {"TV": 200.0, "Search": 50.0}, 8, pattern="front_loaded"
        )
        assert sum(fl["by_channel"]["TV"]) == pytest.approx(200.0)
        assert sum(fl["by_channel"]["Search"]) == pytest.approx(50.0)
        assert fl["total_budget"] == pytest.approx(250.0)

    def test_per_channel_pattern_override(self):
        fl = build_flighting_schedule(
            {"TV": 100.0, "Search": 100.0},
            4,
            pattern="even",
            per_channel_pattern={"Search": "pulsed"},
        )
        assert min(fl["by_channel"]["TV"]) == pytest.approx(max(fl["by_channel"]["TV"]))
        assert min(fl["by_channel"]["Search"]) == 0.0  # pulsed has off weeks

    def test_custom_pattern_validates_length(self):
        with pytest.raises(ValueError):
            _pattern_weights("custom", 4, custom=[1, 2])

    def test_unknown_pattern_raises(self):
        with pytest.raises(ValueError):
            _pattern_weights("nope", 4)


class TestGeoAllocation:
    def test_per_geo_curves_recover_heterogeneity(self):
        mmm = FakeGeoMMM()
        curves = compute_response_curves_per_geo(mmm, max_draws=4)
        assert set(curves) == {"North", "South"}
        # each geo's base spend is its 3 periods × 100
        assert curves["North"].base_spend.tolist() == [300.0, 300.0]

    def test_combine_then_optimize_moves_money_within_geo(self):
        mmm = FakeGeoMMM()
        res = optimize_budget_by_geo(mmm, max_draws=4, n_steps=600)
        t = res.table
        assert {"geo", "channel"} <= set(t.columns)
        north = t[t["geo"] == "North"].set_index("channel")
        south = t[t["geo"] == "South"].set_index("channel")
        # North favors TV, South favors Search — money follows the per-geo curve
        assert north.loc["TV", "optimal_spend"] > north.loc["Search", "optimal_spend"]
        assert south.loc["Search", "optimal_spend"] > south.loc["TV", "optimal_spend"]
        # national budget is conserved
        assert t["optimal_spend"].sum() == pytest.approx(res.total_budget, rel=0.02)

    def test_combine_geo_curves_flattens_arms(self):
        mmm = FakeGeoMMM()
        curves = compute_response_curves_per_geo(mmm, max_draws=4)
        combined = combine_geo_curves(curves)
        assert len(combined.channel_names) == 4  # 2 geos × 2 channels
        assert combined.base_spend.shape == (4,)


class TestPlanOps:
    def test_plan_budget_national(self):
        res = M.OPS["plan_budget"](FakeGeoMMM(), None, by_geo=False, max_draws=4)
        assert not res.get("error"), res.get("error")
        plan = res["dashboard"]["budget_plan"]
        assert plan["by_geo"] is False
        assert {a["channel"] for a in plan["allocation"]} == {"TV", "Search"}
        assert "geo_allocation" not in plan
        assert res["tables"]

    def test_plan_budget_geo_and_flighting(self):
        res = M.OPS["plan_budget"](
            FakeGeoMMM(),
            None,
            by_geo=True,
            flighting={"pattern": "front_loaded", "n_periods": 8},
            max_draws=4,
        )
        assert not res.get("error"), res.get("error")
        plan = res["dashboard"]["budget_plan"]
        assert plan["by_geo"] is True
        assert plan["geos"] == ["North", "South"]
        assert len(plan["geo_allocation"]) == 4
        fl = plan["flighting"]
        assert fl["pattern"] == "front_loaded" and fl["n_periods"] == 8
        # flighting spreads the rolled-up channel budgets
        total_sched = sum(r["total"] for r in fl["schedule"])
        total_alloc = sum(a["optimal_spend"] for a in plan["allocation"])
        assert total_sched == pytest.approx(total_alloc, rel=0.02)

    def test_plan_budget_rejects_bad_bounds(self):
        res = M.OPS["plan_budget"](FakeGeoMMM(), None, bounds={"Ghost": [0, 1]})
        assert res.get("error") and "Unknown channel" in res["error"]

    def test_plan_scenario_structured_output(self):
        res = M.OPS["plan_scenario"](
            FakeGeoMMM(), None, spend_changes={"TV": 0.2}, max_draws=4
        )
        assert not res.get("error"), res.get("error")
        sc = res["dashboard"]["budget_scenario"]
        assert sc["outcome_change"] == pytest.approx(100.0)
        assert sc["outcome_change_hdi"] == [50.0, 150.0]
        assert sc["prob_positive"] == pytest.approx(0.9)
        assert sc["channel_details"]["TV"]["change_pct"] == pytest.approx(20.0)

    def test_plan_scenario_rejects_unknown_channel(self):
        res = M.OPS["plan_scenario"](FakeGeoMMM(), None, spend_changes={"Ghost": 0.2})
        assert res.get("error") and "Unknown channel" in res["error"]


@pytest.mark.slow
def test_plan_budget_end_to_end_with_fitted_model(tmp_path):
    """Full path: fit a tiny national model, then run plan_budget with flighting."""
    import os
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples"))
    )
    from ex_model_workflow import generate_synthetic_mff

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

    res = M.OPS["plan_budget"](
        mmm, None, flighting={"pattern": "even", "n_periods": 13}, max_draws=50
    )
    assert not res.get("error"), res.get("error")
    plan = res["dashboard"]["budget_plan"]
    assert {a["channel"] for a in plan["allocation"]} == {"TV", "Digital"}
    assert len(plan["flighting"]["schedule"]) == 13
    total_sched = sum(r["total"] for r in plan["flighting"]["schedule"])
    total_alloc = sum(a["optimal_spend"] for a in plan["allocation"])
    assert total_sched == pytest.approx(total_alloc, rel=0.02)
