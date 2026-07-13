"""Budget optimizer v2 — #139.

Efficient frontier, goal-seek, risk-aware objectives, breakeven/free mode, and
richer constraints (absolute-$, portfolio groups, keep-on floor), on the fitted
model's posterior response curves. Fast: synthetic concave ``ResponseCurves`` and
a fake model for the op layer — no MCMC.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.agents import model_ops as M
from mmm_framework.planning import (
    budget_frontier,
    goal_seek,
    objective_curves,
    optimize_budget,
)
from mmm_framework.planning.budget import ResponseCurves, _solve_allocation


def _curves(seed: int = 0, n_draws: int = 100) -> ResponseCurves:
    """Concave (saturating) per-channel curves with posterior spread."""
    rng = np.random.default_rng(seed)
    names = ["TV", "Search", "Social", "Display"]
    base = np.array([100.0, 60.0, 50.0, 40.0])
    mults = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
    amp = np.array([300.0, 220.0, 180.0, 120.0])
    half = np.array([80.0, 40.0, 60.0, 30.0])
    C, G = 4, len(mults)
    contrib = np.zeros((n_draws, C, G))
    for d in range(n_draws):
        a = amp * (1 + rng.normal(0, 0.15, C))
        for c in range(C):
            contrib[d, c] = a[c] * (1 - np.exp(-(base[c] * mults) / half[c]))
    return ResponseCurves(
        channel_names=names,
        multipliers=mults,
        base_spend=base,
        contributions=contrib,
        obs_max_spend=base * 2.0,
        n_obs=100,
    )


# ── objective curves ────────────────────────────────────────────────────────


def test_objective_curves_shapes_and_risk_order():
    c = _curves()
    mean = objective_curves(c, "mean")
    p10 = objective_curves(c, "p10")
    cvar5 = objective_curves(c, "cvar5")
    assert mean.shape == cvar5.shape == p10.shape == (4, len(c.multipliers))
    # downside curves sit at or below the mean; CVaR5 is the most conservative.
    assert np.all(p10 <= mean + 1e-9)
    assert np.all(cvar5 <= p10 + 1e-9)


def test_unknown_objective_raises():
    with pytest.raises(ValueError, match="Unknown objective"):
        objective_curves(_curves(), "sharpe")


# ── constrained solver ──────────────────────────────────────────────────────


def test_solve_respects_fixed_budget():
    c = _curves()
    mean = objective_curves(c, "mean")
    alloc, shadow, marg = _solve_allocation(
        mean,
        c.spend_grid,
        total_budget=250.0,
        lo_spend=np.zeros(4),
        hi_spend=c.base_spend * 2.5,
    )
    assert abs(alloc.sum() - 250.0) < 1e-3
    assert shadow > 0 and marg.shape == (4,)


def test_solve_group_constraint_binds():
    c = _curves()
    mean = objective_curves(c, "mean")
    B = 250.0
    # digital = Search+Social+Display (indices 1,2,3) ≥ 70% of budget
    alloc, _, _ = _solve_allocation(
        mean,
        c.spend_grid,
        total_budget=B,
        lo_spend=np.zeros(4),
        hi_spend=c.base_spend * 2.5,
        groups=[{"indices": [1, 2, 3], "min_spend": 0.7 * B}],
    )
    assert (alloc[1] + alloc[2] + alloc[3]) / B >= 0.7 - 1e-3


def test_solve_absolute_bounds_and_free_mode():
    c = _curves()
    mean = objective_curves(c, "mean")
    lo = np.array([50.0, 0.0, 0.0, 0.0])
    hi = np.array([120.0, 150.0, 150.0, 150.0])
    # free / breakeven mode: total is an output; TV pinned in [50, 120]
    alloc, _, marg = _solve_allocation(
        mean,
        c.spend_grid,
        total_budget=None,
        lo_spend=lo,
        hi_spend=hi,
        mode="free",
    )
    assert 50.0 - 1e-6 <= alloc[0] <= 120.0 + 1e-6
    # interior funded channels sit near the breakeven line (marginal ≈ 1)
    interior = (alloc > lo + 1e-6) & (alloc < hi - 1e-6)
    assert np.all(np.abs(marg[interior] - 1.0) < 0.5)


# ── optimize_budget advanced knobs ──────────────────────────────────────────


def test_default_path_unchanged_and_reports_marginals():
    """The mean/fixed default still runs (greedy) and now reports shadow price
    + marginal ROAS without changing the allocation contract."""
    c = _curves()
    res = optimize_budget(curves=c, total_budget=250.0)
    assert res.objective == "mean" and res.mode == "fixed"
    assert res.shadow_price is not None
    assert set(res.marginal_roas) == set(c.channel_names)
    assert abs(res.optimal_alloc.sum() - 250.0) < 1.0


def test_group_and_floor_constraints_honored():
    c = _curves()
    res = optimize_budget(
        curves=c,
        total_budget=250.0,
        groups=[
            {"name": "digital", "channels": ["Search", "Social"], "min_share": 0.5}
        ],
        min_channel_spend=10.0,
    )
    idx = {n: i for i, n in enumerate(c.channel_names)}
    dig = res.optimal_alloc[idx["Search"]] + res.optimal_alloc[idx["Social"]]
    assert dig / res.total_budget >= 0.5 - 1e-2
    assert res.optimal_alloc.min() >= 10.0 - 1e-6


def test_free_mode_total_is_an_output():
    c = _curves()
    res = optimize_budget(curves=c, mode="free", value_per_kpi=1.0)
    assert res.mode == "free"
    assert res.total_budget > 0
    # every channel's marginal return is near breakeven or it is at a bound
    assert res.shadow_price is not None


def test_unknown_group_channel_raises():
    c = _curves()
    with pytest.raises(ValueError, match="unknown channel"):
        optimize_budget(
            curves=c,
            total_budget=250.0,
            groups=[{"name": "x", "channels": ["Nope"], "min_share": 0.5}],
        )


# ── frontier ────────────────────────────────────────────────────────────────


def test_frontier_is_monotone_with_diminishing_marginal_roi():
    c = _curves()
    fr = budget_frontier(curves=c, n_points=8)
    rets = [p.expected_return for p in fr.points]
    assert all(rets[i + 1] >= rets[i] - 1e-6 for i in range(len(rets) - 1))
    mrois = [p.marginal_roi for p in fr.points if np.isfinite(p.marginal_roi)]
    assert all(mrois[i + 1] <= mrois[i] + 1e-6 for i in range(len(mrois) - 1))
    for p in fr.points:
        assert p.return_p5 <= p.expected_return <= p.return_p95 + 1e-6


def test_frontier_to_dict_roundtrip():
    fr = budget_frontier(curves=_curves(), n_points=5)
    d = fr.to_dict()
    assert d["objective"] == "mean"
    assert len(d["points"]) == 5
    assert set(d["points"][0]) >= {
        "total_budget",
        "expected_return",
        "return_p5",
        "return_p95",
        "marginal_roi",
    }


# ── goal-seek ───────────────────────────────────────────────────────────────


def test_goal_seek_hits_reachable_target():
    c = _curves()
    fr = budget_frontier(curves=c, n_points=6)
    target = fr.current_return * 1.1
    gs = goal_seek(curves=c, target_kpi=target)
    assert gs.feasible
    assert gs.required_budget > 0
    assert gs.expected_return >= target - 1.0
    assert 0.0 <= gs.prob_hit_target <= 1.0


def test_goal_seek_infeasible_target():
    gs = goal_seek(curves=_curves(), target_kpi=1e12)
    assert not gs.feasible
    assert gs.required_budget is None


# ── op layer (validation + payload) ─────────────────────────────────────────


class _FakeMMM:
    has_geo = False
    n_geos = 1
    geo_names: list[str] = []

    def __init__(self):
        rng = np.random.default_rng(0)
        self.channel_names = ["TV", "Search", "Social", "Display"]
        self.X_media_raw = np.abs(rng.normal([1.0, 0.6, 0.5, 0.4], 0.2, (80, 4)))
        self._amp = np.array([3.0, 2.2, 1.8, 1.2])
        self._half = np.array([0.8, 0.4, 0.6, 0.3])

    def sample_channel_contributions(
        self, X_media=None, max_draws=100, random_seed=None
    ):
        rng = np.random.default_rng(random_seed or 0)
        X = self.X_media_raw if X_media is None else X_media
        out = np.zeros((max_draws, X.shape[0], 4))
        for d in range(max_draws):
            a = self._amp * (1 + rng.normal(0, 0.15, 4))
            for c in range(4):
                out[d, :, c] = a[c] * (1 - np.exp(-X[:, c] / self._half[c]))
        return out


def test_plan_budget_op_validates_objective_and_mode():
    m = _FakeMMM()
    assert M.plan_budget(m, objective="sharpe")["error"]
    assert M.plan_budget(m, mode="nope")["error"]
    bad = M.plan_budget(
        m, groups=[{"name": "g", "channels": ["Nope"], "min_share": 0.5}]
    )
    assert bad["error"] and "unknown" in bad["error"].lower()


def test_plan_budget_op_returns_v2_payload():
    m = _FakeMMM()
    out = M.plan_budget(
        m, objective="cvar5", frontier={"n_points": 5}, target_kpi=200.0, max_draws=60
    )
    assert out["error"] is None
    plan = out["dashboard"]["budget_plan"]
    assert plan["objective"] == "cvar5"
    assert plan["shadow_price"] is not None
    assert set(plan["marginal_roas"]) == set(m.channel_names)
    assert len(plan["frontier"]["points"]) == 5
    assert "feasible" in plan["goal_seek"]


def test_optimize_budget_op_frontier_and_goal_seek():
    m = _FakeMMM()
    out = M.optimize_budget(m, frontier=True, target_kpi=150.0, max_draws=60)
    assert out["error"] is None
    summ = out["dashboard"]["budget_optimization"]
    assert "frontier" in summ and "goal_seek" in summ
    assert summ["objective"] == "mean"
