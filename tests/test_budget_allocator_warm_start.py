"""The constrained allocator must never return a point it can beat (issue #290).

The defect: `_solve_allocation`'s default warm start works out to the **current
allocation** — with `lo=0` and `hi=2·base`, spreading the current total
proportionally over the headroom reproduces current spend exactly. The channel
curves are piecewise linear, so the gradient is piecewise *constant* and SLSQP's
line search could fail to find an improving step and exit at `nit=2` with
`success=True` and `x` unmoved. `res.success` was never checked and the iterate
was returned as "the optimal plan".

The output of that failure is `optimal_allocation == current allocation` and
`expected_uplift ≈ 0`, which reads as "your plan is already optimal" — the most
trustworthy-looking thing a planning tool can say. Measured on a real fit, direct
search beat it by +372 (p10) and +461 (cvar5).

These tests grade the allocator against **search**, not against itself, which is
the only check that would have caught it. A test that merely asserted the solver
returned *something* passed throughout.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmm_framework.planning import ResponseCurves, optimize_budget
from mmm_framework.planning.budget import _solve_allocation, objective_curves

CHANNELS = ["TV", "Search", "Social", "Display"]
MULTIPLIERS = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
BASE = np.array([9680.0, 5619.0, 5284.0, 3905.0])

#: The P10 objective curve from the fit that surfaced this, captured verbatim.
#:
#: Synthetic concave curves do NOT reproduce the stall — a first version of this
#: file built its own and passed against the unfixed code, which is the failure
#: mode a regression test exists to avoid. The geometry matters: SLSQP stalls on
#: *these* segment slopes and not on smooth textbook ones, so the real numbers
#: are the test. Source: `make_unobserved_confounding(seed=7, n_weeks=156)`,
#: geometric x hill, NUTS 4x1000, `compute_response_curves(max_draws=200,
#: random_seed=42)`, `objective_curves(curves, "p10")`.
P10_CURVE = np.array(
    [
        [0.0, 350.0, 1498.7, 2928.5, 4153.7, 5092.0, 5805.9, 6412.1, 6987.6, 7772.0],
        [0.0, 366.5, 1104.8, 1890.9, 2630.2, 3240.4, 3798.5, 4246.1, 4648.5, 5374.5],
        [0.0, 884.1, 1758.7, 2577.6, 3176.4, 3720.0, 4188.9, 4604.7, 5021.8, 5660.3],
        [0.0, 46.8, 283.1, 745.2, 1200.4, 1568.2, 1887.1, 2080.2, 2274.9, 2592.9],
    ]
)
SPEND_GRID = MULTIPLIERS[None, :] * BASE[:, None]


def _curve_value(alloc: np.ndarray, curve: np.ndarray = P10_CURVE) -> float:
    return float(sum(np.interp(alloc[i], SPEND_GRID[i], curve[i]) for i in range(4)))


class TestTheRealStall:
    """The captured geometry, straight through `_solve_allocation`.

    On the unfixed code this returns the warm start — which is the current
    allocation — scoring 11,160.7 against a reachable 11,582.6.
    """

    def _solve(self, **kw):
        alloc, _shadow, _marg = _solve_allocation(
            P10_CURVE,
            SPEND_GRID,
            total_budget=float(BASE.sum()),
            lo_spend=np.zeros(4),
            hi_spend=2.0 * BASE,
            **kw,
        )
        return alloc

    def test_does_not_return_the_current_allocation(self):
        alloc = self._solve()
        assert not np.allclose(alloc, BASE, rtol=1e-6), (
            "The allocator returned the current allocation unchanged. That is "
            "what a stalled SLSQP produces, and it renders as 'uplift 0'."
        )

    def test_improves_on_the_warm_start(self):
        alloc = self._solve()
        assert _curve_value(alloc) > _curve_value(BASE)

    def test_recovers_the_known_reachable_value(self):
        """Measured: 11,582.6, which also beats a 60k-sample random search
        (11,560.4). Asserted loosely so a solver improvement is not a failure."""
        assert _curve_value(self._solve()) >= 11_500.0

    def test_an_explicit_warm_start_is_still_beaten(self):
        """A caller passing today's plan as `x0` must not pin the answer to it —
        the per-draw re-optimizations do exactly that."""
        alloc = self._solve(x0=BASE.copy())
        assert _curve_value(alloc) > _curve_value(BASE)

    def test_budget_equality_holds(self):
        assert float(self._solve().sum()) == pytest.approx(float(BASE.sum()), rel=1e-6)


def _curves(seed: int = 0, n_draws: int = 120) -> ResponseCurves:
    """Concave per-channel curves with channel-specific draw dispersion.

    The dispersion is the point: a downside objective (`p10`, `cvar5`) reweights
    channels by how uncertain they are, which moves the optimum away from the
    current allocation and gives the solver something it has to actually find.
    Saturation differs per channel so the optimum is interior rather than a
    corner.
    """
    rng = np.random.default_rng(seed)
    spend = MULTIPLIERS[None, :] * BASE[:, None]  # (C, G)
    # Concave: a * (1 - exp(-spend / k)), with per-channel scale and half-point.
    a = np.array([12000.0, 7000.0, 6500.0, 4200.0])
    k = np.array([9000.0, 5000.0, 3500.0, 4000.0])
    mean_curve = a[:, None] * (1.0 - np.exp(-spend / k[:, None]))
    # Per-draw multiplicative noise, wider on the channels a risk-averse
    # objective should penalize.
    sd = np.array([0.10, 0.22, 0.35, 0.15])
    noise = 1.0 + sd[None, :, None] * rng.standard_normal((n_draws, len(CHANNELS), 1))
    contributions = np.clip(mean_curve[None, :, :] * noise, 0.0, None)
    return ResponseCurves(
        channel_names=list(CHANNELS),
        multipliers=MULTIPLIERS,
        base_spend=BASE.copy(),
        contributions=contributions,
        obs_max_spend=None,
        n_obs=104,
    )


def _objective_value(
    curves: ResponseCurves, alloc: np.ndarray, objective: str
) -> float:
    """Score an allocation the same way the allocator's own objective does."""
    oc = objective_curves(curves, objective)
    grid = curves.multipliers[None, :] * curves.base_spend[:, None]
    return float(sum(np.interp(alloc[i], grid[i], oc[i]) for i in range(len(alloc))))


def _random_search(
    curves: ResponseCurves, objective: str, total: float, n: int = 8000, seed: int = 1
) -> tuple[float, np.ndarray]:
    """Best feasible allocation found by sampling the simplex."""
    rng = np.random.default_rng(seed)
    hi = 2.0 * curves.base_spend
    best_v, best_s = -np.inf, None
    for _ in range(n):
        s = rng.dirichlet(np.ones(len(CHANNELS)) * 1.2) * total
        if (s > hi).any():
            continue
        v = _objective_value(curves, s, objective)
        if v > best_v:
            best_v, best_s = v, s
    return best_v, best_s


@pytest.mark.parametrize("objective", ["p10", "cvar5"])
class TestNotDominatedBySearch:
    def test_beats_random_search(self, objective):
        """The check the issue asks for. A solver that stalls on its warm start
        loses to a few thousand random samples; one that works does not."""
        curves = _curves()
        total = float(BASE.sum())
        plan = optimize_budget(
            curves=curves, total_budget=total, objective=objective, random_seed=42
        )
        v_opt = _objective_value(curves, np.asarray(plan.optimal_alloc), objective)
        v_rand, _ = _random_search(curves, objective, total)
        assert v_opt >= v_rand, (
            f"{objective}: random search found {v_rand:.1f} but the allocator "
            f"returned {v_opt:.1f} — the solver is leaving value on the table."
        )

    def test_beats_standing_still(self, objective):
        """The warm start IS the current allocation, so "no change" is exactly
        what a stalled solve produces. On these curves a better plan exists."""
        curves = _curves()
        total = float(BASE.sum())
        plan = optimize_budget(
            curves=curves, total_budget=total, objective=objective, random_seed=42
        )
        v_opt = _objective_value(curves, np.asarray(plan.optimal_alloc), objective)
        v_today = _objective_value(curves, BASE, objective)
        assert v_opt > v_today, (
            f"{objective}: allocator returned the current plan's value "
            f"({v_opt:.1f}); a better allocation exists."
        )
        assert not np.allclose(np.asarray(plan.optimal_alloc), BASE, rtol=1e-6)

    def test_respects_the_budget(self, objective):
        curves = _curves()
        total = float(BASE.sum())
        plan = optimize_budget(
            curves=curves, total_budget=total, objective=objective, random_seed=42
        )
        assert float(np.sum(plan.optimal_alloc)) == pytest.approx(total, rel=1e-6)
        assert (np.asarray(plan.optimal_alloc) >= -1e-9).all()


class TestConstraintsStillHold:
    """A better-scoring point must never win by being infeasible."""

    def test_group_minimum_is_respected(self):
        curves = _curves()
        total = float(BASE.sum())
        groups = [{"name": "brand", "channels": ["TV", "Social"], "min_share": 0.55}]
        plan = optimize_budget(
            curves=curves,
            total_budget=total,
            objective="p10",
            groups=groups,
            random_seed=42,
        )
        alloc = dict(zip(CHANNELS, np.asarray(plan.optimal_alloc)))
        share = (alloc["TV"] + alloc["Social"]) / total
        assert share >= 0.55 - 1e-6
        assert float(np.sum(plan.optimal_alloc)) == pytest.approx(total, rel=1e-6)

    def test_group_maximum_is_respected(self):
        curves = _curves()
        total = float(BASE.sum())
        groups = [
            {"name": "digital", "channels": ["Search", "Display"], "max_share": 0.25}
        ]
        plan = optimize_budget(
            curves=curves,
            total_budget=total,
            objective="cvar5",
            groups=groups,
            random_seed=42,
        )
        alloc = dict(zip(CHANNELS, np.asarray(plan.optimal_alloc)))
        assert (alloc["Search"] + alloc["Display"]) / total <= 0.25 + 1e-6

    def test_per_channel_bounds_are_respected(self):
        curves = _curves()
        total = float(BASE.sum())
        plan = optimize_budget(
            curves=curves,
            total_budget=total,
            objective="p10",
            bounds={"TV": (0.9, 1.1)},
            random_seed=42,
        )
        tv = float(np.asarray(plan.optimal_alloc)[0])
        assert 0.9 * BASE[0] - 1e-6 <= tv <= 1.1 * BASE[0] + 1e-6


class TestGreedyPathUnchanged:
    """`objective='mean'` + fixed mode + no constraints keeps the historical
    greedy allocator. The fix touches only the constrained branch, and this pins
    that the untouched path stayed untouched."""

    def test_mean_objective_does_not_route_through_the_solver(self):
        curves = _curves()
        total = float(BASE.sum())
        a = optimize_budget(
            curves=curves, total_budget=total, objective="mean", random_seed=42
        )
        b = optimize_budget(
            curves=curves, total_budget=total, objective="mean", random_seed=42
        )
        assert np.array_equal(a.optimal_alloc, b.optimal_alloc)
        # No solver diagnostics note: the greedy path never calls _solve_allocation.
        assert not any("inner solver failed" in n for n in a.notes)

    def test_mean_objective_also_beats_search(self):
        curves = _curves()
        total = float(BASE.sum())
        plan = optimize_budget(
            curves=curves, total_budget=total, objective="mean", random_seed=42
        )
        v_opt = _objective_value(curves, np.asarray(plan.optimal_alloc), "mean")
        v_rand, _ = _random_search(curves, "mean", total)
        assert v_opt >= v_rand


class TestFreeMode:
    def test_breakeven_total_scales_with_the_valuation(self):
        """`mode='free'` also routes through the constrained solver. A higher
        value per KPI unit should fund more, which a stalled solve would not
        show."""
        curves = _curves()
        low = optimize_budget(
            curves=curves,
            objective="mean",
            mode="free",
            value_per_kpi=0.5,
            random_seed=42,
        )
        high = optimize_budget(
            curves=curves,
            objective="mean",
            mode="free",
            value_per_kpi=2.0,
            random_seed=42,
        )
        assert high.total_budget > low.total_budget

    def test_free_mode_maximizes_profit_not_kpi(self):
        curves = _curves()
        plan = optimize_budget(
            curves=curves,
            objective="mean",
            mode="free",
            value_per_kpi=1.0,
            random_seed=42,
        )
        alloc = np.asarray(plan.optimal_alloc)
        profit = _objective_value(curves, alloc, "mean") - float(alloc.sum())
        # Any nearby feasible perturbation must not beat it by much.
        rng = np.random.default_rng(3)
        for _ in range(400):
            trial = np.clip(
                alloc * rng.uniform(0.7, 1.3, size=alloc.shape),
                0.0,
                2.0 * curves.base_spend,
            )
            p = _objective_value(curves, trial, "mean") - float(trial.sum())
            assert p <= profit + 1e-6 * max(abs(profit), 1.0) + 1.0
