"""Layer 4 — planner / acquisition.

The decision is ``max_s  value * R(s) - 1^T s`` under one of two regimes:

* **fixed budget** (``mode="fixed"``): ``1^T s = B`` — a pure simplex.
* **free budget** (``mode="free"``): ``0 <= s_c <= cap`` — the total is itself a
  decision; each channel self-funds until its marginal ROAS hits 1.

Every routine here differentiates the **same** ``incremental`` surface the
likelihood fits (via :data:`mmm_framework.continuous_learning.surface.grad_incremental`),
so the allocator can never disagree with the fitted model. The surface is
non-concave (negative ``gamma``), so the allocator multi-starts and keeps the
best.

Acquisition / readouts:

* :func:`plan_from_posterior` — ONE Thompson pass producing every per-wave
  readout coherently (recommendation, funding line at it, warm-started regret
  from it) as a :class:`PlanResult`; prefer it over calling the pieces below
  separately, which each re-sample.
* :func:`thompson_wave` — a posterior over the optimal split (the spread is the
  exploration signal; the mean is the recommendation).
* :func:`marginal_roas` — the funding line: a channel is funded where
  ``P(value * dR/ds_c > 1) > 0.5``.
* :func:`expected_regret` — ``E[regret]`` from posterior uncertainty; with
  :func:`enbs` it drives the stopping rule.
* :func:`knowledge_gradient` — decision-aware EVSI for scoring candidate *test*
  designs (the expensive path; see guide §7.5/9.1).

Import-light apart from JAX (shared with the model) and SciPy SLSQP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .design import assign_geos
from functools import lru_cache

from .model import Posterior
from .surface import (
    ACTIVATIONS,
    activation as _hill_activation,
    surface_over_rows,
    surface_value,
)


@lru_cache(maxsize=None)
def _jitted(act_fn):
    """JIT the scalar surface value and its spend-gradient for one activation.

    Cached per activation function, so the allocator compiles once for Hill and
    once for logistic (etc.) and reuses across the thousands of SLSQP/draw calls.
    ``shape`` is splatted so the same jit serves any parameter count.
    """

    def _val(spend, beta, gamma, *shape):
        return surface_value(spend, beta, gamma, act_fn, shape)

    return jax.jit(_val), jax.jit(jax.grad(_val, argnums=0))


def _params_kernel(params: dict):
    """Extract ``(act_fn, shape_tuple)`` from either param format.

    New format (from :meth:`Posterior.draw_params`): ``{beta, gamma, shape,
    act_fn}``. Legacy Hill dict (tests / acquisition / truth worlds):
    ``{beta, kappa, alpha, gamma}`` -> the Hill activation with ``(kappa, alpha)``.
    """
    if "act_fn" in params:
        return params["act_fn"], tuple(
            jnp.asarray(a, dtype=float) for a in params["shape"]
        )
    return _hill_activation, (
        jnp.asarray(params["kappa"], dtype=float),
        jnp.asarray(params["alpha"], dtype=float),
    )


# ── allocation under a single set of params / a mean surface ───────────────────


def _surface_fns(params: dict):
    """``(R, gR)`` python callables for one draw's params (jit cache reused)."""
    beta = jnp.asarray(params["beta"], dtype=float)
    gamma = jnp.asarray(params["gamma"], dtype=float)
    act_fn, shape = _params_kernel(params)
    val, grad = _jitted(act_fn)

    def r(s: np.ndarray) -> float:
        return float(val(jnp.asarray(s, dtype=float), beta, gamma, *shape))

    def gr(s: np.ndarray) -> np.ndarray:
        return np.asarray(
            grad(jnp.asarray(s, dtype=float), beta, gamma, *shape), dtype=float
        )

    return r, gr


def _mean_surface_fns(post: Posterior, draws: np.ndarray):
    """``(R, gR)`` for the posterior-mean surface over ``draws`` (any activation)."""
    names, act_fn = ACTIVATIONS[post.activation]
    betas = jnp.asarray(post.samples["beta"][draws], dtype=float)  # (q, K)
    gammas = jnp.asarray(
        np.stack([post.gamma_matrix(int(d)) for d in draws]), dtype=float
    )  # (q, K, K)
    shapes = tuple(jnp.asarray(post.samples[nm][draws], dtype=float) for nm in names)
    in_axes = (0, 0) + (0,) * len(shapes)

    def _mean_r(s):
        per = jax.vmap(
            lambda b, g, *sh: surface_value(s, b, g, act_fn, sh), in_axes=in_axes
        )(betas, gammas, *shapes)
        return jnp.mean(per)

    _grad_mean_r = jax.grad(_mean_r)

    def r(s: np.ndarray) -> float:
        return float(_mean_r(jnp.asarray(s, dtype=float)))

    def gr(s: np.ndarray) -> np.ndarray:
        return np.asarray(_grad_mean_r(jnp.asarray(s, dtype=float)), dtype=float)

    return r, gr


def response_grid(
    post: Posterior, spend_matrix: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    """Incremental response for every ``(spend row, draw)`` pair -> ``(G, D)``.

    Activation-agnostic (reads ``post.activation``): the visualizations use this
    to build the posterior mean/uncertainty surfaces for any activation family,
    so the same animation code renders a Hill world or a logistic one.
    """
    names, act_fn = ACTIVATIONS[post.activation]
    betas = jnp.asarray(post.samples["beta"][draws], dtype=float)  # (D, K)
    gammas = jnp.asarray(
        np.stack([post.gamma_matrix(int(d)) for d in draws]), dtype=float
    )  # (D, K, K)
    shapes = tuple(jnp.asarray(post.samples[nm][draws], dtype=float) for nm in names)
    in_axes = (0, 0) + (0,) * len(shapes)

    def _over_draws(s):
        return jax.vmap(
            lambda b, g, *sh: surface_value(s, b, g, act_fn, sh), in_axes=in_axes
        )(betas, gammas, *shapes)

    return np.asarray(jax.vmap(_over_draws)(jnp.asarray(spend_matrix, dtype=float)))


def _starts(k, B, ub, mode, x0, n_starts, seed):
    rng = np.random.default_rng(seed)
    out = [np.minimum(np.full(k, B / k), ub)]
    if x0 is not None:
        out.append(np.clip(np.asarray(x0, dtype=float), 0.0, ub))
    while len(out) < max(1, n_starts):
        out.append(np.clip(rng.dirichlet(np.ones(k)) * B, 0.0, ub))
    return out


def _validate_group_budgets(
    group_budgets, k: int, B: float, ub: float, mode: str
) -> list[tuple[list[int], float]]:
    """Validate grouped budget constraints (arms: parent budget fixed, mix free).

    Groups must be disjoint, in-range, and jointly feasible: each group's budget
    must be reachable under the per-channel cap, and in ``fixed`` mode the group
    totals cannot exceed ``B`` (the ungrouped channels must be able to absorb
    the remainder). Feasibility tolerances are RELATIVE to the budget scale
    (``1e-9 * max(1, |B|)``) so the layer behaves the same whether spend is
    scaled to O(1) or left in dollars (an absolute 1e-9 would spuriously fail
    an exactly-intended dollar-scale partition on ~1 ulp of fp error).
    """
    if not group_budgets:
        return []
    tol = 1e-9 * max(1.0, abs(float(B)))
    seen: set[int] = set()
    total = 0.0
    norm: list[tuple[list[int], float]] = []
    for g, (idx_raw, bg_raw) in enumerate(group_budgets):
        idx = [int(i) for i in idx_raw]
        if not idx:
            raise ValueError(f"group budget {g} has no channel indices")
        for i in idx:
            if not 0 <= i < k:
                raise ValueError(
                    f"group budget {g}: channel index {i} out of range for "
                    f"{k} channels"
                )
            if i in seen:
                raise ValueError(
                    f"group budgets overlap on channel index {i}; groups must "
                    "be disjoint"
                )
            seen.add(i)
        bg = float(bg_raw)
        if not np.isfinite(bg) or bg < 0:
            raise ValueError(f"group budget {g}: budget must be >= 0, got {bg}")
        if len(idx) * ub < bg - tol:
            raise ValueError(
                f"group budget {g} is infeasible: {len(idx)} channels capped at "
                f"{ub:.4g} can only reach {len(idx) * ub:.4g} < {bg:.4g}"
            )
        total += bg
        norm.append((idx, bg))
    if mode == "fixed":
        if total > float(B) + tol:
            raise ValueError(
                f"group budgets sum to {total:.4g} > B={B:.4g}; they must fit "
                "inside the total budget"
            )
        rest = k - len(seen)
        remaining = float(B) - total
        if rest * ub < remaining - tol:
            raise ValueError(
                f"group budgets leave {remaining:.4g} for {rest} ungrouped "
                f"channels capped at {ub:.4g} — infeasible"
            )
    return norm


def _apply_group_starts(starts, groups, k, B, ub, mode):
    """Rescale each start so every group sums to its budget (better SLSQP seeds)."""
    if not groups:
        return starts
    grouped = sorted({i for idx, _ in groups for i in idx})
    rest = [i for i in range(k) if i not in grouped]
    total = sum(bg for _, bg in groups)
    out = []
    for s in starts:
        s = np.asarray(s, dtype=float).copy()
        for idx, bg in groups:
            sub = s[idx]
            tot = float(sub.sum())
            if tot > 1e-12:
                s[idx] = np.clip(sub * (bg / tot), 0.0, ub)
            else:
                s[idx] = np.clip(np.full(len(idx), bg / len(idx)), 0.0, ub)
        if mode == "fixed" and rest:
            rem = float(B) - total
            sub = s[rest]
            tot = float(sub.sum())
            if tot > 1e-12:
                s[rest] = np.clip(sub * (rem / tot), 0.0, ub)
            else:
                s[rest] = np.clip(np.full(len(rest), rem / len(rest)), 0.0, ub)
        out.append(s)
    return out


def _slsqp_allocate(
    r,
    gr,
    k,
    B,
    value,
    *,
    mode,
    cap,
    x0,
    n_starts,
    seed,
    group_budgets=None,
    only_x0=False,
):
    """Generic multi-start SLSQP allocator on a response callable ``r``.

    ``group_budgets`` adds one equality constraint ``sum(s[idx]) == B_g`` per
    ``(idx, B_g)`` group (sub-channel arms: the parent budget is fixed, the mix
    within it is free). ``only_x0=True`` runs a single solve warm-started from
    ``x0`` (the expected-regret second pass).
    """
    ub = float(cap) if cap is not None else float(B)
    if mode == "fixed" and k * ub < float(B) - 1e-9:
        # The budget simplex sum(s) = B is infeasible: even maxed out, the K
        # channels can only reach k*cap < B. Fail loudly rather than return an
        # off-simplex fallback whose profit (= value*R - B) is meaningless.
        raise ValueError(
            f"fixed-budget allocation is infeasible: cap={cap} caps the total at "
            f"{k * ub:.4g} < B={B}. Raise cap to >= B/k ({float(B) / k:.4g}) "
            f"or use mode='free'."
        )
    groups = _validate_group_budgets(group_budgets, k, float(B), ub, mode)
    grouped_idx = {i for idx, _ in groups for i in idx}
    group_total = sum(bg for _, bg in groups)
    bounds = [(0.0, ub)] * k
    if mode == "fixed":
        neg = lambda s: -value * r(s)  # noqa: E731
        neg_jac = lambda s: -value * gr(s)  # noqa: E731
        cons = []
        # When the groups partition ALL channels and exhaust B, the global
        # budget constraint is implied by the group constraints — drop it to
        # avoid a degenerate (redundant) constraint set. Tolerance is relative
        # to the budget scale (dollar-scale callers accumulate ~ulp fp error).
        covers_all = len(grouped_idx) == k and abs(group_total - float(B)) <= (
            1e-9 * max(1.0, abs(float(B)))
        )
        if not covers_all:
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda s: float(np.sum(s)) - float(B),
                    "jac": lambda s: np.ones(k),
                }
            )
        profit = lambda s: value * r(s) - float(B)  # noqa: E731
    elif mode == "free":
        neg = lambda s: -(value * r(s) - float(np.sum(s)))  # noqa: E731
        neg_jac = lambda s: -(value * gr(s) - np.ones(k))  # noqa: E731
        cons = []
        profit = lambda s: value * r(s) - float(np.sum(s))  # noqa: E731
    else:
        raise ValueError(f"mode must be 'fixed' or 'free', got {mode!r}")

    for idx, bg in groups:
        ind = np.zeros(k)
        ind[idx] = 1.0
        cons.append(
            {
                "type": "eq",
                "fun": lambda s, _i=tuple(idx), _b=bg: (
                    float(np.sum(np.asarray(s)[list(_i)])) - _b
                ),
                "jac": lambda s, _ind=ind: _ind,
            }
        )

    if only_x0:
        if x0 is None:
            raise ValueError("only_x0=True requires an x0 warm start")
        starts = [np.clip(np.asarray(x0, dtype=float), 0.0, ub)]
    else:
        starts = _starts(k, B, ub, mode, x0, n_starts, seed)
    starts = _apply_group_starts(starts, groups, k, B, ub, mode)

    best_a: np.ndarray | None = None
    best_p = -np.inf
    for s0 in starts:
        try:
            res = minimize(
                neg,
                s0,
                jac=neg_jac,
                bounds=bounds,
                constraints=cons,
                method="SLSQP",
                options={"maxiter": 200, "ftol": 1e-9},
            )
            s = np.clip(np.asarray(res.x, dtype=float), 0.0, ub)
        except Exception:
            continue
        p = profit(s)
        if np.isfinite(p) and p > best_p:
            best_p = p
            best_a = s
    if best_a is None:  # every solve failed — fall back to the first start
        best_a = starts[0]
        best_p = profit(best_a)
    return best_a, float(best_p)


def allocate_under_sample(
    params: dict[str, np.ndarray],
    B: float,
    value: float,
    *,
    x0: np.ndarray | None = None,
    n_starts: int = 3,
    seed: int = 0,
    mode: str = "fixed",
    cap: float | None = None,
    group_budgets: list[tuple[list[int], float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Optimal allocation + profit under one posterior draw's params.

    ``group_budgets`` fixes sub-budgets over channel groups (arms):
    ``[(indices, B_g), ...]`` adds ``sum(s[indices]) == B_g`` per group.
    """
    r, gr = _surface_fns(params)
    k = int(np.asarray(params["beta"]).shape[0])
    return _slsqp_allocate(
        r,
        gr,
        k,
        B,
        value,
        mode=mode,
        cap=cap,
        x0=x0,
        n_starts=n_starts,
        seed=seed,
        group_budgets=group_budgets,
    )


def _sample_draws(n_draws: int, q: int, seed: int) -> np.ndarray:
    if q >= n_draws:
        return np.arange(n_draws)
    return np.random.default_rng(seed).choice(n_draws, size=q, replace=False)


# ── acquisition ────────────────────────────────────────────────────────────────


def thompson_wave(
    post: Posterior,
    B: float,
    value: float,
    *,
    q: int = 300,
    seed: int = 0,
    mode: str = "fixed",
    cap: float | None = None,
    n_starts: int = 3,
    group_budgets: list[tuple[list[int], float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A posterior over the optimal split — solve the allocation for ``q`` draws.

    Returns ``(allocs, profits, draws)``: ``allocs`` is ``(q, K)`` (the mean is
    the recommended allocation, the spread is the exploration signal),
    ``profits`` is ``(q,)`` (each draw's optimum under itself), and ``draws`` are
    the sampled draw indices.
    """
    draws = _sample_draws(post.n_draws, q, seed)
    k = post.n_channels
    allocs = np.empty((len(draws), k))
    profits = np.empty(len(draws))
    for t, d in enumerate(draws):
        r, gr = _surface_fns(post.draw_params(int(d)))
        a, p = _slsqp_allocate(
            r,
            gr,
            k,
            B,
            value,
            mode=mode,
            cap=cap,
            x0=None,
            n_starts=n_starts,
            seed=seed + int(d),
            group_budgets=group_budgets,
        )
        allocs[t] = a
        profits[t] = p
    return allocs, profits, draws


def recommend_allocation(
    post: Posterior, B: float, value: float, **kwargs
) -> np.ndarray:
    """The recommended allocation — the Thompson posterior mean split."""
    allocs, _, _ = thompson_wave(post, B, value, **kwargs)
    rec = allocs.mean(0)
    if kwargs.get("mode", "fixed") == "fixed" and rec.sum() > 0:
        rec = rec * (B / rec.sum())  # project back onto the budget simplex
    return rec


def marginal_roas(
    post: Posterior,
    alloc: np.ndarray,
    value: float,
    *,
    q: int = 300,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior of ``value * dR/ds_c`` at ``alloc`` — the funding line.

    Returns ``(mean_mroas, prob_above_line, draws_mroas)``. A channel is funded
    where ``prob_above_line > 0.5``.
    """
    draws = _sample_draws(post.n_draws, q, seed)
    k = post.n_channels
    m = np.empty((len(draws), k))
    for t, d in enumerate(draws):
        _, gr = _surface_fns(post.draw_params(int(d)))
        m[t] = value * gr(alloc)
    return m.mean(0), (m > 1.0).mean(0), m


def expected_regret(
    post: Posterior,
    B: float,
    value: float,
    *,
    q: int = 300,
    seed: int = 2,
    mode: str = "fixed",
    cap: float | None = None,
    n_starts: int = 3,
    group_budgets: list[tuple[list[int], float]] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Expected regret of acting on the consensus allocation (guide §7.4).

    ``regret_d = profit(best-for-draw-d under d) - profit(consensus under d) >= 0``
    — the profit still on the table from posterior uncertainty. Each draw's
    "best" is ``max(cold multistart, a solve warm-started FROM the consensus,
    profit at the consensus)`` (review fix F3): the warm-started second pass
    catches per-draw optima the cold multistart missed near the consensus, so
    the regret is less biased low and the ENBS stop fires less prematurely.
    Returns ``(E[regret], consensus_alloc, alloc_sd, optimal_profit_sd)``.
    """
    allocs, profits, draws = thompson_wave(
        post,
        B,
        value,
        q=q,
        seed=seed,
        mode=mode,
        cap=cap,
        n_starts=n_starts,
        group_budgets=group_budgets,
    )
    consensus = allocs.mean(0)
    if mode == "fixed" and consensus.sum() > 0:
        consensus = consensus * (B / consensus.sum())

    k = post.n_channels
    regrets = np.empty(len(draws))
    optimal = np.empty(len(draws))
    for t, d in enumerate(draws):
        r, gr = _surface_fns(post.draw_params(int(d)))
        if mode == "fixed":
            p_consensus = value * r(consensus) - float(B)
        else:
            p_consensus = value * r(consensus) - float(consensus.sum())
        _, warm_p = _slsqp_allocate(
            r,
            gr,
            k,
            B,
            value,
            mode=mode,
            cap=cap,
            x0=consensus,
            n_starts=1,
            seed=seed + int(d),
            group_budgets=group_budgets,
            only_x0=True,
        )
        best = max(float(profits[t]), warm_p, p_consensus)  # => regret >= 0
        regrets[t] = max(0.0, best - p_consensus)
        optimal[t] = best
    return float(regrets.mean()), consensus, allocs.std(0), float(optimal.std())


def posterior_optimal_allocation(
    post: Posterior,
    B: float,
    value: float,
    *,
    q: int = 200,
    seed: int = 3,
    mode: str = "fixed",
    cap: float | None = None,
    n_starts: int = 4,
    group_budgets: list[tuple[list[int], float]] | None = None,
) -> tuple[np.ndarray, float]:
    """``argmax_a E_post[profit(a)]`` and its value — optimize the mean surface."""
    draws = _sample_draws(post.n_draws, q, seed)
    r, gr = _mean_surface_fns(post, draws)
    return _slsqp_allocate(
        r,
        gr,
        post.n_channels,
        B,
        value,
        mode=mode,
        cap=cap,
        x0=None,
        n_starts=n_starts,
        seed=seed,
        group_budgets=group_budgets,
    )


# ── one Thompson pass, all readouts (review fix F4) ─────────────────────────────


@dataclass
class PlanResult:
    """Every per-wave decision readout, derived from ONE Thompson sample.

    Historically ``recommend``/``funding``/``regret`` each re-sampled with a
    different seed (planner seeds 0/1/2), so the funded set and the reported
    regret referred to *different* consensus vectors and paid 2-3x the SLSQP
    cost. Here all readouts share the same ``q`` draws: ``recommendation`` is
    the Thompson mean (rescaled onto the budget simplex in ``fixed`` mode),
    ``consensus`` is that same vector, the funding line is evaluated *at* it,
    and the regret pass is warm-started *from* it.
    """

    channels: list[str]
    B: float
    value: float
    mode: str
    recommendation: np.ndarray  # (K,) the acted-on allocation (== consensus)
    consensus: np.ndarray  # (K,) alias kept for regret-readout parity
    allocs: np.ndarray  # (q, K) per-draw optima
    profits: np.ndarray  # (q,) per-draw optimal profits (cold pass)
    draws: np.ndarray  # (q,) posterior draw indices used everywhere
    mroas_mean: np.ndarray  # (K,) mean of value * dR/ds at the recommendation
    prob_above_line: np.ndarray  # (K,) P(value * dR/ds_c > 1)
    mroas_draws: np.ndarray  # (q, K) per-draw marginal ROAS at the recommendation
    e_regret: float
    alloc_sd: np.ndarray  # (K,) Thompson spread (exploration signal)
    profit_sd: float

    def to_dict(self) -> dict[str, Any]:
        """A small JSON-safe snapshot (summary stats, not the draw matrices)."""
        return {
            "channels": list(self.channels),
            "B": float(self.B),
            "value": float(self.value),
            "mode": self.mode,
            "recommendation": [float(x) for x in self.recommendation],
            "alloc_sd": [float(x) for x in self.alloc_sd],
            "mroas_mean": [float(x) for x in self.mroas_mean],
            "prob_above_line": [float(x) for x in self.prob_above_line],
            "funded": [bool(p > 0.5) for p in self.prob_above_line],
            "e_regret": float(self.e_regret),
            "profit_sd": float(self.profit_sd),
            "n_draws": int(self.allocs.shape[0]),
        }


def plan_from_posterior(
    post: Posterior,
    B: float,
    value: float,
    *,
    q: int = 300,
    seed: int = 0,
    mode: str = "fixed",
    cap: float | None = None,
    n_starts: int = 3,
    group_budgets: list[tuple[list[int], float]] | None = None,
) -> PlanResult:
    """One Thompson pass producing every decision readout coherently (fix F4).

    Runs :func:`thompson_wave` once; the recommendation is the draw-mean
    (rescaled to the simplex in ``fixed`` mode); the funding line and the
    warm-started regret pass reuse the SAME draw indices, so the funded set,
    the consensus, and ``E[regret]`` all describe one allocation.
    """
    allocs, profits, draws = thompson_wave(
        post,
        B,
        value,
        q=q,
        seed=seed,
        mode=mode,
        cap=cap,
        n_starts=n_starts,
        group_budgets=group_budgets,
    )
    rec = allocs.mean(0)
    if mode == "fixed" and rec.sum() > 0:
        rec = rec * (B / rec.sum())  # project back onto the budget simplex

    k = post.n_channels
    mroas = np.empty((len(draws), k))
    regrets = np.empty(len(draws))
    optimal = np.empty(len(draws))
    for t, d in enumerate(draws):
        r, gr = _surface_fns(post.draw_params(int(d)))
        mroas[t] = value * gr(rec)
        if mode == "fixed":
            p_consensus = value * r(rec) - float(B)
        else:
            p_consensus = value * r(rec) - float(rec.sum())
        _, warm_p = _slsqp_allocate(
            r,
            gr,
            k,
            B,
            value,
            mode=mode,
            cap=cap,
            x0=rec,
            n_starts=1,
            seed=seed + int(d),
            group_budgets=group_budgets,
            only_x0=True,
        )
        best = max(float(profits[t]), warm_p, p_consensus)  # => regret >= 0
        regrets[t] = max(0.0, best - p_consensus)
        optimal[t] = best

    return PlanResult(
        channels=list(post.channels),
        B=float(B),
        value=float(value),
        mode=mode,
        recommendation=rec,
        consensus=rec,
        allocs=allocs,
        profits=profits,
        draws=np.asarray(draws),
        mroas_mean=mroas.mean(0),
        prob_above_line=(mroas > 1.0).mean(0),
        mroas_draws=mroas,
        e_regret=float(regrets.mean()),
        alloc_sd=allocs.std(0),
        profit_sd=float(optimal.std()),
    )


# ── stopping: ENBS ──────────────────────────────────────────────────────────────


def enbs(
    e_regret: float, *, margin: float, population: float, wave_cost: float
) -> float:
    """Expected net benefit of sampling ``= E[regret] * margin * population - cost``.

    ``E[regret]`` is profit-per-affected-unit on the response-curve scale;
    ``margin`` and ``population`` convert it to a total dollar value of resolving
    the uncertainty, which a wave must beat to be worth running.
    """
    return float(e_regret) * float(margin) * float(population) - float(wave_cost)


def should_stop(
    e_regret: float, *, margin: float, population: float, wave_cost: float
) -> tuple[bool, float]:
    """``(stop, enbs)`` — stop when no wave's expected value clears its cost."""
    value = enbs(e_regret, margin=margin, population=population, wave_cost=wave_cost)
    return value <= 0.0, value


# ── knowledge gradient (decision-aware EVSI) ────────────────────────────────────


def _default_noise(post: Posterior) -> float:
    """Fantasy observation noise: the posterior mean of ``sigma`` (fix F7).

    Falls back to ``0.6`` (the legacy DGP value) only when the posterior has no
    ``sigma`` site (e.g. a summaries-only fit, which never samples the panel
    noise scale).

    Raises for a non-Gaussian posterior: the knowledge-gradient fantasies are
    generated as ``y = mu + Normal(0, noise)``, which is silently wrong for a
    count likelihood (a NegBinomial posterior has ``phi``, no ``sigma`` — the
    0.6 fallback would fire and produce plausible-looking nonsense).
    """
    likelihood = getattr(post, "likelihood", "normal")
    if likelihood != "normal":
        raise NotImplementedError(
            "knowledge_gradient fantasy generation is Gaussian (y = mu + "
            f"Normal(0, noise)); this posterior was fit with likelihood="
            f"{likelihood!r}. Use the activation-agnostic decision readouts "
            "(thompson_wave / recommend_allocation / marginal_roas / "
            "expected_regret), which read only the surface parameters — note "
            "that for a count likelihood the marginal readouts are on the "
            "LATENT surface scale (the observable count mean is softplus(mu), "
            "derivative sigmoid(mu) ≈ 1 for mu >> 1 but < 1 at low counts; "
            "see model.fit's likelihood note)."
        )
    s = post.samples.get("sigma")
    return float(np.mean(s)) if s is not None else 0.6


def knowledge_gradient(
    post: Posterior,
    candidate_design: np.ndarray,
    refit_fn,
    B: float,
    value: float,
    *,
    n_fantasy: int = 10,
    t_test: int = 10,
    n_geo: int | None = None,
    noise: float | None = None,
    mode: str = "fixed",
    cap: float | None = None,
    q: int = 120,
    seed: int = 4,
) -> float:
    """One-step-lookahead EVSI for a candidate *test* design (guide §7.5).

    ``KG(d) = E_y[ max_a profit(a | posterior updated with fantasised y) ]
              - max_a profit(a | current posterior)``

    For each fantasy: draw params, simulate the candidate design's outcomes,
    refit via ``refit_fn(extra_spend, extra_geo_idx, extra_y) -> Posterior``, and
    re-optimize. **Expensive** — ``refit_fn`` runs a (short) NUTS chain per
    fantasy; in production swap it for a Laplace/conjugate update (guide §9.1).

    ``noise=None`` (default) fantasizes with the posterior mean of ``sigma`` —
    the fitted observation noise — rather than a hard-coded constant.

    Requires a panel-fitted posterior: fantasies simulate geo-week outcomes off
    the geo intercepts (``a_geo`` or the ``A``/``sigma_a`` hypers), which a
    summaries-only fit never samples.
    """
    a_geo_samples = post.samples.get("a_geo")
    if a_geo_samples is None and (
        "A" not in post.samples or "sigma_a" not in post.samples
    ):
        raise ValueError(
            "knowledge_gradient requires a panel-fitted posterior: fantasy "
            "waves simulate geo-week outcomes from the geo intercepts, but "
            "this posterior has no 'a_geo'/'A'/'sigma_a' sites (a "
            "summaries-only fit skips the geo-intercept plate entirely)"
        )
    if noise is None:
        noise = _default_noise(post)
    _, base_value = posterior_optimal_allocation(
        post, B, value, q=q, mode=mode, cap=cap, seed=seed
    )
    rng = np.random.default_rng(seed)
    if n_geo is None:
        n_geo = int(a_geo_samples.shape[1]) if a_geo_samples is not None else 60

    fantasy_values = []
    draws = _sample_draws(post.n_draws, n_fantasy, seed + 1)
    for d in draws:
        params = post.draw_params(int(d))
        geo_per_geo, _ = assign_geos(candidate_design, n_geo, rng)
        spend = np.repeat(geo_per_geo[None, :, :], t_test, axis=0).reshape(
            t_test * n_geo, -1
        )
        geo = np.tile(np.arange(n_geo), t_test)
        if a_geo_samples is not None and a_geo_samples.shape[1] == n_geo:
            a_geo = a_geo_samples[int(d)]
        else:
            a_samples = post.samples.get("A")
            sa_samples = post.samples.get("sigma_a")
            if a_samples is None or sa_samples is None:
                raise ValueError(
                    f"knowledge_gradient: n_geo={n_geo} does not match the "
                    f"posterior's a_geo panel "
                    f"({int(a_geo_samples.shape[1])} geos) and the posterior "
                    "has no 'A'/'sigma_a' hypers to draw fresh intercepts from"
                )
            a = float(a_samples[int(d)])
            sa = float(sa_samples[int(d)])
            a_geo = rng.normal(a, sa, n_geo)
        act_fn, shape = _params_kernel(params)
        mu = a_geo[geo] + np.asarray(
            surface_over_rows(spend, params["beta"], params["gamma"], act_fn, shape),
            dtype=float,
        )
        y = mu + rng.normal(0.0, noise, mu.shape[0])
        post2 = refit_fn(spend, geo, y)
        _, v2 = posterior_optimal_allocation(
            post2, B, value, q=q, mode=mode, cap=cap, seed=seed + int(d)
        )
        fantasy_values.append(v2)
    return float(np.mean(fantasy_values) - base_value)
