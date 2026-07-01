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


def _slsqp_allocate(r, gr, k, B, value, *, mode, cap, x0, n_starts, seed):
    """Generic multi-start SLSQP allocator on a response callable ``r``."""
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
    bounds = [(0.0, ub)] * k
    if mode == "fixed":
        neg = lambda s: -value * r(s)  # noqa: E731
        neg_jac = lambda s: -value * gr(s)  # noqa: E731
        cons = [
            {
                "type": "eq",
                "fun": lambda s: float(np.sum(s)) - float(B),
                "jac": lambda s: np.ones(k),
            }
        ]
        profit = lambda s: value * r(s) - float(B)  # noqa: E731
    elif mode == "free":
        neg = lambda s: -(value * r(s) - float(np.sum(s)))  # noqa: E731
        neg_jac = lambda s: -(value * gr(s) - np.ones(k))  # noqa: E731
        cons = []
        profit = lambda s: value * r(s) - float(np.sum(s))  # noqa: E731
    else:
        raise ValueError(f"mode must be 'fixed' or 'free', got {mode!r}")

    best_a: np.ndarray | None = None
    best_p = -np.inf
    for s0 in _starts(k, B, ub, mode, x0, n_starts, seed):
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
    if best_a is None:  # every solve failed — fall back to the base start
        best_a = _starts(k, B, ub, mode, x0, 1, seed)[0]
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
) -> tuple[np.ndarray, float]:
    """Optimal allocation + profit under one posterior draw's params."""
    r, gr = _surface_fns(params)
    k = int(np.asarray(params["beta"]).shape[0])
    return _slsqp_allocate(
        r, gr, k, B, value, mode=mode, cap=cap, x0=x0, n_starts=n_starts, seed=seed
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
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Expected regret of acting on the consensus allocation (guide §7.4).

    ``regret_d = profit(best-for-draw-d under d) - profit(consensus under d) >= 0``
    — the profit still on the table from posterior uncertainty. Returns
    ``(E[regret], consensus_alloc, alloc_sd, optimal_profit_sd)``.
    """
    allocs, profits, draws = thompson_wave(
        post, B, value, q=q, seed=seed, mode=mode, cap=cap, n_starts=n_starts
    )
    consensus = allocs.mean(0)
    if mode == "fixed" and consensus.sum() > 0:
        consensus = consensus * (B / consensus.sum())

    regrets = np.empty(len(draws))
    optimal = np.empty(len(draws))
    for t, d in enumerate(draws):
        r, _ = _surface_fns(post.draw_params(int(d)))
        if mode == "fixed":
            p_consensus = value * r(consensus) - float(B)
        else:
            p_consensus = value * r(consensus) - float(consensus.sum())
        best = max(float(profits[t]), p_consensus)  # warm-started max => regret >= 0
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
    noise: float = 0.6,
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
    """
    _, base_value = posterior_optimal_allocation(
        post, B, value, q=q, mode=mode, cap=cap, seed=seed
    )
    rng = np.random.default_rng(seed)
    a_geo_samples = post.samples.get("a_geo")
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
            a = float(post.samples["A"][int(d)])
            sa = float(post.samples["sigma_a"][int(d)])
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
