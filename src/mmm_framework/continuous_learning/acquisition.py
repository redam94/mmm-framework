"""Fast acquisition — Laplace knowledge-gradient and pure-EIG (guide §9.1/9.2).

The reference :func:`mmm_framework.continuous_learning.planner.knowledge_gradient`
refits the model with full NUTS once *per fantasy* — too slow to score many
candidate designs. This module replaces that with a **Gaussian-linear**
treatment that needs no MCMC:

1. Moment-match the current posterior over the surface parameters
   ``theta = [beta, kappa, alpha, gamma_pairs]`` to a Gaussian ``N(mu, Sigma)``.
2. Linearize ``incremental`` in ``theta`` around ``mu`` (the Laplace step). A
   candidate design's Fisher information is then
   ``Lambda = sigma^-2 * sum_cells w_c * g_c g_c^T``  with  ``g_c = d incremental / d theta``,
   and the posterior covariance after running it is
   ``Sigma_post = (Sigma^-1 + Lambda)^-1``.

From this one linear-algebra object two acquisitions fall out cheaply:

* **Laplace knowledge-gradient** (decision value, EVSI): the pre-posterior
  spread of the updated mean is ``V = Sigma - Sigma_post``; sample fantasy means
  ``theta_m ~ N(mu, V)``, re-optimize the allocation for each, and average the
  uplift over ``max_a profit(a | mu)``. Milliseconds instead of minutes.
* **Pure EIG** (information, D-/D_s-optimality): the entropy reduction of a
  parameter block ``S`` is ``0.5 * (logdet Sigma_SS - logdet Sigma_post_SS)``.
  Use the full block for D-optimality, or the ``gamma`` sub-block (``target=
  "gamma"``) for **D_s-optimality** — synergy-targeted waves the exploit-heavy
  Thompson waves under-probe.

The geo intercept is profiled out by centering the cell gradients across cells
(the per-geo baselines are randomized, so the identifying variation is
between-cell), mirroring ``planning.identification``.

Prefer the Laplace KG when the goal is the **decision**; use EIG only to shore up
decision-pivotal interactions (guide §9.2).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import planner as _planner
from .model import Pair, Posterior, pair_name
from .surface import incremental

# ── parameter packing ────────────────────────────────────────────────────────


def pack_theta(beta, kappa, alpha, gamma_pairs) -> np.ndarray:
    """Stack ``[beta, kappa, alpha, gamma_pairs]`` into one vector of length P."""
    return np.concatenate(
        [
            np.asarray(beta, float),
            np.asarray(kappa, float),
            np.asarray(alpha, float),
            np.asarray(gamma_pairs, float),
        ]
    )


def unpack_theta(theta, k: int):
    """Inverse of :func:`pack_theta` -> ``(beta, kappa, alpha, gamma_pairs)``."""
    return theta[:k], theta[k : 2 * k], theta[2 * k : 3 * k], theta[3 * k :]


def gamma_indices(k: int, pairs: list[Pair]) -> list[int]:
    """Positions of the gamma parameters within the packed vector."""
    return list(range(3 * k, 3 * k + len(pairs)))


def _gamma_matrix(gp, k: int, pairs: list[Pair]):
    g = jnp.zeros((k, k))
    for idx, (i, j) in enumerate(pairs):
        g = g.at[i, j].set(gp[idx])
    return g


def _incremental_theta(theta, spend, k: int, pairs: list[Pair]):
    beta, kappa, alpha, gp = unpack_theta(theta, k)
    return incremental(spend, beta, kappa, alpha, _gamma_matrix(gp, k, pairs))


def param_grad(theta_bar, spend_rows, k: int, pairs: list[Pair]) -> np.ndarray:
    """``d incremental / d theta`` at ``theta_bar`` for each spend row -> (n, P)."""
    g_one = jax.grad(lambda th, s: _incremental_theta(th, s, k, pairs), argnums=0)
    theta_bar = jnp.asarray(theta_bar, dtype=float)
    rows = jnp.asarray(spend_rows, dtype=float)
    grads = jax.vmap(lambda s: g_one(theta_bar, s))(rows)
    return np.asarray(grads, dtype=float)


# ── posterior moment-matching ────────────────────────────────────────────────


def theta_moments(
    post: Posterior, *, ridge: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian ``(mu, Sigma)`` matched to the posterior over ``theta``."""
    if getattr(post, "activation", "hill") != "hill":
        raise NotImplementedError(
            "the fast Laplace-KG / pure-EIG acquisition currently packs the Hill "
            f"parameters (beta, kappa, alpha, gamma); posterior activation is "
            f"{post.activation!r}. Use the planner's decision readouts "
            "(thompson_wave / marginal_roas / expected_regret), which are "
            "activation-agnostic."
        )
    gp = np.stack(
        [post.samples[pair_name(post.channels, p)] for p in post.pairs], axis=1
    )
    m = np.concatenate(
        [post.samples["beta"], post.samples["kappa"], post.samples["alpha"], gp], axis=1
    )
    mu = m.mean(0)
    sigma = np.cov(m, rowvar=False) + ridge * np.eye(m.shape[1])
    return mu, sigma


def _cell_weights(n_cells: int, n_geo: int, t_test: int) -> np.ndarray:
    """Default per-cell row weight: balanced round-robin geos × test weeks."""
    return np.full(n_cells, (n_geo / n_cells) * t_test)


def design_information(
    design_cells: np.ndarray,
    theta_bar: np.ndarray,
    *,
    sigma: float,
    k: int,
    pairs: list[Pair],
    cell_weights: np.ndarray | None = None,
    n_geo: int = 80,
    t_test: int = 10,
    residualize: bool = True,
) -> np.ndarray:
    """Fisher information ``Lambda`` (P×P) a design carries about ``theta``.

    ``Lambda = sigma^-2 sum_c w_c (g_c - g_bar)(g_c - g_bar)^T`` where ``g_c`` is
    the parameter gradient at cell ``c`` and ``g_bar`` the weighted mean across
    cells (profiling the geo intercept). ``residualize=False`` skips the
    centering (treats the baseline as known).
    """
    g = param_grad(theta_bar, design_cells, k, pairs)  # (n_cells, P)
    w = cell_weights
    if w is None:
        w = _cell_weights(design_cells.shape[0], n_geo, t_test)
    w = np.asarray(w, dtype=float) / float(sigma) ** 2
    if residualize:
        g = g - (w[:, None] * g).sum(0) / w.sum()
    return (g * w[:, None]).T @ g


# ── pure EIG (D / D_s optimality) ─────────────────────────────────────────────


def gaussian_eig(
    sigma0: np.ndarray, lam: np.ndarray, *, idx: list[int] | None = None
) -> float:
    """EIG (nats) of a parameter block under a Gaussian-linear update.

    ``0.5 * (logdet Sigma0_SS - logdet Sigma_post_SS)`` for the block ``idx``
    (all parameters if ``None``) — the entropy the design removes.
    """
    sigma_post = np.linalg.inv(np.linalg.inv(sigma0) + lam)
    if idx is None:
        idx = list(range(sigma0.shape[0]))
    sel = np.ix_(idx, idx)
    _, ld0 = np.linalg.slogdet(sigma0[sel])
    _, ldp = np.linalg.slogdet(sigma_post[sel])
    return float(0.5 * (ld0 - ldp))


def design_eig(
    post: Posterior,
    design_cells: np.ndarray,
    *,
    sigma: float,
    target: str = "all",
    n_geo: int = 80,
    t_test: int = 10,
    cell_weights: np.ndarray | None = None,
) -> float:
    """Pure information gain of a design (D-optimal ``target="all"`` or
    D_s-optimal ``target="gamma"`` over the synergy sub-block)."""
    k, pairs = post.n_channels, post.pairs
    mu, sigma0 = theta_moments(post)
    lam = design_information(
        design_cells,
        mu,
        sigma=sigma,
        k=k,
        pairs=pairs,
        cell_weights=cell_weights,
        n_geo=n_geo,
        t_test=t_test,
    )
    idx = None if target == "all" else gamma_indices(k, pairs)
    if target not in ("all", "gamma"):
        raise ValueError(f"target must be 'all' or 'gamma', got {target!r}")
    return gaussian_eig(sigma0, lam, idx=idx)


# ── Laplace knowledge-gradient ────────────────────────────────────────────────


def _clip_valid(theta: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Map a (possibly Gaussian-sampled) theta to valid surface params."""
    beta, kappa, alpha, gp = unpack_theta(theta, k)
    return {
        "beta": np.clip(beta, 0.0, None),
        "kappa": np.clip(kappa, 1e-3, None),
        "alpha": np.clip(alpha, 0.5, 5.0),
        "gp": gp,
    }


def _allocate_theta(theta, post, B, value, *, mode, cap, n_starts, seed):
    p = _clip_valid(np.asarray(theta, float), post.n_channels)
    gamma = np.zeros((post.n_channels, post.n_channels))
    for idx, (i, j) in enumerate(post.pairs):
        gamma[i, j] = p["gp"][idx]
    params = {
        "beta": p["beta"],
        "kappa": p["kappa"],
        "alpha": p["alpha"],
        "gamma": gamma,
    }
    return _planner.allocate_under_sample(
        params, B, value, mode=mode, cap=cap, n_starts=n_starts, seed=seed
    )


def laplace_knowledge_gradient(
    post: Posterior,
    design_cells: np.ndarray,
    B: float,
    value: float,
    *,
    sigma: float,
    n_geo: int = 80,
    t_test: int = 10,
    n_outcomes: int = 64,
    mode: str = "fixed",
    cap: float | None = None,
    n_starts: int = 3,
    cell_weights: np.ndarray | None = None,
    seed: int = 0,
) -> float:
    """Decision-aware EVSI of a candidate design, with **no refit** (guide §9.1).

    Gaussian-linear surrogate for
    :func:`mmm_framework.continuous_learning.planner.knowledge_gradient`: the
    pre-posterior spread of the updated mean is ``V = Sigma - Sigma_post``;
    fantasy means ``theta_m ~ N(mu, V)`` are re-optimized and averaged against the
    current best. Orders designs the same way as the NUTS KG at a fraction of the
    cost.
    """
    k, pairs = post.n_channels, post.pairs
    mu, sigma0 = theta_moments(post)
    lam = design_information(
        design_cells,
        mu,
        sigma=sigma,
        k=k,
        pairs=pairs,
        cell_weights=cell_weights,
        n_geo=n_geo,
        t_test=t_test,
    )
    sigma_post = np.linalg.inv(np.linalg.inv(sigma0) + lam)
    v = 0.5 * ((sigma0 - sigma_post) + (sigma0 - sigma_post).T)
    # project V to PSD (numerical safety before sampling)
    evals, evecs = np.linalg.eigh(v)
    v = (evecs * np.clip(evals, 0.0, None)) @ evecs.T

    _, base_value = _allocate_theta(
        mu, post, B, value, mode=mode, cap=cap, n_starts=n_starts + 1, seed=seed
    )
    rng = np.random.default_rng(seed)
    thetas = rng.multivariate_normal(mu, v, size=n_outcomes)
    vals = [
        _allocate_theta(
            th, post, B, value, mode=mode, cap=cap, n_starts=n_starts, seed=seed + i
        )[1]
        for i, th in enumerate(thetas)
    ]
    return float(np.mean(vals) - base_value)
