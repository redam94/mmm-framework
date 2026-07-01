"""The response surface — one differentiable JAX function, used everywhere.

This module is the *single source of truth* for how scaled spend drives
incremental outcome in the continuous-learning loop. The very same
:func:`incremental` is evaluated inside

* the NumPyro likelihood (:mod:`mmm_framework.continuous_learning.model`),
* the synthetic data-generating process (:mod:`.dgp`), and
* the allocator / acquisition planner (:mod:`.planner`, which differentiates it
  with :func:`jax.grad`).

Because all three call the identical function, the optimizer can never disagree
with the fitted surface and the recovery test exercises the exact map the
planner exploits.

Functional form (per geo-week, for a scaled spend vector ``s`` of length ``K``)::

    f_c(s_c)       = s_c^alpha_c / (kappa_c^alpha_c + s_c^alpha_c)     # Hill in [0, 1)
    incremental(s) = sum_c beta_c f_c(s_c)
                     + sum_{c<c'} gamma_{cc'} f_c(s_c) f_c'(s_c')

The Hill activation matches the framework's ``SaturationType.HILL``
(``x^slope / (x^slope + sat_half^slope)``) with ``slope = alpha`` and
``sat_half = kappa`` — so a continuous-learning posterior is directly comparable
to a :class:`~mmm_framework.model.BayesianMMM` Hill fit on the same channel.

The interaction block uses an **upper-triangular** ``gamma`` matrix (``gamma[i, j]``
for ``i < j``); ``gamma`` carries the sign of the synergy (positive =
complementarity, negative = cannibalization). The cross-partial is
``d^2 incremental / ds_c ds_c' = gamma_{cc'} f_c'(s_c) f_c''(... )`` — see the
guide for the algebra; the planner never needs it by hand because it
auto-differentiates :func:`incremental` directly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Spend floor mirrors the framework's Hill guard (``pt.maximum(x, 1e-9)`` in
# ``model/base.py::_apply_saturation_pt``): d/dx of x^alpha is unbounded at
# x = 0 for alpha < 1, which would hand NUTS — and SLSQP — infinite gradients on
# a zero-spend (shutoff) cell. Clamping to a tiny positive floor makes f(0) ~ 0
# while keeping every gradient finite, and avoids the ``jnp.where`` nan-grad
# pitfall entirely.
_SPEND_FLOOR = 1e-9


def activation(spend, kappa, alpha):
    """Beta-stripped Hill activation ``f_c(s_c) in [0, 1)``, elementwise.

    Args:
        spend: scaled spend, shape ``(K,)`` (non-negative; a shutoff cell is 0).
        kappa: half-saturation per channel, shape ``(K,)`` (positive).
        alpha: Hill shape per channel, shape ``(K,)`` (positive).

    Returns:
        The activation fraction, shape ``(K,)``.
    """
    s = jnp.maximum(spend, _SPEND_FLOOR)
    s_pow = s**alpha
    return s_pow / (s_pow + kappa**alpha)


def incremental(spend, beta, kappa, alpha, gamma):
    """Incremental response to one scaled spend vector.

    Args:
        spend: scaled spend, shape ``(K,)``.
        beta: channel ceilings, shape ``(K,)`` (non-negative).
        kappa: half-saturations, shape ``(K,)``.
        alpha: Hill shapes, shape ``(K,)``.
        gamma: upper-triangular interaction matrix, shape ``(K, K)`` (zeros on
            and below the diagonal).

    Returns:
        Scalar incremental response.
    """
    f = activation(spend, kappa, alpha)
    main = jnp.sum(beta * f)
    interaction = jnp.sum(gamma * jnp.outer(f, f))
    return main + interaction


# Vectorized over a (N, K) panel of spend rows; params are shared across rows.
incremental_batch = jax.vmap(incremental, in_axes=(0, None, None, None, None))

# JIT-compiled scalar response and its gradient w.r.t. spend (shape (K,)). The
# planner calls these thousands of times across SLSQP iterations and posterior
# draws; compiling once for the channel-count shape and reusing keeps the
# allocator fast. This is the exact same surface the likelihood fits, so the
# optimizer can never disagree with the fitted model.
incremental_jit = jax.jit(incremental)
grad_incremental = jax.jit(jax.grad(incremental, argnums=0))


def response_curve(spend_matrix, beta, kappa, alpha, gamma):
    """Incremental response for every row of a ``(N, K)`` scaled-spend panel.

    A thin numpy-friendly wrapper over :data:`incremental_batch` returning a
    plain :class:`jax.Array` of shape ``(N,)``.
    """
    return incremental_batch(
        jnp.asarray(spend_matrix, dtype=float),
        jnp.asarray(beta, dtype=float),
        jnp.asarray(kappa, dtype=float),
        jnp.asarray(alpha, dtype=float),
        jnp.asarray(gamma, dtype=float),
    )


# ── Pluggable activations ─────────────────────────────────────────────────────
# The loop is not Hill-specific. It needs only a smooth, monotonically increasing,
# saturating activation ``f_c`` with ``f_c(0)=0``, values in ``[0, 1)``, and a
# finite gradient everywhere (the allocator follows ``dR/ds``). Hill is the
# default; ``logistic`` is a genuinely different family — concave (no S-shape),
# with a single shape parameter — used to show the machinery generalizes.


def logistic(spend, lam):
    """Exponential-saturation activation ``f(s) = 1 - exp(-lam * s)`` in ``[0, 1)``.

    Smooth, strictly increasing, strictly **concave** (no inflection), ``f(0)=0``.
    One shape parameter per channel; the half-saturation point is ``ln(2)/lam``.
    Matches the framework's ``SaturationType.LOGISTIC``.
    """
    s = jnp.maximum(spend, 0.0)
    return 1.0 - jnp.exp(-jnp.maximum(lam, 1e-6) * s)


def hill_mixture(spend, kappa1, alpha1, kappa2, alpha2, w):
    """A weighted sum of two Hill curves: ``w f(κ1,α1) + (1-w) f(κ2,α2)``.

    Still smooth, monotone and saturating (``f(0)=0``, ``->`` a value in ``[0,1)``),
    but far more flexible than a single Hill — it can express a two-phase /
    shoulder shape (a low-κ, high-α component that switches on early plus a
    high-κ component that keeps rising) that a single Hill or a logistic curve
    cannot represent. Used as a **true** DGP to study model misspecification: fit
    it with a single Hill (mild) or a logistic (severe) and watch what breaks.
    """
    f1 = activation(spend, kappa1, alpha1)
    f2 = activation(spend, kappa2, alpha2)
    return w * f1 + (1.0 - w) * f2


# Registry: name -> (ordered shape-parameter names, jax ``fn(spend, *shape)``).
ACTIVATIONS: dict[str, tuple] = {
    "hill": (("kappa", "alpha"), activation),
    "logistic": (("lam",), logistic),
    "hill_mixture": (
        ("kappa1", "alpha1", "kappa2", "alpha2", "w"),
        hill_mixture,
    ),
}


def surface_value(spend, beta, gamma, act_fn, shape):
    """Incremental response for one spend vector under an **arbitrary** activation.

    ``act_fn(spend, *shape) -> f`` computes the per-channel activations; ``shape``
    is a tuple of ``(K,)`` arrays in the activation's parameter order. The
    interaction block is unchanged, so any activation drops straight in.
    """
    f = act_fn(spend, *shape)
    return jnp.sum(beta * f) + jnp.sum(gamma * jnp.outer(f, f))


def surface_over_rows(spend_matrix, beta, gamma, act_fn, shape):
    """:func:`surface_value` for every row of a ``(N, K)`` panel -> ``(N,)``."""
    beta = jnp.asarray(beta, dtype=float)
    gamma = jnp.asarray(gamma, dtype=float)
    shape = tuple(jnp.asarray(a, dtype=float) for a in shape)
    apply = jax.vmap(lambda s: surface_value(s, beta, gamma, act_fn, shape))
    return apply(jnp.asarray(spend_matrix, dtype=float))
