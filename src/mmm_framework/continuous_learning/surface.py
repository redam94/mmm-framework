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
# with a single shape parameter — used to show the machinery generalizes;
# ``monotone_spline`` is the SHAPE-AGNOSTIC family (a normalized monotone
# I-spline) for when no parametric curve should be assumed at all.


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


# ── Monotone I-spline activation (shape-agnostic) ─────────────────────────────
# A monotone regression spline (Ramsay 1988 I-splines): the activation is a
# NORMALIZED positive combination of monotone basis functions,
#
#     f(s) = sum_j w_j I_j(s / MSPLINE_S_MAX) / sum_j w_j,   w_j > 0,
#
# where each I_j is a cubic I-spline — the suffix sum of the clamped cubic
# B-spline basis on [0, 1] (partition of unity => each I_j rises monotonically
# from 0 to 1). Any positive weight vector therefore yields a smooth (C^2),
# monotone, saturating curve with f(0) = 0 and values in [0, 1) — concave,
# S-shaped, two-phase, plateaued, whatever the data asks for — WITHOUT assuming
# a parametric family. Only the weight RATIOS matter (the sum normalizes out),
# which is why the model's priors on w_j can stay O(1) at any KPI scale.
#
# The knots are FIXED module constants (in scaled-spend units, half of the
# domain knot grid below): spend is O(1) by the data contract (scaled by
# ``spend_ref``), so a [0, MSPLINE_S_MAX] domain covers every CCD cell the loop
# probes. Spend at or beyond MSPLINE_S_MAX is treated as fully saturated
# (f -> 1, zero gradient) — the same statement Hill makes asymptotically.
# Because the knots are static Python floats, the Cox–de Boor recursion below
# unrolls at trace time with CONSTANT denominators (zero-width spans dropped in
# Python), so there is no traced division and no ``jnp.where`` nan-grad hazard.

MSPLINE_S_MAX = 3.0  # scaled-spend domain ceiling (full saturation at/after it)
_MSPLINE_DEGREE = 3  # cubic
# Interior knots in x = s/S_MAX units -> scaled spends {0.25, 0.5, 0.75, 1.0,
# 1.5, 2.0}: dense where saturation action lives (spend is O(1) by the data
# contract), sparse in the tail. Knot density sets the family's APPROXIMATION
# BIAS FLOOR — with too few knots, accumulated waves contract the posterior
# band onto the best *spline approximation* of the truth rather than the truth
# itself, and interval coverage of the true curve decays (measured on the
# two-Hill-mixture world: a 3-knot grid's coverage fell to ~60% by wave 5;
# this 6-knot grid holds it high while the band still shrinks).
_MSPLINE_INTERIOR = (
    1.0 / 12.0,
    1.0 / 6.0,
    1.0 / 4.0,
    1.0 / 3.0,
    1.0 / 2.0,
    2.0 / 3.0,
)
_MSPLINE_KNOTS: tuple[float, ...] = (
    (0.0,) * (_MSPLINE_DEGREE + 1) + _MSPLINE_INTERIOR + (1.0,) * (_MSPLINE_DEGREE + 1)
)
_MSPLINE_NBASIS = len(_MSPLINE_KNOTS) - _MSPLINE_DEGREE - 1  # 6 cubic B-splines
#: Number of monotone I-spline basis functions (= weight parameters w1..wJ).
MSPLINE_J = _MSPLINE_NBASIS - 1  # suffix sums, dropping the constant I_0 == 1
# guard just below 1.0: the half-open order-1 indicators vanish AT x = 1 exactly
_MSPLINE_X_HI = 1.0 - 1e-6


def _bspline_basis(x):
    """Clamped cubic B-spline basis at ``x`` in [0, 1) -> ``x.shape + (nbasis,)``.

    Cox–de Boor with the fixed knot vector above. All knot arithmetic happens
    on Python floats at trace time; zero-width spans contribute nothing (their
    terms are skipped statically), so every emitted JAX op is a finite
    polynomial in ``x``.
    """
    t = _MSPLINE_KNOTS
    x = jnp.asarray(x, dtype=float)
    # order 1 (degree 0): half-open interval indicators
    basis = [
        (
            jnp.where((x >= t[i]) & (x < t[i + 1]), 1.0, 0.0)
            if t[i] < t[i + 1]
            else jnp.zeros_like(x)
        )
        for i in range(len(t) - 1)
    ]
    for d in range(1, _MSPLINE_DEGREE + 1):  # raise the degree
        nxt = []
        for i in range(len(t) - d - 1):
            term = jnp.zeros_like(x)
            den_l = t[i + d] - t[i]
            if den_l > 0.0:
                term = term + (x - t[i]) / den_l * basis[i]
            den_r = t[i + d + 1] - t[i + 1]
            if den_r > 0.0:
                term = term + (t[i + d + 1] - x) / den_r * basis[i + 1]
            nxt.append(term)
        basis = nxt
    return jnp.stack(basis, axis=-1)  # x.shape + (_MSPLINE_NBASIS,)


def monotone_spline_basis(spend):
    """The ``MSPLINE_J`` monotone I-spline basis functions at ``spend``.

    Returns shape ``spend.shape + (MSPLINE_J,)``; column ``j`` is
    ``I_{j+1}(spend / MSPLINE_S_MAX)``, rising monotonically 0 -> 1 (earlier
    columns rise earlier). Useful for plotting and for understanding what the
    fitted weights mean.
    """
    x = jnp.clip(jnp.asarray(spend, dtype=float) / MSPLINE_S_MAX, 0.0, _MSPLINE_X_HI)
    b = _bspline_basis(x)
    # I-splines are the suffix sums of the (order k+1) B-spline basis; drop
    # I_0 = sum of ALL B-splines == 1 (the constant).
    isp = jnp.cumsum(b[..., ::-1], axis=-1)[..., ::-1]
    return isp[..., 1:]


def monotone_spline(spend, w1, w2, w3, w4, w5, w6, w7, w8, w9):
    """Shape-agnostic monotone-spline activation, values in ``[0, 1)``.

    ``f(s) = sum_j w_j I_j(s / MSPLINE_S_MAX) / sum_j w_j`` with positive
    weights ``w1..w9`` (one per I-spline basis function; only the ratios
    matter). Smooth, strictly monotone on ``[0, MSPLINE_S_MAX)``, ``f(0) = 0``,
    fully saturated at/after ``MSPLINE_S_MAX``. Early weights buy an early,
    steep rise; late weights buy a slow, late one — mixtures express two-phase
    and plateau shapes no single Hill or logistic can.
    """
    isp = monotone_spline_basis(spend)
    ws = [
        jnp.maximum(jnp.asarray(w, dtype=float), 1e-9)
        for w in (w1, w2, w3, w4, w5, w6, w7, w8, w9)
    ]
    num = sum(w * isp[..., j] for j, w in enumerate(ws))
    return num / sum(ws)


# Registry: name -> (ordered shape-parameter names, jax ``fn(spend, *shape)``).
ACTIVATIONS: dict[str, tuple] = {
    "hill": (("kappa", "alpha"), activation),
    "logistic": (("lam",), logistic),
    "hill_mixture": (
        ("kappa1", "alpha1", "kappa2", "alpha2", "w"),
        hill_mixture,
    ),
    "monotone_spline": (
        ("w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"),
        monotone_spline,
    ),
}
assert len(ACTIVATIONS["monotone_spline"][0]) == MSPLINE_J  # keep grid + args in sync


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
