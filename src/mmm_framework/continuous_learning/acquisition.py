"""Fast acquisition — Laplace knowledge-gradient and pure-EIG (guide §9.1/9.2).

The reference :func:`mmm_framework.continuous_learning.planner.knowledge_gradient`
refits the model with full NUTS once *per fantasy* — too slow to score many
candidate designs. This module replaces that with a **Gaussian-linear**
treatment that needs no MCMC:

1. Moment-match the current posterior over the surface parameters to a Gaussian
   ``N(mu, Sigma)`` — in an **unconstrained reparameterization** ``eta``
   (:class:`ThetaMap`): positive parameters (``beta``, ``kappa``, ``lam``, …)
   are matched in log space, bounded ones (``alpha``, the mixture weight ``w``)
   through a scaled logit, and sign-constrained synergies through a sign-aware
   log. Skewed posteriors of positive parameters are far closer to Gaussian on
   the log scale, and every fantasy sample maps back to a VALID parameter
   vector by construction (no clipping).
2. Linearize the surface in ``eta`` around ``mu`` (the Laplace step). A
   candidate design's Fisher information is then
   ``Lambda = sum_cells w_c * u_c * g_c g_c^T`` with ``g_c = d incremental / d eta``
   and ``u_c`` the observation family's **unit Fisher information** at cell
   ``c`` (``1/sigma^2`` for the Gaussian; ``(nu+1)/((nu+3) sigma^2)`` for the
   Student-t; ``sigmoid(eta)^2 / (m + m^2/phi)`` with ``m = softplus(eta)``
   for the NegBinomial count family — the GLM weight through the softplus
   link). The posterior covariance after running the design is
   ``Sigma_post = (Sigma^-1 + Lambda)^-1``.

From this one linear-algebra object two acquisitions fall out cheaply:

* **Laplace knowledge-gradient** (decision value, EVSI): the pre-posterior
  spread of the updated mean is ``V = Sigma - Sigma_post``; sample fantasy means
  ``eta_m ~ N(mu, V)``, map them back to valid surface parameters, re-optimize
  the allocation for each, and average the uplift over ``max_a profit(a | mu)``.
  Milliseconds instead of minutes.
* **Pure EIG** (information, D-/D_s-optimality): the entropy reduction of a
  parameter block ``S`` is ``0.5 * (logdet Sigma_SS - logdet Sigma_post_SS)``.
  Use the full block for D-optimality, or the ``gamma`` sub-block (``target=
  "gamma"``) for **D_s-optimality** — synergy-targeted waves the exploit-heavy
  Thompson waves under-probe. (EIG differences are invariant to the fixed
  reparameterization — the transform's log-Jacobian cancels between the prior
  and posterior terms — so the D/D_s orderings match the constrained-space
  treatment.)

Any activation registered in :data:`surface.ACTIVATIONS` with a transform spec
in :data:`SHAPE_TRANSFORMS` is supported (Hill, logistic, hill_mixture,
monotone_spline), as is
any observation family in :data:`model.VALID_LIKELIHOODS`. The geo intercept is
profiled out by centering the cell gradients across cells (the per-geo
baselines are randomized, so the identifying variation is between-cell),
mirroring ``planning.identification``.

Prefer the Laplace KG when the goal is the **decision**; use EIG only to shore up
decision-pivotal interactions (guide §9.2).

Two robustness layers guard the surrogate itself:

* **Numerical PSD safeguards.** ``V = Sigma - Sigma_post`` is PSD in exact
  arithmetic (``Lambda >= 0``) but finite-precision inversion/subtraction can
  leave tiny negative eigenvalues in ``V`` or a log-det sub-block. ``V`` is
  symmetrized and eigenvalue-clipped before fantasy sampling, and every
  log-determinant goes through the same symmetrize-and-clip guard — the
  nearest-PSD projection of Higham (1988; 2002).
* **Surrogate validity** (:func:`surrogate_validity`). The Gaussian
  moment-match is an approximation to the carried NUTS posterior; it degrades
  when the posterior is skewed, heavy-tailed or ridge-shaped (negative-gamma
  cannibalisation ridges are the expected failure mode — Kuss & Rasmussen
  2005 for the intuition). The diagnostic scores the fit of ``N(mu, Sigma)``
  to the draws in the unconstrained space — per-parameter skew/kurtosis flags
  plus a generalized-Pareto tail-shape ``khat`` of the Mahalanobis
  exceedances (the same Zhang & Stephens 2009 estimator and the same 0.7
  alarm bar PSIS uses; Vehtari, Simpson, Gelman, Yao & Gabry 2024). When it
  fires, prefer the NUTS-refit :func:`planner.knowledge_gradient` for that
  wave; :func:`loop.select_next_design` records the report in its meta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import planner as _planner
from .model import Pair, Posterior, pair_name
from .surface import ACTIVATIONS, surface_value

# ── parameter transforms ─────────────────────────────────────────────────────
# Each transform is a bijection between the parameter's support and the real
# line. ``unconstrain`` runs on numpy posterior draws (moment matching);
# ``constrain`` must be JAX-traceable (it sits inside the jax.grad of the
# surface and inside the fantasy re-mapping).

_FRAC_EPS = 1e-6  # boundary guard for logit (draws pinned at a truncation bound)
_POS_FLOOR = 1e-12  # log-transform floor (HalfNormal draws are a.s. positive)


class Transform(NamedTuple):
    """A support <-> R bijection: numpy ``unconstrain``, JAX ``constrain``."""

    label: str
    unconstrain: Callable[[np.ndarray], np.ndarray]
    constrain: Callable[[Any], Any]


_LOG = Transform(
    "log",
    lambda x: np.log(np.maximum(np.asarray(x, dtype=float), _POS_FLOOR)),
    jnp.exp,
)

_IDENTITY = Transform("identity", lambda x: np.asarray(x, dtype=float), lambda z: z)

# Sign-constrained synergies: a "pos" pair lives on (0, inf), a "neg" pair on
# (-inf, 0) — matched in log |gamma| space so fantasies keep the prior's sign.
_POS_LOG = _LOG
_NEG_LOG = Transform(
    "neg-log",
    lambda x: np.log(np.maximum(-np.asarray(x, dtype=float), _POS_FLOOR)),
    lambda z: -jnp.exp(z),
)


def _interval(lo: float, hi: float) -> Transform:
    """Scaled-logit transform for a parameter supported on ``[lo, hi]``."""
    span = float(hi) - float(lo)

    def unconstrain(x: np.ndarray) -> np.ndarray:
        f = np.clip(
            (np.asarray(x, dtype=float) - lo) / span, _FRAC_EPS, 1.0 - _FRAC_EPS
        )
        return np.log(f) - np.log1p(-f)

    def constrain(z):
        return lo + span * jax.nn.sigmoid(z)

    return Transform(f"logit[{lo:g},{hi:g}]", unconstrain, constrain)


#: Per-activation transform spec for the shape parameters, matching the prior
#: supports in :func:`model._sample_activation_shape`. Register a new
#: activation family here (plus in :data:`surface.ACTIVATIONS` and the model's
#: shape sampler) and the whole acquisition layer picks it up.
SHAPE_TRANSFORMS: dict[str, dict[str, Transform]] = {
    "hill": {"kappa": _LOG, "alpha": _interval(0.5, 5.0)},
    "logistic": {"lam": _LOG},
    "hill_mixture": {
        "kappa1": _LOG,
        "alpha1": _interval(0.5, 6.0),
        "kappa2": _LOG,
        "alpha2": _interval(0.5, 5.0),
        "w": _interval(0.0, 1.0),
    },
    # positive normalized I-spline weights (LogNormal prior) -> log space
    "monotone_spline": {nm: _LOG for nm in ACTIVATIONS["monotone_spline"][0]},
}


# ── the packed-parameter map ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ThetaMap:
    """Layout + bijection between posterior sites and the packed ``eta`` vector.

    The flat layout is ``[beta (K), *shape sites (K each), gamma_pairs (P)]``
    — the Hill case reproduces the historical ``[beta, kappa, alpha, gamma]``
    packing order. All moment matching, linearization and fantasy sampling
    happen in the unconstrained ``eta`` space; :meth:`constrain_params` maps
    any ``eta`` back to a VALID planner params dict.
    """

    channels: tuple[str, ...]
    pairs: tuple[Pair, ...]
    activation: str
    site_transforms: tuple[tuple[str, Transform], ...]  # beta + shape sites
    gamma_transforms: tuple[Transform, ...]  # one per pair

    @property
    def k(self) -> int:
        return len(self.channels)

    @property
    def dim(self) -> int:
        return self.k * len(self.site_transforms) + len(self.pairs)

    @property
    def gamma_idx(self) -> list[int]:
        """Positions of the synergy parameters within the packed vector."""
        start = self.k * len(self.site_transforms)
        return list(range(start, start + len(self.pairs)))

    def unconstrain_draws(self, samples: dict[str, np.ndarray]) -> np.ndarray:
        """Posterior draws -> the unconstrained ``(n_draws, dim)`` matrix."""
        cols = [
            tr.unconstrain(np.asarray(samples[nm], dtype=float))
            for nm, tr in self.site_transforms
        ]
        gcols = [
            self.gamma_transforms[idx].unconstrain(
                np.asarray(samples[pair_name(list(self.channels), p)], dtype=float)
            )[:, None]
            for idx, p in enumerate(self.pairs)
        ]
        return np.concatenate(cols + gcols, axis=1)

    def constrain_jax(self, eta):
        """``eta`` -> ``(beta, shape_tuple, gamma_matrix)`` in JAX ops."""
        k = self.k
        blocks = [
            tr.constrain(eta[i * k : (i + 1) * k])
            for i, (_nm, tr) in enumerate(self.site_transforms)
        ]
        beta, shape = blocks[0], tuple(blocks[1:])
        start = k * len(self.site_transforms)
        g = jnp.zeros((k, k))
        for idx, (i, j) in enumerate(self.pairs):
            g = g.at[i, j].set(self.gamma_transforms[idx].constrain(eta[start + idx]))
        return beta, shape, g

    def constrain_params(self, eta: np.ndarray) -> dict[str, Any]:
        """``eta`` -> a planner params dict (``allocate_under_sample`` format).

        The transforms enforce every support constraint, so the result is
        always a valid surface parameterization — no clipping needed.
        """
        beta, shape, g = self.constrain_jax(jnp.asarray(eta, dtype=float))
        return {
            "beta": np.asarray(beta, dtype=float),
            "gamma": np.asarray(g, dtype=float),
            "shape": tuple(np.asarray(s, dtype=float) for s in shape),
            "act_fn": ACTIVATIONS[self.activation][1],
            "activation": self.activation,
        }

    def surface_from_eta(self, eta, spend):
        """Incremental response at ``spend`` under packed ``eta`` (JAX)."""
        beta, shape, g = self.constrain_jax(eta)
        return surface_value(spend, beta, g, ACTIVATIONS[self.activation][1], shape)


def theta_map(post: Posterior) -> ThetaMap:
    """Build the :class:`ThetaMap` for a posterior's activation + pair signs."""
    activation = getattr(post, "activation", "hill")
    if activation not in SHAPE_TRANSFORMS or activation not in ACTIVATIONS:
        raise NotImplementedError(
            f"the fast Laplace-KG / pure-EIG acquisition has no transform spec "
            f"for activation {activation!r}; known: {tuple(SHAPE_TRANSFORMS)}. "
            "Add an entry to acquisition.SHAPE_TRANSFORMS (matching the prior "
            "supports in model._sample_activation_shape), or use the planner's "
            "activation-agnostic decision readouts (thompson_wave / "
            "marginal_roas / expected_regret)."
        )
    signs = getattr(post, "pair_signs", None) or {}
    gamma_transforms = tuple(
        (
            _NEG_LOG
            if signs.get(p) == "neg"
            else _POS_LOG if signs.get(p) == "pos" else _IDENTITY
        )
        for p in post.pairs
    )
    return ThetaMap(
        channels=tuple(post.channels),
        pairs=tuple(post.pairs),
        activation=activation,
        site_transforms=(("beta", _LOG),)
        + tuple(
            (nm, SHAPE_TRANSFORMS[activation][nm]) for nm in ACTIVATIONS[activation][0]
        ),
        gamma_transforms=gamma_transforms,
    )


def _hill_theta_map(k: int, pairs: list[Pair]) -> ThetaMap:
    """Default Hill map for low-level callers that pass only ``(k, pairs)``.

    Uses placeholder channel names and all-"weak" (identity) gamma transforms
    — the layout the historical constrained-Hill packing used. High-level
    entry points (:func:`design_eig`, :func:`laplace_knowledge_gradient`)
    always build the true map from the posterior instead.
    """
    return ThetaMap(
        channels=tuple(f"ch{i}" for i in range(k)),
        pairs=tuple(pairs),
        activation="hill",
        site_transforms=(
            ("beta", _LOG),
            ("kappa", _LOG),
            ("alpha", _interval(0.5, 5.0)),
        ),
        gamma_transforms=tuple(_IDENTITY for _ in pairs),
    )


def eta_grad(eta_bar: np.ndarray, spend_rows: np.ndarray, tmap: ThetaMap) -> np.ndarray:
    """``d incremental / d eta`` at ``eta_bar`` for each spend row -> (n, dim).

    The chain rule through the constraining bijection is automatic — JAX
    differentiates ``surface(constrain(eta))`` directly.
    """
    g_one = jax.grad(tmap.surface_from_eta, argnums=0)
    eta_bar = jnp.asarray(eta_bar, dtype=float)
    rows = jnp.asarray(spend_rows, dtype=float)
    grads = jax.vmap(lambda s: g_one(eta_bar, s))(rows)
    return np.asarray(grads, dtype=float)


# ── posterior moment-matching ────────────────────────────────────────────────


def theta_moments(
    post: Posterior, *, ridge: float = 1e-6, tmap: ThetaMap | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian ``(mu, Sigma)`` matched to the posterior — in ``eta`` space.

    The match happens in the unconstrained reparameterization of
    :func:`theta_map`: positive parameters in log space, bounded ones through a
    scaled logit, sign-constrained synergies through a sign-aware log. Any
    activation with a :data:`SHAPE_TRANSFORMS` entry is supported; an unknown
    family raises ``NotImplementedError`` (the planner's decision readouts stay
    activation-agnostic).
    """
    tmap = tmap if tmap is not None else theta_map(post)
    m = tmap.unconstrain_draws(post.samples)
    mu = m.mean(0)
    sigma = np.cov(m, rowvar=False) + ridge * np.eye(m.shape[1])
    return mu, sigma


# ── surrogate validity (is the Gaussian moment-match good enough?) ───────────


def _gpd_khat(exceedances: np.ndarray) -> float:
    """Generalized-Pareto shape ``k`` of positive exceedances.

    The Zhang & Stephens (2009) profile-posterior estimator — the same one
    PSIS uses for its Pareto-:math:`\\hat k` (Vehtari, Simpson, Gelman, Yao &
    Gabry 2024), including the weak prior regularization toward ``k = 0.5``.
    ``k`` near 0 means an exponential-class tail; large positive ``k`` means
    a heavy (polynomial) tail. Returns ``nan`` for fewer than 5 exceedances.
    """
    y = np.sort(np.asarray(exceedances, dtype=float))
    y = y[y > 0]
    n = y.size
    if n < 5:
        return float("nan")
    prior_bs, prior_k = 3.0, 10.0
    m_est = 30 + int(np.sqrt(n))
    idx = np.arange(1, m_est + 1, dtype=float)
    b = 1.0 - np.sqrt(m_est / (idx - 0.5))
    b = b / (prior_bs * y[int(n / 4.0 + 0.5) - 1]) + 1.0 / y[-1]
    k = np.mean(np.log1p(-b[:, None] * y), axis=1)
    len_scale = n * (np.log(-(b / k)) - k - 1.0)
    weights = 1.0 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    weights = weights / weights.sum()
    b_post = float((b * weights).sum())
    k_post = float(np.mean(np.log1p(-b_post * y)))
    return float((n * k_post + prior_k * 0.5) / (n + prior_k))


def surrogate_validity(
    post: Posterior,
    *,
    tmap: ThetaMap | None = None,
    tail_prob: float = 0.25,
    khat_threshold: float = 0.7,
    skew_threshold: float = 0.5,
    kurt_threshold: float = 1.0,
) -> dict[str, Any]:
    """Diagnose whether ``N(mu, Sigma)`` is a valid stand-in for the posterior.

    The Laplace acquisitions replace the carried NUTS posterior with its
    Gaussian moment-match in the unconstrained ``eta`` space (Eq. 12 of the
    math page). This runs entirely off the existing draws — no densities, no
    refit — and checks the two ways that stand-in fails:

    * **Shape**: per-parameter skewness and excess kurtosis of the
      unconstrained draws (a Gaussian has both ~0; the transforms already
      absorb the *generic* positivity skew, so what remains is real).
    * **Tails / ridges**: the generalized-Pareto shape ``khat`` of the
      Mahalanobis-distance exceedances above their ``1 - tail_prob``
      quantile. Under a genuinely Gaussian posterior the squared distances
      are chi-square (exponential-class tail, ``khat ~ 0``); a skewed,
      heavy-tailed or ridge-shaped posterior (the negative-``gamma``
      cannibalisation ridge is the expected offender) inflates ``khat``.
      The alarm bar is the same **0.7** PSIS uses (Vehtari et al. 2024),
      estimated with the same Zhang–Stephens fit.

    Returns a JSON-safe dict: ``ok`` (False when ``khat`` clears the
    threshold, any parameter is flagged, or there are too few draws to tell),
    ``khat``, ``params_flagged``, ``max_abs_skew``, ``max_excess_kurtosis``,
    and the ``per_param`` detail. When ``ok`` is False, spot-check or replace
    the wave's Laplace scores with the NUTS-refit
    :func:`~mmm_framework.continuous_learning.planner.knowledge_gradient`
    (the correction path in the literature is a skewness-aware expansion —
    Rue, Martino & Chopin 2009).
    """
    tmap = tmap if tmap is not None else theta_map(post)
    m = tmap.unconstrain_draws(post.samples)  # (n_draws, dim)
    n, dim = m.shape
    labels = [
        f"{nm}[{ch}]" for nm, _tr in tmap.site_transforms for ch in tmap.channels
    ] + [pair_name(list(tmap.channels), p) for p in tmap.pairs]

    mu = m.mean(0)
    sd = np.maximum(m.std(0, ddof=1), 1e-12)
    z = (m - mu) / sd
    skew = np.mean(z**3, axis=0)
    ex_kurt = np.mean(z**4, axis=0) - 3.0
    flagged = [
        labels[i]
        for i in range(dim)
        if abs(skew[i]) > skew_threshold or ex_kurt[i] > kurt_threshold
    ]

    cov = np.cov(m, rowvar=False) + 1e-10 * np.eye(dim)
    dm = m - mu
    d2 = np.einsum("ij,ij->i", dm, np.linalg.solve(cov, dm.T).T)
    u = float(np.quantile(d2, 1.0 - tail_prob))
    khat = _gpd_khat(d2[d2 > u] - u)

    ok = bool(np.isfinite(khat)) and khat < khat_threshold and not flagged
    return {
        "ok": ok,
        "khat": float(khat),
        "khat_threshold": float(khat_threshold),
        "params_flagged": flagged,
        "max_abs_skew": float(np.max(np.abs(skew))),
        "max_excess_kurtosis": float(np.max(ex_kurt)),
        "n_draws": int(n),
        "dim": int(dim),
        "per_param": {
            labels[i]: {
                "skew": float(skew[i]),
                "excess_kurtosis": float(ex_kurt[i]),
            }
            for i in range(dim)
        },
    }


# ── observation-family unit information (the GLM weights) ────────────────────


def observation_unit_info(
    post: Posterior,
    design_cells: np.ndarray,
    tmap: ThetaMap,
    mu_eta: np.ndarray,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    """Per-cell Fisher information of ONE observation, for the fitted family.

    * ``"normal"``: ``1 / sigma^2`` (homoskedastic — constant across cells).
    * ``"studentt"``: ``(nu + 1) / ((nu + 3) sigma^2)`` with the posterior-mean
      tail df — the classic heavy-tail efficiency discount (< the Gaussian
      info at any finite ``nu``, recovering it as ``nu -> inf``).
    * ``"negbinomial"``: the GLM weight through the model's softplus link,
      ``sigmoid(eta_c)^2 / (m_c + m_c^2 / phi)`` with
      ``eta_c = baseline + R(s_c; mu)``, ``m_c = softplus(eta_c)`` and the
      posterior-mean ``phi``. The baseline is the posterior-mean ``A`` (or the
      grand mean of ``a_geo`` when the hyper is absent). ``sigma`` is ignored
      — a count family has no Gaussian noise scale.

    ``sigma=None`` reads the posterior-mean ``sigma`` site for the Gaussian
    families; a missing required site raises ``ValueError`` (summaries-only
    fits never sample the observation sites, and no fixed guess is meaningful
    across KPI scales).
    """
    n_cells = int(np.shape(design_cells)[0])
    likelihood = getattr(post, "likelihood", "normal")

    def _mean_site(name: str, why: str) -> float:
        s = post.samples.get(name)
        if s is None:
            raise ValueError(
                f"likelihood={likelihood!r} posterior has no {name!r} site "
                f"({why}) — a summaries-only fit never samples the panel "
                "observation sites, and no fixed guess is meaningful across "
                "KPI scales"
            )
        return float(np.mean(s))

    if likelihood == "normal":
        s = (
            float(sigma)
            if sigma is not None
            else _mean_site("sigma", "the observation-noise scale")
        )
        return np.full(n_cells, 1.0 / s**2)
    if likelihood == "studentt":
        s = (
            float(sigma)
            if sigma is not None
            else _mean_site("sigma", "the observation-noise scale")
        )
        nu = _mean_site("nu", "the Student-t tail df")
        return np.full(n_cells, (nu + 1.0) / ((nu + 3.0) * s**2))
    if likelihood == "negbinomial":
        phi = _mean_site("phi", "the NB concentration")
        if "A" in post.samples:
            baseline = float(np.mean(post.samples["A"]))
        elif "a_geo" in post.samples:
            baseline = float(np.mean(post.samples["a_geo"]))
        else:
            raise ValueError(
                "likelihood='negbinomial' posterior has no 'A'/'a_geo' site — "
                "the softplus-link GLM weight needs the baseline level to "
                "evaluate the count mean at each design cell"
            )
        r_cells = np.array(
            [
                float(tmap.surface_from_eta(jnp.asarray(mu_eta, dtype=float), s_c))
                for s_c in np.asarray(design_cells, dtype=float)
            ]
        )
        eta = baseline + r_cells
        m = np.logaddexp(0.0, eta)  # softplus
        link = 1.0 / (1.0 + np.exp(-eta))  # sigmoid = dm/deta
        return link**2 / (m + m**2 / phi)
    raise NotImplementedError(
        f"observation_unit_info does not know likelihood {likelihood!r}; "
        "known: ('normal', 'studentt', 'negbinomial')"
    )


def _cell_weights(n_cells: int, n_geo: int, t_test: int) -> np.ndarray:
    """Default per-cell row weight: balanced round-robin geos × test weeks."""
    return np.full(n_cells, (n_geo / n_cells) * t_test)


def design_information(
    design_cells: np.ndarray,
    theta_bar: np.ndarray,
    *,
    sigma: float | None = None,
    k: int | None = None,
    pairs: list[Pair] | None = None,
    tmap: ThetaMap | None = None,
    unit_info: np.ndarray | None = None,
    cell_weights: np.ndarray | None = None,
    n_geo: int = 80,
    t_test: int = 10,
    residualize: bool = True,
) -> np.ndarray:
    """Fisher information ``Lambda`` (dim×dim) a design carries about ``eta``.

    ``Lambda = sum_c w_c u_c (g_c - g_bar)(g_c - g_bar)^T`` where ``g_c`` is
    the unconstrained-space parameter gradient at cell ``c``, ``u_c`` the
    observation family's per-cell unit information (``unit_info``; defaults to
    the homoskedastic Gaussian ``1/sigma^2``), and ``g_bar`` the weighted mean
    across cells (profiling the geo intercept). ``residualize=False`` skips
    the centering (treats the baseline as known).

    ``theta_bar`` is the unconstrained ``eta`` vector from
    :func:`theta_moments`. Pass the matching ``tmap``; when omitted, a default
    Hill map with identity gamma transforms is built from ``(k, pairs)`` —
    only correct for a Hill posterior with default ("weak"/"zero") pair signs.
    """
    if tmap is None:
        if k is None or pairs is None:
            raise ValueError("design_information needs either tmap or (k, pairs)")
        tmap = _hill_theta_map(k, pairs)
    g = eta_grad(theta_bar, design_cells, tmap)  # (n_cells, dim)
    n_cells = g.shape[0]
    if unit_info is None:
        if sigma is None:
            raise ValueError("design_information needs either unit_info or sigma")
        unit_info = np.full(n_cells, 1.0 / float(sigma) ** 2)
    w = cell_weights
    if w is None:
        w = _cell_weights(n_cells, n_geo, t_test)
    w = np.asarray(w, dtype=float) * np.asarray(unit_info, dtype=float)
    if residualize:
        g = g - (w[:, None] * g).sum(0) / w.sum()
    return (g * w[:, None]).T @ g


# ── pure EIG (D / D_s optimality) ─────────────────────────────────────────────


def _logdet_psd(a: np.ndarray, *, rel_floor: float = 1e-12) -> float:
    """log-determinant of a nominally-PSD symmetric matrix, with a guard.

    Symmetrizes, then clips the eigenvalues at ``rel_floor * max_eig`` — the
    eigenvalue-clipping nearest-PSD projection of Higham (1988; see also
    Higham 2002) — so finite-precision inversion noise (a tiny negative
    eigenvalue in a near-singular sub-block) degrades to a bounded
    perturbation instead of a NaN/±inf log-det.
    """
    a = 0.5 * (a + a.T)
    evals = np.linalg.eigvalsh(a)
    floor = max(float(evals.max(initial=0.0)) * rel_floor, np.finfo(float).tiny)
    return float(np.sum(np.log(np.clip(evals, floor, None))))


def gaussian_eig(
    sigma0: np.ndarray, lam: np.ndarray, *, idx: list[int] | None = None
) -> float:
    """EIG (nats) of a parameter block under a Gaussian-linear update.

    ``0.5 * (logdet Sigma0_SS - logdet Sigma_post_SS)`` for the block ``idx``
    (all parameters if ``None``) — the entropy the design removes. Both
    log-dets run through the :func:`_logdet_psd` symmetrize-and-clip guard:
    ``Sigma_post`` is PSD in exact arithmetic (``Lambda >= 0``) but the
    finite-precision double inversion can leave a marginal sub-block —
    equivalently the Schur complement ``Lambda_{S|rest}`` — with tiny
    negative eigenvalues.
    """
    sigma_post = np.linalg.inv(np.linalg.inv(sigma0) + lam)
    if idx is None:
        idx = list(range(sigma0.shape[0]))
    sel = np.ix_(idx, idx)
    return float(0.5 * (_logdet_psd(sigma0[sel]) - _logdet_psd(sigma_post[sel])))


def design_eig(
    post: Posterior,
    design_cells: np.ndarray,
    *,
    sigma: float | None = None,
    target: str = "all",
    n_geo: int = 80,
    t_test: int = 10,
    cell_weights: np.ndarray | None = None,
) -> float:
    """Pure information gain of a design (D-optimal ``target="all"`` or
    D_s-optimal ``target="gamma"`` over the synergy sub-block).

    Works for any registered activation and any fitted observation family;
    ``sigma=None`` derives the noise scale (Gaussian families) from the
    posterior. ``sigma`` is ignored for a count posterior — the GLM weights
    come from ``phi`` and the baseline instead.
    """
    if target not in ("all", "gamma"):
        raise ValueError(f"target must be 'all' or 'gamma', got {target!r}")
    tmap = theta_map(post)
    mu, sigma0 = theta_moments(post, tmap=tmap)
    unit_info = observation_unit_info(post, design_cells, tmap, mu, sigma=sigma)
    lam = design_information(
        design_cells,
        mu,
        tmap=tmap,
        unit_info=unit_info,
        cell_weights=cell_weights,
        n_geo=n_geo,
        t_test=t_test,
    )
    idx = None if target == "all" else tmap.gamma_idx
    return gaussian_eig(sigma0, lam, idx=idx)


# ── Laplace knowledge-gradient ────────────────────────────────────────────────


def _allocate_eta(eta, tmap, B, value, *, mode, cap, n_starts, seed, x0=None):
    params = tmap.constrain_params(np.asarray(eta, dtype=float))
    return _planner.allocate_under_sample(
        params, B, value, mode=mode, cap=cap, n_starts=n_starts, seed=seed, x0=x0
    )


def laplace_knowledge_gradient(
    post: Posterior,
    design_cells: np.ndarray,
    B: float,
    value: float,
    *,
    sigma: float | None = None,
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
    fantasy means ``eta_m ~ N(mu, V)`` are mapped back to valid surface
    parameters, re-optimized and averaged against the current best. Orders
    designs the same way as the NUTS KG at a fraction of the cost.

    Works for any registered activation and fitted observation family (the
    Fisher weights come from :func:`observation_unit_info`); ``sigma=None``
    derives the Gaussian-family noise scale from the posterior.

    Each fantasy's re-optimization is warm-started at the base (``mu``)
    allocation, in addition to the usual uniform/random multi-starts. Fantasy
    means are draws local to ``mu`` (``V`` is a *reduction* in spread from the
    prior), so the true optimum is almost always near the base one; without
    the anchor, a non-concave surface (``gamma`` can be negative) evaluated
    with only a handful of random restarts can converge to a different local
    optimum depending on the platform's BLAS/LAPACK build (SLSQP's floating-
    point path through the same starting point is not bitwise-portable),
    occasionally flipping a close two-candidate comparison. Anchoring removes
    that basin-hopping risk at no extra cost (it replaces one of the existing
    random starts, per :func:`~mmm_framework.continuous_learning.planner._starts`).
    """
    tmap = theta_map(post)
    mu, sigma0 = theta_moments(post, tmap=tmap)
    unit_info = observation_unit_info(post, design_cells, tmap, mu, sigma=sigma)
    lam = design_information(
        design_cells,
        mu,
        tmap=tmap,
        unit_info=unit_info,
        cell_weights=cell_weights,
        n_geo=n_geo,
        t_test=t_test,
    )
    sigma_post = np.linalg.inv(np.linalg.inv(sigma0) + lam)
    v = 0.5 * ((sigma0 - sigma_post) + (sigma0 - sigma_post).T)
    # Symmetrize + eigenvalue-clip V before sampling: V = Sigma - Sigma_post
    # is PSD in exact arithmetic (Lambda >= 0) but finite-precision
    # subtraction can leave tiny negative eigenvalues; clipping is the
    # nearest-PSD projection in the Frobenius norm (Higham 1988).
    evals, evecs = np.linalg.eigh(v)
    v = (evecs * np.clip(evals, 0.0, None)) @ evecs.T

    base_alloc, base_value = _allocate_eta(
        mu, tmap, B, value, mode=mode, cap=cap, n_starts=n_starts + 1, seed=seed
    )
    rng = np.random.default_rng(seed)
    etas = rng.multivariate_normal(mu, v, size=n_outcomes)
    vals = [
        _allocate_eta(
            eta,
            tmap,
            B,
            value,
            mode=mode,
            cap=cap,
            n_starts=n_starts,
            seed=seed + i,
            x0=base_alloc,
        )[1]
        for i, eta in enumerate(etas)
    ]
    return float(np.mean(vals) - base_value)
