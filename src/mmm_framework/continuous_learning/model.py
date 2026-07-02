"""Layer 1 — the model-free response model (NumPyro).

A lightweight Bayesian response surface fit **directly from designed experiment
data** — no observational time series, no pre-fit MMM. The whole point of the
continuous-learning loop is that the experiment's designed cross-sectional
variation identifies the surface, so the priors inform but the data dominates.

Priors (guide §4.2; ``y_loc``/``y_scale`` default to ``0``/``1``)::

    beta_c   ~ HalfNormal(beta_scale * y_scale)        # ceilings on the y scale
    kappa_c  ~ LogNormal(0, 0.5)                       # O(1) half-saturation
    alpha_c  ~ TruncatedNormal(2, 0.5, [0.5, 5])       # Hill shape
    gamma    ~ sign-informed per pair (see PAIR_SIGNS) # synergy (× y_scale)
    A        ~ Normal(y_loc, 5 * y_scale)              # geo-intercept hypers
    sigma_a  ~ HalfNormal(2 * y_scale)
    a_geo    ~ Normal(A, sigma_a)                      # baseline random intercept
    sigma    ~ HalfNormal(y_scale)                     # observation noise

The data contract mandates ``y`` in NATURAL units (never normalized), so the
prior scales cannot be hard-coded O(1): :func:`fit` derives ``y_loc``/``y_scale``
from the evidence (``prior_scaling="auto"``, the default — ``y_loc = mean(y)``
and ``y_scale = `` the nearest DECADE of ``std(y)``, or of the ``|lift|/scale``
magnitudes for summaries-only fits) so a revenue-scale KPI (1e5+) gets priors
on its own order of magnitude while O(1) data keeps the original priors
exactly (``y_scale = 1``). ``prior_scaling="unit"`` forces the original O(1)
priors regardless of the data.

Likelihood::

    y ~ Normal(a_geo[geo_idx] + incremental(spend), sigma)

The geo intercept ``a_geo`` is pinned by the pre-period (where every geo shares
the status-quo allocation), which breaks the within-geo collinearity between
baseline and incremental response — the CUPED-style identification requirement
of guide §3.2.

Two **opt-in** extensions (the defaults reproduce the graph above
byte-identically — no new sample sites, same rng stream):

* ``likelihood="negbinomial"`` — count KPIs. The panel observation becomes
  ``y ~ NegativeBinomial2(softplus(mu), phi)`` with concentration
  ``phi ~ LogNormal(log 30, 2)`` replacing the ``sigma`` site; ``y`` must be
  non-negative integer counts (``_validate_panel`` enforces it — note
  :func:`preprocess.cuped_adjust` mutates ``y`` to non-integer/possibly-negative
  values and is therefore incompatible). The **summary** block stays Gaussian
  for BOTH likelihoods: a historical ``lift ± se`` readout is already a
  normal-approximation aggregate, so a count observation model has nothing to
  add there. Link caveat: the planner's marginal readouts differentiate the
  LATENT surface ``R`` while the observable count mean is ``softplus(mu)``
  (derivative ``sigmoid(mu)``) — asymptotically exact for ``mu >> 1``, an
  overstatement of the count-scale marginal near ``mu ≈ 0`` (see :func:`fit`).
* ``time_effect="national"`` — a zero-centered hierarchical national per-period
  shock ``tau_t ~ Normal(0, sigma_tau)``, ``sigma_tau ~ HalfNormal(y_scale)``,
  added to every panel row of period ``t`` (requires ``data["period_idx"]``).
  Zero-centered partial pooling, NOT fixed effects: a free tau level is exactly
  collinear with the intercept hyper ``A``. Summaries need no tau — a national
  per-period constant cancels in the lift *difference* exactly as the geo
  intercept does, so the evidence/summary path is unchanged.

**Summary observations** (past experiments, no panel required). A historical
lift test is a *difference* measurement, so the geo intercept cancels and no
pre-period is needed — the atomic evidence unit is::

    {"spend_test": (K,), "spend_base": (K,), "lift": float, "se": float,
     "scale": float}   # scale = n_units * n_periods the total lift aggregates

with likelihood ``lift ~ Normal(scale * (R(spend_test) - R(spend_base)), se)``.
:func:`fit` accepts ``data["summaries"]`` alongside the panel, or *instead of*
it (``{"summaries": [...], "n_geo": 0}``) — a team with historical readouts and
no panel can still fit the surface. Same structural-stationarity caveat as the
MMM's off-panel calibration: the response curve is assumed stable across the
periods the summaries span. A handful of summaries constrains only a few
contrasts of the surface, so κ/α stay prior-dominated — trust the funded set,
not the curve shape.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import numpy as np

from .surface import ACTIVATIONS

_LOGGER = logging.getLogger(__name__)

# Once-only JAX platform guard. ``jax.config.update("jax_platform_name", ...)``
# is process-global: flipping it inside a library call silently moves a
# GPU-hosting process onto CPU for everything else. We therefore touch it at
# most ONCE per process, and honor ``MMM_CL_JAX_PLATFORM`` ("cpu" default;
# "keep" leaves whatever the host configured untouched).
_PLATFORM_SET = False


def _ensure_cpu() -> None:
    """Point JAX at the configured platform, once per process.

    Reads ``MMM_CL_JAX_PLATFORM``: ``"cpu"`` (default) pins the CL fits to CPU
    (they are small; CPU NUTS avoids GPU contention), any other platform name is
    passed through to JAX, and ``"keep"`` never touches the global config.
    """
    global _PLATFORM_SET
    if _PLATFORM_SET:
        return
    _PLATFORM_SET = True
    choice = os.environ.get("MMM_CL_JAX_PLATFORM", "cpu").strip().lower()
    if choice == "keep":
        return
    import jax

    jax.config.update("jax_platform_name", choice)


Pair = tuple[int, int]
PairSign = str  # "neg" | "pos" | "zero" | "weak"

VALID_SIGNS = ("neg", "pos", "zero", "weak")

#: Observation families for the panel likelihood ("normal" is the default and
#: reproduces the original graph byte-identically).
VALID_LIKELIHOODS = ("normal", "negbinomial")

#: National per-period time-effect modes ("none" default = no tau sites).
VALID_TIME_EFFECTS = ("none", "national")

# A worked PAIR_SIGNS example for the reference channel set
# ["Chatter", "Pulse", "Orbit", "Vibe"]. Real deployments pass their own.
PAIR_SIGNS_EXAMPLE: dict[Pair, PairSign] = {
    (0, 1): "neg",  # Chatter x Pulse -> shared-audience cannibalization
    (0, 2): "weak",  # Chatter x Orbit
    (0, 3): "zero",  # Chatter x Vibe  -> ~0, leave to prior
    (1, 2): "pos",  # Pulse   x Orbit -> demand-gen complementarity
    (1, 3): "weak",  # Pulse   x Vibe
    (2, 3): "pos",  # Orbit   x Vibe  -> demand-gen complementarity
}


def default_pairs(n_channels: int) -> list[Pair]:
    """All upper-triangular channel pairs ``(i, j)`` with ``i < j``."""
    return [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]


def normalize_pair_signs(
    pair_signs: dict[Pair, PairSign] | None,
) -> dict[Pair, PairSign]:
    """Normalize pair-sign keys to ``(min, max)`` orientation and merge.

    The model's pairs are always upper-triangular (:func:`default_pairs`) and
    the prior lookup is ``pair_signs.get((i, j))`` — a reversed key like
    ``(1, 0)`` would otherwise validate but be silently ignored (the user's
    explicit prior replaced by ``"weak"``). Duplicate entries that agree are
    merged; conflicting duplicates (``(0, 1)`` vs ``(1, 0)`` with different
    signs) raise.
    """
    out: dict[Pair, PairSign] = {}
    for pair, sign in (pair_signs or {}).items():
        i, j = int(pair[0]), int(pair[1])
        if i == j:
            raise ValueError(
                f"pair_signs key {tuple(pair)} pairs a channel with itself"
            )
        key = (min(i, j), max(i, j))
        if key in out and out[key] != sign:
            raise ValueError(
                f"pair_signs contains conflicting entries for pair {key}: "
                f"{out[key]!r} vs {sign!r}"
            )
        out[key] = sign
    return out


def pair_name(channels: list[str], pair: Pair) -> str:
    """Posterior site name for a pair's interaction, ``gamma_<ci>_<cj>``."""
    i, j = pair
    return f"gamma_{channels[i]}_{channels[j]}"


def demote_channel(
    channels: list[str],
    name: str,
    *,
    pairs: list[Pair] | None = None,
    base: dict[Pair, PairSign] | None = None,
) -> dict[Pair, PairSign]:
    """Demote a non-randomizable channel (a walled garden) to main-effect-only.

    Every interaction that touches ``name`` becomes ``"zero"`` (prior-dominated);
    its main effect is still identified. Flag ``gamma(., name)`` as the
    least-trustworthy parameter in any decision (guide §5.4).
    """
    if name not in channels:
        raise ValueError(f"{name!r} not in channels {channels}")
    c = channels.index(name)
    pairs = pairs if pairs is not None else default_pairs(len(channels))
    signs = dict(base or {})
    for i, j in pairs:
        if i == c or j == c:
            signs[(i, j)] = "zero"
    return signs


def probe_pairs_excluding(
    channels: list[str], name: str, *, pairs: list[Pair] | None = None
) -> list[Pair]:
    """Drop the off-axis design cells for a demoted channel's pairs."""
    c = channels.index(name)
    pairs = pairs if pairs is not None else default_pairs(len(channels))
    return [(i, j) for (i, j) in pairs if i != c and j != c]


# ── The generative model ──────────────────────────────────────────────────────


def _sample_activation_shape(activation: str, k: int, numpyro, dist) -> tuple:
    """Sample the shape parameters for the chosen activation (in param order).

    Hill: half-saturation ``kappa`` and shape ``alpha``. Logistic: a single
    saturation rate ``lam``. Add a case here (plus an entry in
    :data:`surface.ACTIVATIONS`) to plug in another smooth, saturating family.
    """
    if activation == "hill":
        kappa = numpyro.sample(
            "kappa", dist.LogNormal(0.0, 0.5).expand([k]).to_event(1)
        )
        alpha = numpyro.sample(
            "alpha",
            dist.TruncatedNormal(2.0, 0.5, low=0.5, high=5.0).expand([k]).to_event(1),
        )
        return (kappa, alpha)
    if activation == "logistic":
        # f(s) = 1 - exp(-lam s); lam ~ LogNormal keeps it positive and O(1) in
        # scaled units (half-saturation ln(2)/lam ~ O(1), like the Hill kappa).
        lam = numpyro.sample("lam", dist.LogNormal(0.0, 0.5).expand([k]).to_event(1))
        return (lam,)
    if activation == "hill_mixture":
        # Two ordered Hill components + a mixing weight. Deliberately more
        # flexible — and correspondingly WEAKLY identified from a local design:
        # the priors nudge kappa1 < kappa2 to curb label-switching, but the
        # honest cost of the extra freedom is wider posteriors (that is the
        # point of the misspecification study).
        kappa1 = numpyro.sample(
            "kappa1", dist.LogNormal(-0.4, 0.4).expand([k]).to_event(1)
        )
        kappa2 = numpyro.sample(
            "kappa2", dist.LogNormal(0.4, 0.4).expand([k]).to_event(1)
        )
        alpha1 = numpyro.sample(
            "alpha1",
            dist.TruncatedNormal(2.5, 0.7, low=0.5, high=6.0).expand([k]).to_event(1),
        )
        alpha2 = numpyro.sample(
            "alpha2",
            dist.TruncatedNormal(2.0, 0.5, low=0.5, high=5.0).expand([k]).to_event(1),
        )
        w = numpyro.sample("w", dist.Beta(2.0, 2.0).expand([k]).to_event(1))
        return (kappa1, alpha1, kappa2, alpha2, w)
    raise ValueError(f"unknown activation {activation!r}; known: {tuple(ACTIVATIONS)}")


def model(
    spend,
    geo_idx,
    n_geo,
    y=None,
    *,
    channels: list[str],
    pairs: list[Pair],
    pair_signs: dict[Pair, PairSign],
    beta_scale: float = 1.0,
    gamma_scale: float = 0.8,
    activation: str = "hill",
    likelihood: str = "normal",
    time_effect: str = "none",
    period_idx=None,
    n_period: int = 0,
    y_loc: float = 0.0,
    y_scale: float = 1.0,
    summary_test=None,
    summary_base=None,
    summary_scale=None,
    summary_lift=None,
    summary_se=None,
):
    """NumPyro generative model (guide §4.2/4.3), for any pluggable activation.

    ``y=None`` draws from the prior predictive (prior checks). ``pair_signs``
    maps each pair to a prior family; pairs absent from the map default to
    ``"weak"``. Sign-constrained pairs (``neg``/``pos``) record a signed
    deterministic ``gamma_<ci>_<cj>`` so every pair has a uniform posterior site.
    ``activation`` selects the per-channel saturation family (``"hill"`` default,
    ``"logistic"`` for exponential saturation).

    ``likelihood`` selects the panel observation family: ``"normal"`` (default —
    the original graph, byte-identical) or ``"negbinomial"`` for count KPIs
    (``phi ~ LogNormal(log 30, 2)`` replaces the ``sigma`` site and
    ``y ~ NegativeBinomial2(softplus(mu), phi)``; the identity-link ``mu``
    composition ``a_geo[geo_idx] + surface (+ tau)`` is unchanged, with a
    softplus positivity guard). The summary block below stays **Gaussian for
    both** likelihoods: a lift ± se is a normal-approximation aggregate, and the
    geo intercept AND any national per-period constant cancel in the difference.

    ``time_effect="national"`` (with ``period_idx (N,)`` / ``n_period``) adds a
    zero-centered hierarchical national shock ``tau_t ~ Normal(0, sigma_tau)``
    to every panel row of period ``t``. Zero-centered partial pooling, NOT
    fixed effects — a free tau level is exactly collinear with the intercept
    hyper ``A`` (both are national constants); the hierarchical prior resolves
    it softly and the pre-period still pins ``a_geo``. The tau sites are only
    sampled on the opt-in path, so the default graph is untouched.

    ``y_loc``/``y_scale`` (defaults ``0``/``1`` = the original O(1) priors,
    byte-identical graph) put the intercept/noise/effect priors on the KPI's
    natural scale: ``A ~ N(y_loc, 5·y_scale)``, ``sigma_a ~ HalfNormal(2·y_scale)``,
    ``sigma ~ HalfNormal(y_scale)``, and the ``beta``/``gamma`` scales multiply
    by ``y_scale`` too — the incremental response must reach y-magnitude, so the
    effect ceilings live on the y scale. :func:`fit` derives them from the data.

    The panel likelihood is gated on ``spend.shape[0] > 0`` — a summaries-only
    fit skips the geo-intercept plate and observation-noise sites entirely. The
    optional summary block (``summary_test (M, K)``, ``summary_base (M, K)``,
    ``summary_scale (M,)``, ``summary_lift (M,)``, ``summary_se (M,)``) adds one
    Gaussian observation per historical lift readout::

        lift_m ~ Normal(scale_m * (R(test_m) - R(base_m)), se_m)

    The geo intercept cancels in the difference, so summaries need no
    pre-period. Works with any activation.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from .surface import surface_over_rows

    k = len(channels)
    pair_signs = normalize_pair_signs(pair_signs)
    y_loc = float(y_loc)
    y_scale = float(y_scale)
    beta_prior_scale = beta_scale * y_scale
    gamma_prior_scale = gamma_scale * y_scale

    beta = numpyro.sample(
        "beta", dist.HalfNormal(beta_prior_scale).expand([k]).to_event(1)
    )
    shape = _sample_activation_shape(activation, k, numpyro, dist)

    gamma = jnp.zeros((k, k))
    for i, j in pairs:
        sign = pair_signs.get((i, j), "weak")
        nm = pair_name(channels, (i, j))
        if sign == "neg":
            mag = numpyro.sample(f"{nm}__mag", dist.HalfNormal(gamma_prior_scale))
            g = numpyro.deterministic(nm, -mag)
        elif sign == "pos":
            mag = numpyro.sample(f"{nm}__mag", dist.HalfNormal(gamma_prior_scale))
            g = numpyro.deterministic(nm, mag)
        elif sign == "zero":
            g = numpyro.sample(nm, dist.Normal(0.0, 0.05 * gamma_prior_scale))
        else:  # "weak"
            g = numpyro.sample(nm, dist.Normal(0.0, gamma_prior_scale))
        gamma = gamma.at[i, j].set(g)

    act_fn = ACTIVATIONS[activation][1]

    if spend is not None and int(np.shape(spend)[0]) > 0:
        a = numpyro.sample("A", dist.Normal(y_loc, 5.0 * y_scale))
        sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(2.0 * y_scale))
        with numpyro.plate("geos", n_geo):
            a_geo = numpyro.sample("a_geo", dist.Normal(a, sigma_a))
        if likelihood == "normal":
            sigma = numpyro.sample("sigma", dist.HalfNormal(y_scale))
        elif likelihood == "negbinomial":
            # Dispersion prior: NB2 variance is m + m^2/phi, so near-Poisson
            # behavior at row mean m needs phi >~ m — and m scales with the
            # KPI, so a fixed O(10) prior cannot serve every program. A
            # Gamma(2, 0.1) (mean 20, exponential tail ~e^{-0.1 phi}) puts
            # ~15 prior nats against phi=300 and imposes a data-immune
            # variance floor of m^2/O(10^2) on high-mean, weakly-overdispersed
            # counts (verified: Poisson(1e4) data pinned phi near ~1e3, a 9x
            # variance inflation). LogNormal(log 30, 2) keeps the same O(10)
            # center for genuinely overdispersed KPIs but its heavy log-scale
            # tail (2 sigma spans ~0.5 .. ~1.6e3, and beyond) lets the data
            # buy near-Poisson dispersion when it earns it.
            phi = numpyro.sample("phi", dist.LogNormal(np.log(30.0), 2.0))
        else:
            raise ValueError(
                f"unknown likelihood {likelihood!r}; known: {VALID_LIKELIHOODS}"
            )
        mu = a_geo[geo_idx] + surface_over_rows(spend, beta, gamma, act_fn, shape)
        if time_effect != "none" and period_idx is not None and int(n_period) > 0:
            # Zero-centered hierarchical national shocks (NOT fixed effects —
            # a free tau level is exactly collinear with the intercept hyper A).
            sigma_tau = numpyro.sample("sigma_tau", dist.HalfNormal(y_scale))
            with numpyro.plate("periods", int(n_period)):
                tau = numpyro.sample("tau", dist.Normal(0.0, sigma_tau))
            mu = mu + tau[period_idx]
        if likelihood == "normal":
            numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
        else:
            import jax

            numpyro.sample("y", dist.NegativeBinomial2(jax.nn.softplus(mu), phi), obs=y)

    if summary_lift is not None and int(np.shape(summary_lift)[0]) > 0:
        pred = jnp.asarray(summary_scale, dtype=float) * (
            surface_over_rows(summary_test, beta, gamma, act_fn, shape)
            - surface_over_rows(summary_base, beta, gamma, act_fn, shape)
        )
        numpyro.sample(
            "summary_obs",
            dist.Normal(pred, jnp.asarray(summary_se, dtype=float)),
            obs=jnp.asarray(summary_lift, dtype=float),
        )


# ── Posterior container + fitter ──────────────────────────────────────────────


@dataclass
class Posterior:
    """Posterior draws plus the metadata the planner needs.

    ``samples`` holds plain numpy arrays keyed by site name (``beta``, ``kappa``,
    ``alpha``, ``A``, ``sigma_a``, ``a_geo``, ``sigma``, and one
    ``gamma_<ci>_<cj>`` per pair; the intercept/noise sites are absent for a
    summaries-only fit). A ``likelihood="negbinomial"`` fit carries ``phi``
    (the NB concentration) instead of ``sigma``; a ``time_effect="national"``
    fit adds ``sigma_tau`` and ``tau`` (``(draws, n_period)``). ``spend_ref``
    is the per-channel reference constant used to scale spend — convert with
    :func:`~mmm_framework.continuous_learning.scaling.to_dollars` /
    :func:`~mmm_framework.continuous_learning.scaling.to_scaled`.
    """

    samples: dict[str, np.ndarray]
    channels: list[str]
    pairs: list[Pair]
    pair_signs: dict[Pair, PairSign] = field(default_factory=dict)
    activation: str = "hill"
    likelihood: str = "normal"
    time_effect: str = "none"
    spend_ref: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_draws(self) -> int:
        return int(self.samples["beta"].shape[0])

    def gamma_matrix(self, d: int) -> np.ndarray:
        """Assemble the ``(K, K)`` upper-triangular gamma matrix for draw ``d``."""
        k = self.n_channels
        g = np.zeros((k, k))
        for i, j in self.pairs:
            g[i, j] = float(self.samples[pair_name(self.channels, (i, j))][d])
        return g

    @property
    def shape_names(self) -> tuple[str, ...]:
        """The activation's shape-parameter site names, in order."""
        return ACTIVATIONS[self.activation][0]

    def draw_params(self, d: int) -> dict[str, Any]:
        """Per-draw params for the surface functions, activation-agnostic.

        Returns ``{beta, gamma, shape, act_fn, activation}`` where ``shape`` is a
        tuple of ``(K,)`` arrays in the activation's parameter order and
        ``act_fn`` is its JAX activation. (For the Hill default the shape is
        ``(kappa, alpha)``.)
        """
        names, act_fn = ACTIVATIONS[self.activation]
        return {
            "beta": np.asarray(self.samples["beta"][d], dtype=float),
            "gamma": self.gamma_matrix(d),
            "shape": tuple(
                np.asarray(self.samples[nm][d], dtype=float) for nm in names
            ),
            "act_fn": act_fn,
            "activation": self.activation,
        }

    def gamma_summary(self) -> dict[str, dict[str, float]]:
        """Per-pair posterior mean and 5/95 percentiles of the synergy."""
        out: dict[str, dict[str, float]] = {}
        for i, j in self.pairs:
            nm = pair_name(self.channels, (i, j))
            s = self.samples[nm]
            out[nm] = {
                "mean": float(np.mean(s)),
                "p5": float(np.percentile(s, 5)),
                "p95": float(np.percentile(s, 95)),
                "sign": self.pair_signs.get((i, j), "weak"),
            }
        return out


def _validate_summaries(summaries: Any, k: int) -> list[dict[str, Any]]:
    """Validate + normalize summary-observation dicts (see the module header).

    Each summary must carry ``spend_test``/``spend_base`` of shape ``(K,)``
    (finite, non-negative scaled spend), a finite ``lift``, a positive finite
    ``se``, and a positive ``scale`` (default ``1.0`` = one geo-period). Returns
    copies with the numeric fields coerced to numpy/float; extra keys (e.g.
    provenance from :func:`~mmm_framework.continuous_learning.evidence.experiments_to_summaries`)
    are preserved untouched.
    """
    if summaries is None:
        return []
    out: list[dict[str, Any]] = []
    for m, s in enumerate(summaries):
        if not isinstance(s, dict):
            raise ValueError(f"summary {m} must be a dict, got {type(s).__name__}")
        missing = [
            key for key in ("spend_test", "spend_base", "lift", "se") if key not in s
        ]
        if missing:
            raise ValueError(f"summary {m} is missing keys {missing}")
        st = np.asarray(s["spend_test"], dtype=float)
        sb = np.asarray(s["spend_base"], dtype=float)
        lift = float(s["lift"])
        se = float(s["se"])
        scale = float(s.get("scale", 1.0))
        if st.shape != (k,) or sb.shape != (k,):
            raise ValueError(
                f"summary {m}: spend_test/spend_base must have shape ({k},), "
                f"got {st.shape} / {sb.shape}"
            )
        if not (np.all(np.isfinite(st)) and np.all(np.isfinite(sb))):
            raise ValueError(f"summary {m}: spend vectors contain non-finite values")
        if np.any(st < 0) or np.any(sb < 0):
            raise ValueError(f"summary {m}: scaled spend must be non-negative")
        if not np.isfinite(lift):
            raise ValueError(f"summary {m}: lift must be finite, got {lift}")
        if not np.isfinite(se) or se <= 0:
            raise ValueError(f"summary {m}: se must be positive and finite, got {se}")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"summary {m}: scale must be positive and finite, got {scale}"
            )
        norm = dict(s)
        norm.update(
            {"spend_test": st, "spend_base": sb, "lift": lift, "se": se, "scale": scale}
        )
        out.append(norm)
    return out


def _validate_panel(
    data: dict[str, Any], k: int, *, likelihood: str = "normal"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray | None, int]:
    """Validate the panel part of the data contract (review fix F1).

    JAX *clamps* out-of-bounds indices silently, so a wrong ``n_geo`` or a
    1-based ``geo_idx`` would corrupt the fit with no signal — this check makes
    it loud. Returns ``(spend, geo_idx, y, n_geo, period_idx, n_period)`` with
    an empty ``(0, K)`` panel when the data dict carries no panel keys
    (summaries-only mode); ``period_idx`` is ``None`` (and ``n_period`` 0) when
    the data has no ``"period_idx"`` key. An optional ``period_idx (N,)`` gets
    the same integer/range treatment as ``geo_idx`` (same silent-clamp hazard),
    with ``n_period`` derived as ``max + 1``.

    When ``likelihood="negbinomial"``, ``y`` must be non-negative and
    integer-valued (within 1e-6, then rounded) — count data on its natural
    scale.
    """
    if "spend" not in data:
        return (
            np.zeros((0, k), dtype=float),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=float),
            int(data.get("n_geo", 0)),
            None,
            0,
        )
    spend = np.asarray(data["spend"], dtype=float)
    geo_raw = np.asarray(data["geo_idx"])
    y = np.asarray(data["y"], dtype=float)
    n_geo = int(data["n_geo"])
    if spend.ndim != 2 or spend.shape[1] != k:
        raise ValueError(f"spend must be (N, {k}) for {k} channels, got {spend.shape}")
    n_rows = spend.shape[0]
    if geo_raw.shape != (n_rows,) or y.shape != (n_rows,):
        raise ValueError(
            f"row counts disagree: spend has {n_rows} rows, "
            f"geo_idx has shape {geo_raw.shape}, y has shape {y.shape}"
        )
    if not np.all(np.isfinite(spend)):
        raise ValueError("spend contains non-finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")
    if n_rows > 0:
        if not np.issubdtype(geo_raw.dtype, np.integer):
            raise ValueError(
                f"geo_idx must be integer-typed, got dtype {geo_raw.dtype}"
            )
        if n_geo < 1:
            raise ValueError(f"n_geo must be >= 1 for a non-empty panel, got {n_geo}")
        lo, hi = int(geo_raw.min()), int(geo_raw.max())
        if lo < 0 or hi >= n_geo:
            raise ValueError(
                f"geo_idx must lie in [0, {n_geo}); got range [{lo}, {hi}] — "
                "JAX clamps out-of-bounds indices silently, which would corrupt "
                "the fit without any signal"
            )
    geo_idx = np.ascontiguousarray(geo_raw, dtype=int)
    if likelihood == "negbinomial" and n_rows > 0:
        if np.any(y < 0) or np.max(np.abs(y - np.round(y))) > 1e-6:
            raise ValueError(
                "likelihood='negbinomial' requires y to be non-negative "
                "integer counts on their natural scale; got negative and/or "
                "non-integer values. Note preprocess.cuped_adjust mutates y to "
                "non-integer (possibly negative) values and is INCOMPATIBLE "
                "with a count likelihood — fit the raw counts instead (a "
                "national time effect, time_effect='national', covers much of "
                "CUPED's job for counts)"
            )
        y = np.round(y)
    period_raw = data.get("period_idx")
    period_idx: np.ndarray | None = None
    n_period = 0
    if period_raw is not None:
        period_arr = np.asarray(period_raw)
        if period_arr.shape != (n_rows,):
            raise ValueError(
                f"period_idx must have shape ({n_rows},) to match the panel "
                f"rows, got {period_arr.shape}"
            )
        if n_rows > 0:
            if not np.issubdtype(period_arr.dtype, np.integer):
                raise ValueError(
                    f"period_idx must be integer-typed, got dtype "
                    f"{period_arr.dtype}"
                )
            lo_p = int(period_arr.min())
            if lo_p < 0:
                raise ValueError(
                    f"period_idx must be non-negative (0-based), got minimum "
                    f"{lo_p} — JAX clamps out-of-bounds indices silently, "
                    "which would corrupt the fit without any signal"
                )
            n_period = int(period_arr.max()) + 1
        period_idx = np.ascontiguousarray(period_arr, dtype=int)
    return (
        np.ascontiguousarray(spend),
        geo_idx,
        np.ascontiguousarray(y),
        n_geo,
        period_idx,
        n_period,
    )


def _decade(x: float) -> float:
    """The nearest power of ten to ``x`` (in log10), floored at 1e-6.

    The priors here are ORDER-OF-MAGNITUDE regularizers, calibrated to O(1)
    data: multiplying them by a raw ``std(y)`` would jitter a well-tuned
    geometry by small factors (empirically, a 1.22x widening on the synthetic
    worlds pushed a marginally-identified ``beta``/``gamma`` direction into
    non-identification — R-hat 1.04 -> 1.32, min ESS 60 -> 3). Quantizing the
    anchor to the nearest decade makes ``"auto"`` an exact no-op for data
    already on the O(1) scale the priors were calibrated for, while still
    rescaling by exactly ``10**k`` for real-scale KPIs (the point of the fix).
    """
    return float(10.0 ** np.round(np.log10(max(float(x), 1e-6))))


def _resolve_prior_scaling(
    prior_scaling: str,
    y: np.ndarray,
    summaries: list[dict[str, Any]],
) -> tuple[float, float]:
    """Derive the ``(y_loc, y_scale)`` prior anchors from the evidence (fix [29]).

    ``"unit"`` returns ``(0.0, 1.0)`` — the original hard-coded O(1) priors,
    reproducing the old graph exactly. ``"auto"`` (the default) reads the
    natural-unit KPI scale off the data: panel fits use ``y_loc = mean(y)``
    and ``y_scale = `` the nearest DECADE of ``std(y)`` (see :func:`_decade`;
    O(1) data keeps ``y_scale = 1`` exactly); summaries-only fits have no
    ``y``, so the scale comes from the per-geo-period lift magnitudes
    ``|lift|/scale`` (the decade of the larger of their mean and spread) with
    ``y_loc = 0`` (the geo intercept cancels in a lift, so there is no level
    to anchor).
    """
    if prior_scaling == "unit":
        return 0.0, 1.0
    if prior_scaling != "auto":
        raise ValueError(
            f"prior_scaling must be 'auto' or 'unit', got {prior_scaling!r}"
        )
    if y is not None and np.size(y) > 0:
        return float(np.mean(y)), _decade(float(np.std(y)))
    mags = np.array(
        [abs(float(s["lift"])) / float(s["scale"]) for s in summaries], dtype=float
    )
    if mags.size == 0:  # unreachable: fit() requires some evidence first
        return 0.0, 1.0
    return 0.0, _decade(float(max(np.mean(mags), np.std(mags))))


def fit(
    data: dict[str, Any],
    *,
    channels: list[str],
    pairs: list[Pair] | None = None,
    pair_signs: dict[Pair, PairSign] | None = None,
    activation: str = "hill",
    likelihood: str = "normal",
    time_effect: str = "none",
    beta_scale: float = 1.0,
    gamma_scale: float = 0.8,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    seed: int = 0,
    progress_bar: bool = False,
    spend_ref: np.ndarray | None = None,
    prior_scaling: str = "auto",
) -> Posterior:
    """Fit the response surface to a geo-week panel and/or summary readouts.

    Args:
        data: the data contract — ``{"spend": (N, K), "geo_idx": (N,),
            "n_geo": int, "y": (N,)}`` (scaled spend, natural-unit ``y``),
            optionally with ``"summaries": list[dict]`` (see the module header),
            an optional ``"period_idx": (N,)`` (0-based int; required when
            ``time_effect="national"``), or summaries-only
            (``{"summaries": [...], "n_geo": 0}`` / an empty ``(0, K)``
            panel). At least one of panel rows / summaries required.
        channels: channel names, length ``K``.
        pairs: interaction pairs to model (default: all upper-triangular pairs).
        pair_signs: sign-informed prior family per pair (default: all ``"weak"``).
            Keys are normalized to ``(min, max)`` orientation; conflicting
            duplicate entries raise.
        likelihood: panel observation family — ``"normal"`` (default,
            byte-identical to the original graph) or ``"negbinomial"`` for
            count KPIs (``y`` must be non-negative integer counts; the summary
            block stays Gaussian either way). Link caveat for the planner:
            with ``"negbinomial"`` the marginal readouts (``marginal_roas``,
            ``expected_regret``, free-mode allocation) differentiate the
            LATENT surface ``R``, while the observable count mean is
            ``softplus(mu)`` whose derivative is ``sigmoid(mu)`` — so the
            count-scale marginal response is ``sigmoid(mu) * dR/ds``. Since
            ``sigmoid(mu) ≈ 1`` whenever ``mu >> 1`` (typical counts), the
            readouts are asymptotically exact; near ``mu ≈ 0`` (per-row means
            of a few counts) they OVERSTATE the count-scale marginal response
            by up to ``1/sigmoid(mu)``. :func:`fit` warns loudly when the
            observed count mean is below ~20. Fixed-budget Thompson argmax is
            unaffected (softplus is monotone).
        time_effect: ``"none"`` (default) or ``"national"`` — a zero-centered
            hierarchical per-period national shock ``tau_t`` added to the panel
            mean (a non-empty panel requires ``data["period_idx"]``; see
            :func:`model`). A summaries-only fit needs NO period identity:
            there are no panel rows for ``tau_t`` to index and a national
            per-period constant cancels in the lift difference, so the
            summary path is unchanged (no tau sites are sampled).
        gamma_scale: interaction prior scale — the most consequential knob; audit
            it with :func:`prior_sensitivity` (guide §8.2).
        spend_ref: per-channel reference constant used to scale spend, carried on
            the returned :class:`Posterior` for dollar mapping (see
            :func:`~mmm_framework.continuous_learning.scaling.to_dollars`).
        prior_scaling: ``"auto"`` (default) derives the intercept/noise/effect
            prior scales from the evidence so natural-unit KPIs at any
            magnitude fit sanely; ``"unit"`` reproduces the original O(1)
            priors exactly (see :func:`_resolve_prior_scaling`).

    Returns:
        A :class:`Posterior` with merged-chain numpy samples, R-hat/ESS
        diagnostics on the key parameters, and
        ``diagnostics["evidence"] = {"n_rows", "n_summaries"}``.
    """
    import jax
    from numpyro.infer import MCMC, NUTS

    _ensure_cpu()

    k = len(channels)
    pairs = pairs if pairs is not None else default_pairs(k)
    pair_signs = normalize_pair_signs(pair_signs)
    for sign in pair_signs.values():
        if sign not in VALID_SIGNS:
            raise ValueError(f"pair sign must be one of {VALID_SIGNS}, got {sign!r}")
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {activation!r}; known: {tuple(ACTIVATIONS)}"
        )
    if likelihood not in VALID_LIKELIHOODS:
        raise ValueError(
            f"unknown likelihood {likelihood!r}; known: {VALID_LIKELIHOODS}"
        )
    if time_effect not in VALID_TIME_EFFECTS:
        raise ValueError(
            f"time_effect must be one of {VALID_TIME_EFFECTS}, got {time_effect!r}"
        )

    spend, geo_idx, y, n_geo, period_idx, n_period = _validate_panel(
        data, k, likelihood=likelihood
    )
    n_rows = int(spend.shape[0])
    # Only a NON-empty panel needs period identity: a summaries-only fit has
    # no rows for tau_t to index, and the national per-period constant cancels
    # in the lift difference (the module's documented summary semantics), so
    # tau is simply a no-op there.
    if time_effect == "national" and n_rows > 0 and period_idx is None:
        raise ValueError(
            "time_effect='national' requires data['period_idx'] (a 0-based "
            "(N,) integer period index per panel row) — without it the "
            "national tau_t shocks have nothing to index"
        )
    summaries = _validate_summaries(data.get("summaries"), k)
    if n_rows == 0 and not summaries:
        raise ValueError(
            "fit needs evidence: provide panel rows (spend/geo_idx/y) and/or "
            "data['summaries']"
        )
    if likelihood == "negbinomial" and n_rows > 0 and float(np.mean(y)) < 20.0:
        # Softplus-link caveat: the planner's marginal readouts differentiate
        # the latent surface R, but the observable count mean is softplus(mu)
        # with derivative sigmoid(mu) — near-1 for mu >> 1, but well below 1
        # at low counts, where the readouts overstate the count-scale marginal
        # response. mean(y) is a cheap proxy for the posterior-mean mu level.
        warnings.warn(
            f"likelihood='negbinomial' with a low observed count mean "
            f"(mean(y) = {float(np.mean(y)):.2f} < 20): the planner's "
            "marginal readouts (marginal_roas / expected_regret / free-mode "
            "allocation) differentiate the LATENT surface, while the "
            "count-scale marginal response is sigmoid(mu) * dR/ds — at this "
            "count level sigmoid(mu) can be well below 1, so those readouts "
            "may OVERSTATE the count-scale marginal response (the funding "
            "line is optimistic). Fixed-budget Thompson allocation is "
            "unaffected (softplus is monotone).",
            stacklevel=2,
        )
    y_loc, y_scale = _resolve_prior_scaling(
        prior_scaling, y if n_rows > 0 else None, summaries
    )
    if summaries:
        summary_kwargs = {
            "summary_test": np.stack([s["spend_test"] for s in summaries]),
            "summary_base": np.stack([s["spend_base"] for s in summaries]),
            "summary_lift": np.array([s["lift"] for s in summaries], dtype=float),
            "summary_se": np.array([s["se"] for s in summaries], dtype=float),
            "summary_scale": np.array([s["scale"] for s in summaries], dtype=float),
        }
    else:
        summary_kwargs = {}

    bound = partial(
        model,
        channels=channels,
        pairs=pairs,
        pair_signs=pair_signs,
        beta_scale=beta_scale,
        gamma_scale=gamma_scale,
        activation=activation,
        likelihood=likelihood,
        time_effect=time_effect,
        y_loc=y_loc,
        y_scale=y_scale,
    )
    mcmc = MCMC(
        NUTS(bound),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        spend=spend,
        geo_idx=geo_idx,
        n_geo=n_geo,
        y=y,
        period_idx=period_idx,
        n_period=n_period,
        **summary_kwargs,
    )

    samples = {key: np.asarray(val) for key, val in mcmc.get_samples().items()}
    # drop the latent magnitudes for sign-constrained pairs (the signed
    # deterministic gamma_<ci>_<cj> is the public site)
    samples = {key: val for key, val in samples.items() if not key.endswith("__mag")}

    diagnostics = _diagnostics(mcmc, activation)
    diagnostics["evidence"] = {"n_rows": n_rows, "n_summaries": len(summaries)}
    diagnostics["prior_scaling"] = {
        "mode": prior_scaling,
        "y_loc": float(y_loc),
        "y_scale": float(y_scale),
    }
    return Posterior(
        samples=samples,
        channels=list(channels),
        pairs=list(pairs),
        pair_signs=pair_signs,
        activation=activation,
        likelihood=likelihood,
        time_effect=time_effect,
        spend_ref=None if spend_ref is None else np.asarray(spend_ref, dtype=float),
        diagnostics=diagnostics,
    )


def _diagnostics(mcmc: Any, activation: str = "hill") -> dict[str, Any]:
    """Worst-case R-hat / min ESS over the headline parameters.

    The site list is derived from the activation family's shape-parameter
    names (:data:`surface.ACTIVATIONS`) plus the always-present ``beta``/
    ``sigma`` sites, so R̂ covers whatever family was fit — a hard-coded list
    would leave e.g. ``hill_mixture``'s five shape sites (the very place a
    misspecified fit shows R̂≈1.5) outside the convergence gate. ``phi`` (the
    NegBinomial concentration) and ``tau`` (the national time effect) are
    included for the same reason; absent sites are skipped, so Normal /
    no-time-effect fits are unaffected.
    """
    try:
        import numpyro

        shape_sites = ACTIVATIONS.get(activation, (("kappa", "alpha", "lam"), None))[0]
        grouped = mcmc.get_samples(group_by_chain=True)
        rhats: list[float] = []
        ess: list[float] = []
        for key in ("beta", "sigma", "phi", "tau", *shape_sites):
            if key not in grouped:
                continue
            arr = grouped[key]
            rhats.append(float(np.nanmax(numpyro.diagnostics.gelman_rubin(arr))))
            ess.append(float(np.nanmin(numpyro.diagnostics.effective_sample_size(arr))))
        return {
            "max_rhat": max(rhats) if rhats else None,
            "min_ess": min(ess) if ess else None,
        }
    except Exception:  # diagnostics are advisory; never fail a fit over them
        _LOGGER.debug(
            "continuous-learning fit diagnostics failed; returning None values",
            exc_info=True,
        )
        return {"max_rhat": None, "min_ess": None}


def refit_fn_from_data(
    base_data: dict[str, Any],
    *,
    channels: list[str],
    pairs: list[Pair] | None = None,
    pair_signs: dict[Pair, PairSign] | None = None,
    activation: str = "hill",
    beta_scale: float = 1.0,
    gamma_scale: float = 0.8,
    num_warmup: int = 200,
    num_samples: int = 200,
    num_chains: int = 1,
    seed: int = 7,
    prior_scaling: str = "auto",
    time_effect: str = "none",
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], Posterior]:
    """Build a ``refit_fn(extra_spend, extra_geo_idx, extra_y) -> Posterior``.

    The knowledge-gradient acquisition (:func:`planner.knowledge_gradient`)
    fantasizes wave outcomes and refits; this closure appends the fantasy rows to
    ``base_data`` and runs a *short* NUTS chain. This is the expensive path — see
    guide §9.1 for the Laplace-update replacement in production.

    Requires a panel base dataset: fantasy observations are geo-week rows, so a
    summaries-only data dict has nothing to append them to.

    ``time_effect`` other than ``"none"`` raises ``NotImplementedError``: the
    knowledge-gradient fantasy rows carry no period identity, so a national
    ``tau_t`` cannot be refit on the augmented panel. A ``period_idx`` present
    in ``base_data`` is DROPPED from the refit base for the same reason (the
    refit models no time effect, so the index is inert either way).
    """
    if time_effect != "none":
        raise NotImplementedError(
            f"refit_fn_from_data does not support time_effect={time_effect!r}: "
            "the knowledge-gradient fantasy rows have no period identity, so a "
            "national tau_t cannot be refit on the augmented panel — score "
            "designs with the Laplace acquisition instead, or build the refit "
            "closure with time_effect='none'"
        )
    spend0, geo0, y0, n_geo, _period0, _n_period0 = _validate_panel(
        base_data, len(channels)
    )
    if spend0.shape[0] == 0:
        raise ValueError(
            "refit_fn_from_data requires a panel base dataset "
            "({'spend', 'geo_idx', 'y', 'n_geo'}); a summaries-only state has "
            "no panel rows to append the knowledge-gradient fantasy "
            "observations to"
        )

    def refit(
        extra_spend: np.ndarray, extra_geo_idx: np.ndarray, extra_y: np.ndarray
    ) -> Posterior:
        data = {
            "spend": np.vstack([spend0, np.asarray(extra_spend, dtype=float)]),
            "geo_idx": np.concatenate([geo0, np.asarray(extra_geo_idx, dtype=int)]),
            "y": np.concatenate([y0, np.asarray(extra_y, dtype=float)]),
            "n_geo": n_geo,
        }
        return fit(
            data,
            channels=channels,
            pairs=pairs,
            pair_signs=pair_signs,
            activation=activation,
            beta_scale=beta_scale,
            gamma_scale=gamma_scale,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            seed=seed,
            prior_scaling=prior_scaling,
        )

    return refit
