"""Layer 1 — the model-free response model (NumPyro).

A lightweight Bayesian response surface fit **directly from designed experiment
data** — no observational time series, no pre-fit MMM. The whole point of the
continuous-learning loop is that the experiment's designed cross-sectional
variation identifies the surface, so the priors inform but the data dominates.

Priors (guide §4.2)::

    beta_c   ~ HalfNormal(beta_scale)                 # non-negative ceilings
    kappa_c  ~ LogNormal(0, 0.5)                       # O(1) half-saturation
    alpha_c  ~ TruncatedNormal(2, 0.5, [0.5, 5])       # Hill shape
    gamma    ~ sign-informed per pair (see PAIR_SIGNS) # synergy
    A        ~ Normal(0, 5);  sigma_a ~ HalfNormal(2)  # geo-intercept hypers
    a_geo    ~ Normal(A, sigma_a)                      # baseline random intercept
    sigma    ~ HalfNormal(1)                           # observation noise

Likelihood::

    y ~ Normal(a_geo[geo_idx] + incremental(spend), sigma)

The geo intercept ``a_geo`` is pinned by the pre-period (where every geo shares
the status-quo allocation), which breaks the within-geo collinearity between
baseline and incremental response — the CUPED-style identification requirement
of guide §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import numpy as np

from .surface import ACTIVATIONS

Pair = tuple[int, int]
PairSign = str  # "neg" | "pos" | "zero" | "weak"

VALID_SIGNS = ("neg", "pos", "zero", "weak")

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
):
    """NumPyro generative model (guide §4.2/4.3), for any pluggable activation.

    ``y=None`` draws from the prior predictive (prior checks). ``pair_signs``
    maps each pair to a prior family; pairs absent from the map default to
    ``"weak"``. Sign-constrained pairs (``neg``/``pos``) record a signed
    deterministic ``gamma_<ci>_<cj>`` so every pair has a uniform posterior site.
    ``activation`` selects the per-channel saturation family (``"hill"`` default,
    ``"logistic"`` for exponential saturation).
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    from .surface import surface_over_rows

    k = len(channels)

    beta = numpyro.sample("beta", dist.HalfNormal(beta_scale).expand([k]).to_event(1))
    shape = _sample_activation_shape(activation, k, numpyro, dist)

    gamma = jnp.zeros((k, k))
    for i, j in pairs:
        sign = pair_signs.get((i, j), "weak")
        nm = pair_name(channels, (i, j))
        if sign == "neg":
            mag = numpyro.sample(f"{nm}__mag", dist.HalfNormal(gamma_scale))
            g = numpyro.deterministic(nm, -mag)
        elif sign == "pos":
            mag = numpyro.sample(f"{nm}__mag", dist.HalfNormal(gamma_scale))
            g = numpyro.deterministic(nm, mag)
        elif sign == "zero":
            g = numpyro.sample(nm, dist.Normal(0.0, 0.05 * gamma_scale))
        else:  # "weak"
            g = numpyro.sample(nm, dist.Normal(0.0, gamma_scale))
        gamma = gamma.at[i, j].set(g)

    a = numpyro.sample("A", dist.Normal(0.0, 5.0))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(2.0))
    with numpyro.plate("geos", n_geo):
        a_geo = numpyro.sample("a_geo", dist.Normal(a, sigma_a))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))

    act_fn = ACTIVATIONS[activation][1]
    mu = a_geo[geo_idx] + surface_over_rows(spend, beta, gamma, act_fn, shape)
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)


# ── Posterior container + fitter ──────────────────────────────────────────────


@dataclass
class Posterior:
    """Posterior draws plus the metadata the planner needs.

    ``samples`` holds plain numpy arrays keyed by site name (``beta``, ``kappa``,
    ``alpha``, ``A``, ``sigma_a``, ``a_geo``, ``sigma``, and one
    ``gamma_<ci>_<cj>`` per pair). ``spend_ref`` is the per-channel reference
    constant used to scale spend (so the planner can map scaled allocations back
    to dollars).
    """

    samples: dict[str, np.ndarray]
    channels: list[str]
    pairs: list[Pair]
    pair_signs: dict[Pair, PairSign] = field(default_factory=dict)
    activation: str = "hill"
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


def fit(
    data: dict[str, Any],
    *,
    channels: list[str],
    pairs: list[Pair] | None = None,
    pair_signs: dict[Pair, PairSign] | None = None,
    activation: str = "hill",
    beta_scale: float = 1.0,
    gamma_scale: float = 0.8,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 2,
    seed: int = 0,
    progress_bar: bool = False,
    spend_ref: np.ndarray | None = None,
) -> Posterior:
    """Fit the response surface to a geo-week panel by NUTS.

    Args:
        data: the data contract — ``{"spend": (N, K), "geo_idx": (N,),
            "n_geo": int, "y": (N,)}`` (scaled spend, natural-unit ``y``).
        channels: channel names, length ``K``.
        pairs: interaction pairs to model (default: all upper-triangular pairs).
        pair_signs: sign-informed prior family per pair (default: all ``"weak"``).
        gamma_scale: interaction prior scale — the most consequential knob; audit
            it with :func:`prior_sensitivity` (guide §8.2).
        spend_ref: per-channel reference constant used to scale spend, carried on
            the returned :class:`Posterior` for dollar mapping.

    Returns:
        A :class:`Posterior` with merged-chain numpy samples and R-hat/ESS
        diagnostics on the key parameters.
    """
    import jax
    from numpyro.infer import MCMC, NUTS

    jax.config.update("jax_platform_name", "cpu")

    k = len(channels)
    pairs = pairs if pairs is not None else default_pairs(k)
    pair_signs = dict(pair_signs or {})
    for sign in pair_signs.values():
        if sign not in VALID_SIGNS:
            raise ValueError(f"pair sign must be one of {VALID_SIGNS}, got {sign!r}")
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {activation!r}; known: {tuple(ACTIVATIONS)}"
        )

    spend = np.asarray(data["spend"], dtype=float)
    geo_idx = np.asarray(data["geo_idx"], dtype=int)
    y = np.asarray(data["y"], dtype=float)
    n_geo = int(data["n_geo"])
    if spend.ndim != 2 or spend.shape[1] != k:
        raise ValueError(f"spend must be (N, {k}) for {k} channels, got {spend.shape}")

    bound = partial(
        model,
        channels=channels,
        pairs=pairs,
        pair_signs=pair_signs,
        beta_scale=beta_scale,
        gamma_scale=gamma_scale,
        activation=activation,
    )
    mcmc = MCMC(
        NUTS(bound),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.PRNGKey(seed), spend=spend, geo_idx=geo_idx, n_geo=n_geo, y=y)

    samples = {key: np.asarray(val) for key, val in mcmc.get_samples().items()}
    # drop the latent magnitudes for sign-constrained pairs (the signed
    # deterministic gamma_<ci>_<cj> is the public site)
    samples = {key: val for key, val in samples.items() if not key.endswith("__mag")}

    diagnostics = _diagnostics(mcmc)
    return Posterior(
        samples=samples,
        channels=list(channels),
        pairs=list(pairs),
        pair_signs=pair_signs,
        activation=activation,
        spend_ref=None if spend_ref is None else np.asarray(spend_ref, dtype=float),
        diagnostics=diagnostics,
    )


def _diagnostics(mcmc: Any) -> dict[str, Any]:
    """Worst-case R-hat / min ESS over the headline parameters."""
    try:
        import numpyro

        grouped = mcmc.get_samples(group_by_chain=True)
        rhats: list[float] = []
        ess: list[float] = []
        for key in ("beta", "kappa", "alpha", "lam", "sigma"):
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
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], Posterior]:
    """Build a ``refit_fn(extra_spend, extra_geo_idx, extra_y) -> Posterior``.

    The knowledge-gradient acquisition (:func:`planner.knowledge_gradient`)
    fantasizes wave outcomes and refits; this closure appends the fantasy rows to
    ``base_data`` and runs a *short* NUTS chain. This is the expensive path — see
    guide §9.1 for the Laplace-update replacement in production.
    """
    spend0 = np.asarray(base_data["spend"], dtype=float)
    geo0 = np.asarray(base_data["geo_idx"], dtype=int)
    y0 = np.asarray(base_data["y"], dtype=float)
    n_geo = int(base_data["n_geo"])

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
        )

    return refit
