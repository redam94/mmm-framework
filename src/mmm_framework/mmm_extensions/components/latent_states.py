"""Latent-state dynamics + measurement builders for StructuralNestedMMM.

The state builders realize a mediator's (or latent factor's) time series from
its driver sum via the scan-free decay-matrix trick: the AR(1) recursion
``z_t = rho * z_{t-1} + u_t`` (``z_{-1} = 0``) has the closed form
``z_t = sum_{s<=t} rho^(t-s) * u_s`` -- a lower-triangular Toeplitz matmul.
O(T^2) memory is fine for MMM horizons and avoids ``pytensor.scan``'s slow
gradient graph.

The measurement builders are the scale anchors: each observed family pins the
latent's location/scale through its own geometry (see MediatorMeasurement in
``mmm_extensions/config.py``), which is what lets survey data identify the
media -> mediator path instead of leaving mediation to be inferred from the
outcome residual (the documented NestedMMM failure mode,
technical-docs/nested-recovery-search.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    from ..config import MediatorDynamics


def ar1_decay_matrix(rho: "pt.TensorVariable", n_obs: int) -> "pt.TensorVariable":
    """Lower-triangular Toeplitz matrix ``D[t, s] = rho^(t-s)`` for ``s <= t``.

    ``D @ u`` realizes the AR(1) accumulation of the impulse series ``u``.
    ``rho`` may be an RV or a constant tensor (RANDOM_WALK passes 1.0).
    """
    tau = np.arange(n_obs)
    lag = tau[:, None] - tau[None, :]
    causal = pt.as_tensor_variable(lag >= 0)
    lag_clamped = pt.as_tensor_variable(np.maximum(lag, 0))
    return pt.where(causal, rho**lag_clamped, 0.0)


def build_latent_state(
    name: str,
    drivers: "pt.TensorVariable",
    dynamics: "MediatorDynamics | str",
    n_obs: int,
    *,
    level: "pt.TensorVariable | float" = 0.0,
    rho_prior_alpha: float = 6.0,
    rho_prior_beta: float = 2.0,
    innovation_sigma: float | None = None,
    centered: bool = False,
) -> "pt.TensorVariable":
    """Realize a latent state series from its driver sum.

    ``STATIC``      -> ``level + drivers``
    ``AR1``         -> ``level + D(rho) @ drivers + AR-noise``
    ``RANDOM_WALK`` -> AR1 with rho fixed at 1 (no rho RV)

    The ``level`` intercept sits OUTSIDE the recursion (state deviations follow
    ``z_t - level = rho * (z_{t-1} - level) + u_t``), so the baseline does not
    ramp up from zero over a burn-in window while media impulses correctly
    accumulate through the state (sustained unit driver -> steady-state lift
    ``1/(1-rho)``).

    ``innovation_sigma=None`` builds a deterministic state (no process noise):
    required for unmeasured mediators, where n_obs free innovations with no
    measurement would just absorb outcome residual.

    ``centered`` picks the AR-noise parameterization (dynamic states only; the
    two are the SAME model, different sampling geometry):

    - non-centered (default): unit innovations ``{name}_innovation ~ N(0, 1)``
      scaled by sigma and accumulated through the decay matrix. Right for WEAK
      measurements (sparse/noisy surveys), where the state is mostly prior.
    - centered: the AR noise ``{name}_state_noise`` is sampled directly
      (``pm.AR`` / ``pm.GaussianRandomWalk``, zero-history init ``N(0, sigma)``).
      Right for STRONG measurements (a dense high-n_t tracker pins z_t, and
      the non-centered form then puts a funnel between sigma and every
      innovation).

    Must be called inside a ``pm.Model`` context.
    """
    from ..config import MediatorDynamics

    dyn = MediatorDynamics(str(getattr(dynamics, "value", dynamics)).lower())

    if dyn == MediatorDynamics.STATIC:
        # With innovations this is iid state noise -- the overdispersion slack
        # for measured non-Gaussian STATIC mediators. (Centered vs non-centered
        # is moot for iid noise.)
        impulse = drivers
        if innovation_sigma is not None:
            sigma = pm.HalfNormal(f"{name}_innovation_sigma", sigma=innovation_sigma)
            eps = pm.Normal(f"{name}_innovation", mu=0.0, sigma=1.0, shape=n_obs)
            impulse = impulse + sigma * eps
        return level + impulse

    if dyn == MediatorDynamics.AR1:
        rho = pm.Beta(f"{name}_persistence", alpha=rho_prior_alpha, beta=rho_prior_beta)
        # rho**lag has a singular gradient at rho == 0 for the lag-0 term; Beta
        # sampling lives strictly inside (0, 1) but MAP/ADVI can push to the
        # boundary -- clip like the base media transform does for adstock alpha.
        rho_safe = pt.clip(rho, 1e-6, 1.0 - 1e-6)
        if innovation_sigma is None:
            return level + ar1_decay_matrix(rho_safe, n_obs) @ drivers
        sigma = pm.HalfNormal(f"{name}_innovation_sigma", sigma=innovation_sigma)
        if centered:
            noise = pm.AR(
                f"{name}_state_noise",
                rho=pt.stack([rho_safe]),
                sigma=sigma,
                init_dist=pm.Normal.dist(0.0, sigma),
                shape=n_obs,
            )
            return level + ar1_decay_matrix(rho_safe, n_obs) @ drivers + noise
        eps = pm.Normal(f"{name}_innovation", mu=0.0, sigma=1.0, shape=n_obs)
        return level + ar1_decay_matrix(rho_safe, n_obs) @ (drivers + sigma * eps)

    # RANDOM_WALK: cumulative sum of impulses (rho == 1), no rho RV
    if innovation_sigma is None:
        return level + pt.cumsum(drivers)
    sigma = pm.HalfNormal(f"{name}_innovation_sigma", sigma=innovation_sigma)
    if centered:
        noise = pm.GaussianRandomWalk(
            f"{name}_state_noise",
            sigma=sigma,
            init_dist=pm.Normal.dist(0.0, sigma),
            shape=n_obs,
        )
        return level + pt.cumsum(drivers) + noise
    eps = pm.Normal(f"{name}_innovation", mu=0.0, sigma=1.0, shape=n_obs)
    return level + pt.cumsum(drivers + sigma * eps)


def standardize_in_graph(x: "pt.TensorVariable") -> "pt.TensorVariable":
    """Center + scale a realized series to unit empirical variance in-graph.

    Pins the scale of an unmeasured latent FACTOR so its loadings carry the
    identified units -- without this the AR(1) marginal variance
    ``1/(1-rho^2)`` trades off against the loadings and the sampler can
    collapse them (the latent_factor_mmm lesson).

    ONLY safe for media-independent series (factors built purely from
    innovation RVs): a media-driven latent standardized this way would have
    its constants recomputed under a counterfactual ``set_data`` swap,
    contaminating every contrast -- which is why LATENT *mediators* are not
    standardized (their beta*gamma products are identified through the priors).

    The variance guard sits INSIDE the sqrt: at the model's initial point the
    innovations are all zero, so ``x.std()`` has a NaN gradient at var == 0
    (0.5/sqrt(0) * 0) and MAP/ADVI would silently stall at the start point.
    """
    return (x - x.mean()) / pt.sqrt(x.var() + 1e-8)


# =============================================================================
# Measurement models
# =============================================================================


def _effective_counts(counts: np.ndarray, design_effect: float) -> np.ndarray:
    """Deflate survey counts by the design effect (``n_eff = n / deff``)."""
    if design_effect == 1.0:
        return counts
    return np.round(counts / design_effect)


def build_gaussian_state_measurement(
    name: str,
    latent: "pt.TensorVariable",
    observed_std: np.ndarray,
    mask: np.ndarray,
    noise_sigma: float = 0.3,
) -> None:
    """Masked Normal observation of a latent state on the STANDARDIZED scale.

    ``observed_std`` must already be z-scored over its observed entries (the
    caller owns data prep); the latent is thereby defined on the standardized
    survey scale with loading fixed at 1 -- the survey pins media -> mediator.
    """
    sigma = pm.HalfNormal(f"{name}_obs_sigma", sigma=noise_sigma)
    idx = np.flatnonzero(mask)
    pm.Normal(
        f"{name}_obs",
        mu=latent[idx],
        sigma=sigma,
        observed=np.asarray(observed_std, dtype=float)[idx],
    )


def build_binomial_state_measurement(
    name: str,
    latent: "pt.TensorVariable",
    counts: np.ndarray,
    trials: np.ndarray,
    mask: np.ndarray,
    design_effect: float = 1.0,
) -> "pt.TensorVariable":
    """Binomial survey observation of a latent state through a logit link.

    ``p_t = sigmoid(z_t)``; ``counts_t ~ Binomial(n_t, p_t)`` on observed weeks.
    Weekly-varying ``trials`` gives each week exactly its finite-sample
    precision (variance ``n_t * p * (1-p)``); the logit link replaces the old
    ``pt.clip``-as-probability construction (dead gradients outside [0, 1])
    and pins the latent's location AND scale absolutely.

    Returns the full-length probability tensor ``sigmoid(latent)`` so the
    caller can register it as a Deterministic / feed it downstream.
    """
    idx = np.flatnonzero(mask)
    n_eff = _effective_counts(np.asarray(trials, dtype=float), design_effect)[idx]
    n_eff = np.maximum(n_eff, 1.0)  # deflation rounding must not drop a wave
    k_eff = _effective_counts(np.asarray(counts, dtype=float), design_effect)[idx]
    k_eff = np.minimum(k_eff, n_eff)  # deflation rounding must keep k <= n

    p_full = pm.math.sigmoid(latent)
    # Guard the extreme tails (|z| > ~35 under MAP) without touching the
    # interior geometry.
    p_obs = pt.clip(p_full[idx], 1e-9, 1.0 - 1e-9)
    # np.round (not astype truncation) so fractional inputs behave the same
    # with and without a design effect.
    pm.Binomial(
        f"{name}_obs",
        n=np.round(n_eff).astype("int64"),
        p=p_obs,
        observed=np.round(k_eff).astype("int64"),
    )
    return p_full


def build_ordered_state_measurement(
    name: str,
    latent: "pt.TensorVariable",
    counts: np.ndarray,
    mask: np.ndarray,
    n_categories: int,
    cutpoint_prior_sigma: float = 2.0,
    design_effect: float = 1.0,
) -> None:
    """Cumulative-logit Multinomial observation of Likert category counts.

    ``P(Y <= k) = sigmoid(c_k - z_t)`` with ordered cutpoints ``c``; per-week
    category counts (rows of ``counts``, shape ``(n_obs, K)``) are Multinomial
    with ``n_t`` = the week's row sum -- so weekly-varying Likert sample sizes
    carry their own precision. Location lives in the cutpoints (the mediator
    equation must NOT have a free level -- exactly confounded); scale is
    identified against the implicit unit-logistic response noise.
    """
    K = int(n_categories)
    idx = np.flatnonzero(mask)
    counts_eff = _effective_counts(np.asarray(counts, dtype=float), design_effect)[idx]
    counts_int = np.round(counts_eff).astype("int64")
    # Row sums come from the SAME rounded cells, so the Multinomial row-sum
    # invariant holds by construction even under design-effect deflation.
    n_per_row = counts_int.sum(axis=1)

    # Generatively ordered cutpoints: anchor + cumulative positive gaps.
    # NOT `transform=ordered` -- transforms only affect logp space, so prior
    # -predictive forward draws would be unordered (negative cell
    # probabilities silently laundered by Multinomial), breaking
    # sample_prior_predictive / parameter-learning diagnostics.
    c0 = pm.Normal(
        f"{name}_cutpoint_anchor",
        mu=-float(K - 2) / 2.0,
        sigma=cutpoint_prior_sigma,
    )
    if K > 2:
        gaps = pm.Gamma(f"{name}_cutpoint_gaps", alpha=2.0, beta=2.0, shape=K - 2)
        cutpoints = c0 + pt.concatenate([pt.zeros(1), pt.cumsum(gaps)])
    else:
        cutpoints = c0[None]
    cutpoints = pm.Deterministic(f"{name}_cutpoints", cutpoints)

    z_obs = latent[idx]
    cum = pm.math.sigmoid(cutpoints[None, :] - z_obs[:, None])  # (n_m, K-1)
    p = pt.concatenate(
        [cum[:, :1], cum[:, 1:] - cum[:, :-1], 1.0 - cum[:, -1:]], axis=1
    )
    # Cells can underflow to 0 in the tails; clip + renormalize keeps the
    # simplex valid without moving interior probabilities meaningfully.
    p = pt.clip(p, 1e-9, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)

    pm.Multinomial(f"{name}_obs", n=n_per_row, p=p, observed=counts_int)


__all__ = [
    "ar1_decay_matrix",
    "build_latent_state",
    "standardize_in_graph",
    "build_gaussian_state_measurement",
    "build_binomial_state_measurement",
    "build_ordered_state_measurement",
]
