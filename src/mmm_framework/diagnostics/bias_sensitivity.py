"""Sensitivity of a conclusion to unmeasured confounding — the bias-parameter engine.

Every causal number in this framework rests on an assumption no dataset can check:
that no unmeasured common cause drives both the treatment and the outcome. In an
MMM that assumption is *unobserved demand* (budgets rise when demand is expected
to rise); in a matched-market geo readout it is *parallel trends*. A good fit,
tight intervals and passing posterior-predictive checks are all perfectly
compatible with the assumption being false.

The honest response is not to test the assumption — you cannot — but to price it.
Decompose the observed effect into a causal part and a bias part, ``observed =
tau + beta``, put a **prior** on the bias, and report which conclusions survive
which priors. That is the device in the PyMC Labs notebook `Sensitivity analysis
for unmeasured confounding <https://www.pymc.io/projects/examples/en/latest/
causal_inference/sensitivity_unmeasured_confounding.html>`_. It answers the
decision-scale question — *how much hidden bias would it take to change what I
do?* — rather than the parameter-scale one.

What comes out of it
--------------------
* **A tipping point.** The bias commitment at which ``P(tau > reference)`` falls
  through the decision threshold: "the estimate would have to be overstated by
  more than 22% of its own size for the recommendation to change".
* **A sensitivity surface.** The same probability over a grid of ``(mu, sigma)``
  commitments — a partition of the analyst positions that do and do not support
  the conclusion, rather than a single verdict.
* **A verdict**, in a vocabulary chosen not to overclaim (see :data:`VERDICTS`).

Two scales, and why there is no third
-------------------------------------
``scale="absolute"`` puts the bias in the estimand's own units. ``scale=
"fraction_of_mean"`` puts it in units of the estimate's own magnitude,
``tau_d = theta_d - b * |mean(theta)|``, which is what makes a commitment
comparable across channels with wildly different spend and what lines up with
the Cinelli–Hazlett bound (a bias expressed as a *fraction* of the estimate).
Both are affine in the draws, so the de-biased quantity stays exactly Gaussian
per draw and every quantity below is closed-form.

A **per-draw multiplicative** form, ``tau_d = theta_d * (1 - b_d)``, looks like
the natural way to say "relative" and is a trap; it is deliberately not offered:

* For an efficiency channel the break-even reference is **0**, and every media
  coefficient in this framework has positive support by construction (``Gamma``,
  ``LogNormal`` on ROI, or ``exp(...)``), so ``theta_d > 0`` for every draw and
  ``P(tau > 0) = P(b < 1)`` — a number with no data in it whatsoever, identical
  for every channel in the portfolio.
* On draws that cross zero (``counterfactual_roi`` and ``marginal_roas``
  numerators routinely do) the direction of the bias flips with the sign of the
  draw, so "the confounder inflated this" silently becomes "the confounder
  deflated this" partway through the posterior.
* The product of two normals is skewed with heavy tails, so the closed forms
  below would stop being exact.

``fraction_of_mean`` has none of these properties: it is a rescaling of
``absolute``, so at reference 0 it still reads the draws, it never flips
direction, and it stays Gaussian. At reference 0 it is also *scale-invariant* —
two posteriors differing only by a change of units give the same answer, and what
distinguishes them is the shape of the posterior relative to its own mean. That
is the property that makes a commitment quotable across a portfolio, and it is a
strictly different statement from ``P(b < 1)``, which is the same number for every
channel no matter what its posterior looks like.

Exact, not Monte-Carlo
----------------------
The de-biased posterior is a Gaussian mixture over draws you already have, so
nothing here samples: ``tau | theta_d ~ N(theta_d - mu, sigma^2)`` and therefore
``P(tau > r) = mean_d Phi((theta_d - mu - r) / sigma)`` in closed form. Three
things follow, all of which matter: results are deterministic (no seed to carry
around and reproduce), they are exact rather than noisy at small draw counts —
which is what makes a bisected tipping point trustworthy — and the identical
arithmetic runs in JavaScript, so the interactive report can recompute a tipping
point in the browser without another posterior pass.

Read this before quoting a number
---------------------------------
* **A tipping point is not evidence of causality.** It says "a confounder this
  strong would overturn the conclusion", never "the conclusion is correct". A
  high tipping point means the *required* confounder is implausible, which is an
  argument, not a proof. The route to genuine causal anchoring is a randomized
  experiment (:mod:`mmm_framework.calibration`).
* **The implied prior on ``tau`` is not the one anyone elicited.** Subtracting an
  independent bias is exactly right when the input draws are a *flat-prior*
  posterior of ``tau + beta``; then the data are provably uninformative about
  ``beta`` and its posterior equals its prior. An MMM posterior is not flat — it
  carries a ``LogNormal(0,1)`` ROI or ``Gamma(mu=1.5, sigma=1)`` coefficient
  prior — so the effective prior on ``tau`` is the media prior convolved with the
  bias prior. That is coherent but *undeclared*, which is why
  :func:`bias_sensitivity_report` accepts ``prior_draws`` and, when given them, reports
  what the commitment implies **before any data** (:attr:`BiasSensitivity.
  implied_prior_prob`). A conclusion whose prior already clears the threshold was
  not established by this analysis.
* **Positive-support priors can make the headline probability free.** When
  ``prob_at_zero_bias`` is exactly 1.0 the posterior places no mass below the
  reference *by construction of the prior*, so every unit of doubt in the output
  came from the bias prior rather than from the data. Flagged as
  :attr:`BiasSensitivity.positivity_constrained`.
* **A tight prior inflates the tipping point**, exactly as it inflates the
  robustness value (see :mod:`mmm_framework.validation.sensitivity_unobserved`).
  A narrow posterior needs a larger bias to be pushed across break-even, and that
  narrowness can come from the prior. Pass ``prior_contraction`` and let
  :func:`prior_dominance_caveat` refuse to quote the number.
* **The bias prior is the whole argument.** ``sigma`` pulled from thin air
  produces a tipping point pulled from thin air. Every :class:`BiasPrior` carries
  a ``source``, and every reporting layer shows it, so a ladder of named guesses
  can never be mistaken for a spread measured from a placebo distribution or
  benchmarked against an observed covariate.
* **This is a re-weighting, not a re-fit.** It widens and shifts an estimand; it
  cannot move a coefficient *relative to the controls*, because the model is
  never re-estimated. For that an unobserved confounder has to enter the graph.
* **Independent bias does not aggregate.** Unobserved demand is one latent factor
  hitting several channels at once. Combining per-channel priors as if they were
  independent diversifies the very thing being assumed, and makes a portfolio
  look more robust than any channel in it — hence :attr:`BiasPrior.correlation`
  and the refusal in the aggregation layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy.special import ndtr

__all__ = [
    "BiasPrior",
    "BiasScenario",
    "BiasSensitivity",
    "BiasSurface",
    "EValueResult",
    "TippingPoint",
    "DEFAULT_DECISION_THRESHOLD",
    "DEFAULT_INTERVAL_PROB",
    "FRAGILE_FRACTION",
    "NAMED_BIAS_PRIORS",
    "VERDICTS",
    "bias_adjusted_moments",
    "bias_sensitivity_report",
    "evalue",
    "mixture_interval",
    "mixture_moments",
    "named_prior_ladder",
    "prior_dominance_caveat",
    "prob_above",
    "sensitivity_surface",
    "tipping_point",
    "tipping_point_mu",
]

#: ``"absolute"`` — bias in the estimand's own units. ``"fraction_of_mean"`` —
#: bias as a fraction of ``|mean(draws)|``, the portable scale (see the module
#: docstring for why a per-draw multiplicative scale is not offered).
BiasScale = Literal["absolute", "fraction_of_mean"]

#: Posterior probability a conclusion must clear to count as supported. 0.95 is
#: the source notebook's convention and the default on every surface here.
DEFAULT_DECISION_THRESHOLD = 0.95

#: Interval mass reported alongside each scenario. Equal-tailed, matching the
#: framework's ``compute_hdi_bounds`` convention (percentile-based despite the
#: name), so an adjusted interval means the same thing as the one beside it.
DEFAULT_INTERVAL_PROB = 0.90

#: A conclusion overturned by a bias no larger than 30% of its own estimate is
#: fragile. Tied to the ``moderate`` rung of the named ladder so the threshold
#: and the vocabulary cannot drift apart.
FRAGILE_FRACTION = 0.30

#: Verdict vocabulary. Deliberately avoids "robust" as a stored value:
#: ``resilient`` claims only that the conclusion survived the range actually
#: scanned, which is all this device can establish.
VERDICTS = ("overturned", "fragile", "resilient", "not_assessable")

#: The analyst-commitment ladder from the source notebook, on the
#: fraction-of-estimate scale so a rung means the same for a $2M channel and a
#: $20k one. ``dismissive`` = "I do not believe meaningful confounding is
#: present"; ``skeptical`` = "this could be off by most of its own size". These
#: are **named guesses** and are labelled as such — prefer a prior measured from
#: a placebo distribution or benchmarked against an observed covariate.
NAMED_BIAS_PRIORS: dict[str, float] = {
    "dismissive": 0.05,
    "moderate": 0.30,
    "skeptical": 0.70,
}

#: Points kept on a transported tipping-point curve.
_CURVE_POINTS = 40

# Grids stop here by default: a bias commitment wider than 150% of the estimate
# is no longer a sensitivity analysis, it is the statement that the estimate
# carries no information.
_DEFAULT_MAX_FRACTION = 1.5


# --------------------------------------------------------------------------- #
# the prior
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BiasPrior:
    """A commitment about how much unmeasured confounding could be present.

    Parameters
    ----------
    mu:
        Prior mean of the bias. **Positive means the observed estimate is too
        large** — the usual direction for demand-chasing media — and shifts the
        estimate *down* by that amount. Zero is the agnostic default: bias is
        possible but unsigned.
    sigma:
        Prior standard deviation — the width of the commitment, and the quantity
        a tipping point is usually expressed in.
    scale:
        ``"fraction_of_mean"`` (default) reads ``mu``/``sigma`` as fractions of
        ``|mean(draws)|``; ``"absolute"`` reads them in the estimand's own units.
    correlation:
        Whether this bias is assumed independent across the quantities it is
        applied to, or driven by one shared latent factor. Unobserved demand is
        ``"shared"``; the field exists so an aggregation layer can refuse to sum
        independent biases and quietly diversify away the assumption.
    label:
        Short human-readable name, e.g. ``"moderate"`` or ``"1x Price"``.
    source:
        **Where the number came from.** ``"named"`` for a ladder rung,
        ``"placebo"`` / ``"aa_simulation"`` / ``"pre_trend"`` for a measured one,
        ``"benchmark:<covariate>"`` for a Cinelli–Hazlett benchmark. Carried into
        every report so a guess can never be presented as a measurement.
    """

    mu: float = 0.0
    sigma: float = 0.0
    scale: BiasScale = "fraction_of_mean"
    correlation: Literal["independent", "shared"] = "shared"
    label: str = "custom"
    source: str = "named"

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError(f"bias prior mu must be finite, got {self.mu!r}")
        if not math.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError(
                f"bias prior sigma must be finite and non-negative, got {self.sigma!r}"
            )
        if self.scale not in ("absolute", "fraction_of_mean"):
            raise ValueError(
                "scale must be 'absolute' or 'fraction_of_mean' (a per-draw "
                "multiplicative scale is deliberately not offered — see the "
                f"module docstring), got {self.scale!r}"
            )
        if self.correlation not in ("independent", "shared"):
            raise ValueError(
                f"correlation must be 'independent' or 'shared', got "
                f"{self.correlation!r}"
            )

    @property
    def is_measured(self) -> bool:
        """Whether this prior came from evidence rather than from the ladder."""
        return self.source not in ("named", "custom", "scan", "grid", "")

    def to_absolute(self, magnitude: float) -> "BiasPrior":
        """This prior in estimand units, given the estimate's magnitude.

        ``magnitude`` should be ``|mean(draws)|``. Both ``mu`` and ``sigma`` scale
        by it, so a positive ``mu`` shifts the estimate down by that fraction of
        its own size regardless of the estimate's sign.
        """
        if self.scale == "absolute":
            return self
        m = abs(float(magnitude))
        return BiasPrior(
            mu=self.mu * m,
            sigma=self.sigma * m,
            scale="absolute",
            correlation=self.correlation,
            label=self.label,
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu": float(self.mu),
            "sigma": float(self.sigma),
            "scale": self.scale,
            "correlation": self.correlation,
            "label": self.label,
            "source": self.source,
            "is_measured": bool(self.is_measured),
        }


def named_prior_ladder(scale: BiasScale = "fraction_of_mean") -> list[BiasPrior]:
    """The dismissive / moderate / skeptical ladder as :class:`BiasPrior` objects.

    Only meaningful on ``"fraction_of_mean"``, where a rung means the same thing
    whatever the estimand's units. Asking for ``"absolute"`` returns the same
    bare numbers, which is almost certainly not what you want — they are marked
    ``source="named"`` so the arbitrariness is at least visible downstream.
    """
    return [
        BiasPrior(mu=0.0, sigma=sigma, scale=scale, label=label, source="named")
        for label, sigma in NAMED_BIAS_PRIORS.items()
    ]


# --------------------------------------------------------------------------- #
# the mixture (exact, RNG-free)
# --------------------------------------------------------------------------- #


def _finite_draws(draws: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(draws, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _components(
    draws: np.ndarray,
    prior: BiasPrior,
    *,
    magnitude: float,
    base_sd: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-draw ``(mean, sd)`` of the de-biased quantity.

    The prior is converted to absolute units once, at ``magnitude``, so the whole
    engine works in one scale and the two public scales cannot diverge.

    ``base_sd`` is measurement noise *not* already in ``draws`` — used by the
    point-estimate path, where ``draws`` is the single observed effect and the
    standard error has to be added in quadrature. Zero on the posterior path,
    where the draws already carry the uncertainty.
    """
    p = prior.to_absolute(magnitude)
    m = draws - p.mu
    s = np.full(draws.shape, math.hypot(float(base_sd), p.sigma))
    return m, s


def _mixture_sf(m: np.ndarray, s: np.ndarray, t: float) -> float:
    """``P(X > t)`` for the equal-weight mixture of ``N(m_d, s_d^2)``.

    Components with ``s_d == 0`` are point masses contributing a hard 0/1 — the
    ``sigma = 0`` case, which must return the unmodified probability rather than
    divide by zero.
    """
    if m.size == 0:
        return float("nan")
    out = np.empty(m.shape, dtype=float)
    degenerate = s <= 0
    if degenerate.any():
        out[degenerate] = (m[degenerate] > t).astype(float)
    live = ~degenerate
    if live.any():
        out[live] = ndtr((m[live] - t) / s[live])
    return float(out.mean())


def mixture_moments(m: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of the equal-weight Gaussian mixture."""
    if m.size == 0:
        return float("nan"), float("nan")
    mean = float(m.mean())
    var = float((s**2 + m**2).mean()) - mean**2
    return mean, float(math.sqrt(max(var, 0.0)))


def _mixture_quantile(m: np.ndarray, s: np.ndarray, q: float) -> float:
    """Inverse of the mixture CDF at ``q``, by bisection (monotone, so safe)."""
    spread = float(s.max()) if s.size else 0.0
    lo = float(m.min()) - 10.0 * spread - 1e-9
    hi = float(m.max()) + 10.0 * spread + 1e-9
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return float("nan")
    target = 1.0 - float(q)  # work with the survival function
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _mixture_sf(m, s, mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, abs(hi)):
            break
    return 0.5 * (lo + hi)


def mixture_interval(
    m: np.ndarray, s: np.ndarray, prob: float = DEFAULT_INTERVAL_PROB
) -> tuple[float, float]:
    """Equal-tailed interval of the mixture, by bisection on its exact CDF."""
    if m.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - float(prob)) / 2.0
    return _mixture_quantile(m, s, tail), _mixture_quantile(m, s, 1.0 - tail)


def prob_above(
    draws: Sequence[float] | np.ndarray,
    prior: BiasPrior,
    reference: float,
    *,
    base_sd: float = 0.0,
    magnitude: float | None = None,
) -> float:
    """``P(de-biased quantity > reference)`` under ``prior``. Exact, no sampling."""
    x = _finite_draws(draws)
    if x.size == 0:
        return float("nan")
    mag = abs(float(x.mean())) if magnitude is None else abs(float(magnitude))
    m, s = _components(x, prior, magnitude=mag, base_sd=base_sd)
    return _mixture_sf(m, s, float(reference))


# --------------------------------------------------------------------------- #
# the conjugate point-estimate solve (the notebook's model)
# --------------------------------------------------------------------------- #


def bias_adjusted_moments(
    estimate: float,
    se: float,
    prior: BiasPrior,
    *,
    tau_prior_sd: float | None = None,
    tau_prior_mean: float = 0.0,
) -> tuple[float, float]:
    """Posterior mean and sd of the causal effect under a bias prior.

    Solves ``d_hat = tau + beta`` in closed form with ``tau ~ N(tau_prior_mean,
    tau_prior_sd)`` and ``beta ~ N(mu, sigma)`` — the conjugate case of the source
    notebook's model, no MCMC.

    ``tau_prior_sd=None`` (the default) takes the flat-``tau`` limit, which
    reduces **exactly** to ``N(d_hat - mu, se^2 + sigma^2)``. That is the right
    default here: the effect prior belongs to whoever produced ``estimate``, and
    layering another one on top of an MMM posterior would count the same
    information twice. The parameter is kept explicit so the equivalence between
    this path and the draws path is executable rather than asserted — at
    ``tau_prior_sd=1e6`` the two must agree.
    """
    if not math.isfinite(estimate):
        raise ValueError(f"estimate must be finite, got {estimate!r}")
    if not math.isfinite(se) or se <= 0:
        raise ValueError(f"se must be positive and finite, got {se!r}")
    p = prior.to_absolute(estimate)

    if tau_prior_sd is None:
        return float(estimate - p.mu), float(math.hypot(se, p.sigma))

    if not math.isfinite(tau_prior_sd) or tau_prior_sd <= 0:
        raise ValueError(
            f"tau_prior_sd must be positive and finite (or None), got {tau_prior_sd!r}"
        )
    if p.sigma <= 0:
        # A point-mass bias prior: tau is the shifted estimate updated by its own
        # prior. The general solve would divide by zero.
        precision = 1.0 / tau_prior_sd**2 + 1.0 / se**2
        mean = (
            tau_prior_mean / tau_prior_sd**2 + (estimate - p.mu) / se**2
        ) / precision
        return float(mean), float(math.sqrt(1.0 / precision))

    prior_precision = np.diag([1.0 / tau_prior_sd**2, 1.0 / p.sigma**2])
    f = np.array([[1.0, 1.0]])
    post_precision = prior_precision + f.T @ f / se**2
    post_cov = np.linalg.inv(post_precision)
    prior_mean = np.array([tau_prior_mean, p.mu])
    rhs = prior_precision @ prior_mean + f.flatten() * float(estimate) / se**2
    post_mean = post_cov @ rhs
    return float(post_mean[0]), float(math.sqrt(post_cov[0, 0]))


# --------------------------------------------------------------------------- #
# tipping points
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TippingPoint:
    """The bias commitment at which a conclusion stops being supported.

    ``value`` is ``None`` in two very different situations, which
    :attr:`already_below` separates and which every caller must keep separate:
    the conclusion was never supported even at zero bias, or it survived the
    whole range scanned. Reporting the second as "robust" without saying what was
    scanned is exactly the overclaim this module exists to avoid — hence
    ``max_scanned`` on every instance and its appearance in :meth:`describe`.
    """

    value: float | None
    scale: BiasScale
    parameter: str  # "sigma" | "mu"
    crossed: bool
    already_below: bool
    max_scanned: float
    threshold: float
    #: Whether the probability fell monotonically across the scan. It does
    #: whenever the posterior sits clearly on one side of the reference; a
    #: posterior straddling it need not, and a bisection that assumed
    #: monotonicity would return a confident wrong answer.
    monotone: bool = True
    #: ``(commitment, P(tau > reference))`` pairs from the scan, thinned for
    #: transport. The chart and the browser draw this rather than recomputing —
    #: the scan already produced it, and a redrawn curve that disagreed with the
    #: reported tipping point would be worse than no curve.
    curve: tuple[tuple[float, float], ...] = ()

    def describe(self, *, units: str = "") -> str:
        suffix = f" {units}" if units and self.scale == "absolute" else ""
        if self.already_below:
            return (
                f"not supported even with no assumed bias "
                f"(P < {self.threshold:.0%} at zero bias)"
            )
        if not self.crossed or self.value is None:
            span = (
                f"{self.max_scanned:.0%} of the estimate"
                if self.scale == "fraction_of_mean"
                else f"{self.max_scanned:g}{suffix}"
            )
            return f"still supported at the widest bias scanned ({span})"
        magnitude = (
            f"{self.value:.0%} of the estimate"
            if self.scale == "fraction_of_mean"
            else f"{self.value:g}{suffix}"
        )
        noun = "spread" if self.parameter == "sigma" else "overstatement"
        return (
            f"overturned by a bias {noun} of {magnitude} "
            f"(P falls below {self.threshold:.0%})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else float(self.value),
            "scale": self.scale,
            "parameter": self.parameter,
            "crossed": bool(self.crossed),
            "already_below": bool(self.already_below),
            "max_scanned": float(self.max_scanned),
            "threshold": float(self.threshold),
            "monotone": bool(self.monotone),
            "curve": [[float(a), float(b)] for a, b in self.curve],
        }


def _refine(fn, lo: float, hi: float, threshold: float, iterations: int = 60) -> float:
    """Bisect ``fn`` (a probability, decreasing across the bracket) to ``threshold``."""
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if fn(mid) > threshold:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _tipping(
    x: np.ndarray,
    *,
    reference: float,
    scale: BiasScale,
    parameter: str,
    fixed: float,
    threshold: float,
    max_value: float,
    n_grid: int,
    base_sd: float,
    magnitude: float,
) -> TippingPoint:
    """Shared scan-then-bisect for both tipping points.

    A **scan**, not a monotonicity assumption. ``P(tau > r)`` tends to 0.5 as the
    commitment widens, so it is monotone in the overwhelmingly common case where
    the conclusion starts well above 0.5 — but a posterior with mass on both
    sides of the reference need not be, and bisecting a non-monotone function
    would return a confident wrong answer. The realized monotonicity is reported.
    """
    empty = TippingPoint(
        value=None,
        scale=scale,
        parameter=parameter,
        crossed=False,
        already_below=False,
        max_scanned=float(max_value),
        threshold=float(threshold),
    )
    if x.size == 0:
        return empty

    def probe(value: float) -> float:
        mu, sigma = (fixed, value) if parameter == "sigma" else (value, fixed)
        p = BiasPrior(mu=mu, sigma=sigma, scale=scale, label="probe", source="scan")
        m, s = _components(x, p, magnitude=magnitude, base_sd=base_sd)
        return _mixture_sf(m, s, float(reference))

    grid = np.linspace(0.0, float(max_value), int(n_grid))
    probs = np.array([probe(g) for g in grid])
    monotone = bool(np.all(np.diff(probs) <= 1e-12))
    # Thin for transport. A deterministic stride keeps the endpoints and needs
    # no seed; ~40 points is plenty for a curve nobody zooms into.
    stride = max(1, int(grid.size // _CURVE_POINTS))
    keep = sorted({*range(0, grid.size, stride), grid.size - 1})
    curve = tuple((float(grid[i]), float(probs[i])) for i in keep)

    if probs[0] <= threshold:
        return TippingPoint(
            value=0.0,
            scale=scale,
            parameter=parameter,
            crossed=True,
            already_below=True,
            max_scanned=float(max_value),
            threshold=float(threshold),
            monotone=monotone,
            curve=curve,
        )

    below = np.nonzero(probs <= threshold)[0]
    if below.size == 0:
        return TippingPoint(
            value=None,
            scale=scale,
            parameter=parameter,
            crossed=False,
            already_below=False,
            max_scanned=float(max_value),
            threshold=float(threshold),
            monotone=monotone,
            curve=curve,
        )
    idx = int(below[0])
    value = _refine(probe, float(grid[idx - 1]), float(grid[idx]), float(threshold))
    return TippingPoint(
        value=float(value),
        scale=scale,
        parameter=parameter,
        crossed=True,
        already_below=False,
        max_scanned=float(max_value),
        threshold=float(threshold),
        monotone=monotone,
        curve=curve,
    )


def _default_span(
    x: np.ndarray, reference: float, scale: BiasScale, base_sd: float
) -> float:
    if scale == "fraction_of_mean":
        return _DEFAULT_MAX_FRACTION
    spread = float(x.std(ddof=1)) if x.size > 1 else 0.0
    gap = abs(float(x.mean()) - float(reference)) if x.size else 0.0
    return max(3.0 * (spread + base_sd) + 3.0 * gap, 1e-9)


def tipping_point(
    draws: Sequence[float] | np.ndarray,
    *,
    reference: float,
    scale: BiasScale = "fraction_of_mean",
    mu: float = 0.0,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    max_sigma: float | None = None,
    n_grid: int = 161,
    base_sd: float = 0.0,
    magnitude: float | None = None,
) -> TippingPoint:
    """Smallest bias *spread* that pushes ``P(tau > reference)`` below ``threshold``.

    ``max_sigma`` defaults to 150% of the estimate on the fraction scale; on the
    absolute scale it defaults to three times the draws' own spread plus three
    times their distance from the reference, so a conclusion far from break-even
    still gets a bracket wide enough to contain its crossing.
    """
    x = _finite_draws(draws)
    mag = (
        abs(float(x.mean()))
        if magnitude is None and x.size
        else float(magnitude or 0.0)
    )
    span = (
        _default_span(x, reference, scale, base_sd) if max_sigma is None else max_sigma
    )
    return _tipping(
        x,
        reference=reference,
        scale=scale,
        parameter="sigma",
        fixed=mu,
        threshold=threshold,
        max_value=span,
        n_grid=n_grid,
        base_sd=base_sd,
        magnitude=mag,
    )


def tipping_point_mu(
    draws: Sequence[float] | np.ndarray,
    *,
    reference: float,
    scale: BiasScale = "fraction_of_mean",
    sigma: float = 0.0,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    max_mu: float | None = None,
    n_grid: int = 161,
    base_sd: float = 0.0,
    magnitude: float | None = None,
) -> TippingPoint:
    """Smallest *systematic* overstatement that pushes the conclusion below threshold.

    The companion to :func:`tipping_point`: that one asks how uncertain the bias
    would have to be, this one asks how large a directional bias would have to be
    — "the estimate would have to be overstated by 18% for the recommendation to
    change". Usually the more legible of the two for a non-technical reader.
    """
    x = _finite_draws(draws)
    mag = (
        abs(float(x.mean()))
        if magnitude is None and x.size
        else float(magnitude or 0.0)
    )
    span = _default_span(x, reference, scale, base_sd) if max_mu is None else max_mu
    return _tipping(
        x,
        reference=reference,
        scale=scale,
        parameter="mu",
        fixed=sigma,
        threshold=threshold,
        max_value=span,
        n_grid=n_grid,
        base_sd=base_sd,
        magnitude=mag,
    )


# --------------------------------------------------------------------------- #
# the surface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BiasSurface:
    """``P(tau > reference)`` over a grid of ``(mu, sigma)`` commitments."""

    mu_grid: tuple[float, ...]
    sigma_grid: tuple[float, ...]
    #: ``prob[i][j]`` for ``sigma_grid[i]``, ``mu_grid[j]`` — row-major over sigma
    #: so it drops straight into a contour trace as ``z``.
    prob: tuple[tuple[float, ...], ...]
    scale: BiasScale
    reference: float
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu_grid": [float(v) for v in self.mu_grid],
            "sigma_grid": [float(v) for v in self.sigma_grid],
            "prob": [[float(v) for v in row] for row in self.prob],
            "scale": self.scale,
            "reference": float(self.reference),
            "threshold": float(self.threshold),
        }


def sensitivity_surface(
    draws: Sequence[float] | np.ndarray,
    *,
    reference: float,
    scale: BiasScale = "fraction_of_mean",
    mu_grid: Sequence[float] | None = None,
    sigma_grid: Sequence[float] | None = None,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    base_sd: float = 0.0,
    magnitude: float | None = None,
    max_draws: int = 2000,
) -> BiasSurface:
    """The two-dimensional audit: which analyst commitments support the conclusion.

    Draws are thinned to ``max_draws`` because the grid is evaluated
    ``len(mu_grid) * len(sigma_grid)`` times; the thinning is a deterministic
    stride rather than a sample, so the surface is reproducible without a seed.
    """
    x = _finite_draws(draws)
    if x.size > max_draws > 0:
        x = x[:: max(1, x.size // max_draws)][:max_draws]
    mag = (
        abs(float(x.mean()))
        if magnitude is None and x.size
        else float(magnitude or 0.0)
    )
    span = _default_span(x, reference, scale, base_sd)

    if mu_grid is None:
        mu_grid = np.linspace(-span / 3.0, span / 3.0, 25)
    if sigma_grid is None:
        sigma_grid = np.linspace(span / 100.0, span, 25)

    mus = [float(v) for v in np.asarray(mu_grid, dtype=float)]
    sigmas = [float(v) for v in np.asarray(sigma_grid, dtype=float)]
    rows: list[tuple[float, ...]] = []
    for s_b in sigmas:
        row: list[float] = []
        for m_b in mus:
            p = BiasPrior(mu=m_b, sigma=s_b, scale=scale, label="grid", source="grid")
            m, s = _components(x, p, magnitude=mag, base_sd=base_sd)
            row.append(_mixture_sf(m, s, float(reference)))
        rows.append(tuple(row))
    return BiasSurface(
        mu_grid=tuple(mus),
        sigma_grid=tuple(sigmas),
        prob=tuple(rows),
        scale=scale,
        reference=float(reference),
        threshold=float(threshold),
    )


# --------------------------------------------------------------------------- #
# E-value (ratio measures only)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EValueResult:
    """VanderWeele & Ding (2017) E-value, or a stated refusal.

    The E-value is the minimum strength of association — on the risk-ratio scale,
    with *both* the treatment and the outcome — that an unmeasured confounder
    would need to explain away an observed association. It is defined for **ratio
    measures on a rate or binary outcome** and nowhere else. An MMM's ROI is a
    ratio of two continuous quantities, not a risk ratio, so feeding it here would
    produce a number with no interpretation; this returns ``available=False`` with
    a reason instead.
    """

    available: bool
    point: float | None = None
    ci_limit: float | None = None
    measure: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "point": None if self.point is None else float(self.point),
            "ci_limit": None if self.ci_limit is None else float(self.ci_limit),
            "measure": self.measure,
            "reason": self.reason,
        }


#: Ratio measures the formula applies to directly. Odds and hazard ratios need a
#: documented conversion to the risk-ratio scale first (VanderWeele & Ding,
#: "Extensions"), and the right conversion depends on outcome rarity — so this
#: refuses rather than converting silently on the caller's behalf.
_EVALUE_MEASURES = frozenset({"risk_ratio", "rate_ratio", "prevalence_ratio"})


def _evalue_from_rr(rr: float) -> float:
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + math.sqrt(rr * (rr - 1.0)))


def evalue(
    estimate: float,
    *,
    measure: str = "risk_ratio",
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> EValueResult:
    """E-value for a ratio measure, with the null at 1.

    The CI E-value uses the confidence limit **closest to the null**, and is 1
    whenever the interval already crosses 1 — no confounding at all is needed to
    make the association compatible with the null.
    """
    if measure not in _EVALUE_MEASURES:
        return EValueResult(
            available=False,
            measure=measure,
            reason=(
                "the E-value is defined for ratio measures on a rate or binary "
                f"outcome ({', '.join(sorted(_EVALUE_MEASURES))}), not for "
                f"'{measure}'. An ROI or ROAS is a ratio of continuous "
                "quantities, not a risk ratio — use the bias-parameter tipping "
                "point instead."
            ),
        )
    if not math.isfinite(estimate) or estimate <= 0:
        return EValueResult(
            available=False,
            measure=measure,
            reason=f"a ratio measure must be positive and finite, got {estimate!r}",
        )

    point = _evalue_from_rr(float(estimate))
    ci_value: float | None = None
    if ci_low is not None and ci_high is not None:
        if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
            ci_value = None
        elif ci_low <= 1.0 <= ci_high:
            ci_value = 1.0
        else:
            limit = ci_low if ci_low > 1.0 else ci_high
            ci_value = _evalue_from_rr(float(limit))
    return EValueResult(available=True, point=point, ci_limit=ci_value, measure=measure)


# --------------------------------------------------------------------------- #
# the assembled report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BiasScenario:
    """The conclusion re-stated under one bias prior."""

    prior: BiasPrior
    mean: float
    sd: float
    lower: float
    upper: float
    prob_above: float
    supported: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior": self.prior.to_dict(),
            "label": self.prior.label,
            "source": self.prior.source,
            "mean": float(self.mean),
            "sd": float(self.sd),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "prob_above": float(self.prob_above),
            "supported": bool(self.supported),
        }


@dataclass(frozen=True)
class BiasSensitivity:
    """How much unmeasured confounding it would take to change the conclusion.

    ``verdict`` is one of :data:`VERDICTS`. ``resilient`` deliberately does not
    say "robust": it claims only that the conclusion survived every bias prior in
    the range actually scanned, which is stated in ``tipping_sigma.max_scanned``.
    """

    label: str
    estimate: float
    sd: float
    reference: float
    reference_label: str
    prob_at_zero_bias: float
    scenarios: tuple[BiasScenario, ...]
    tipping_sigma: TippingPoint
    tipping_mu: TippingPoint
    verdict: str
    scale: BiasScale
    threshold: float
    interval_prob: float
    units: str = ""
    surface: BiasSurface | None = None
    #: ``P(theta > reference)`` under the model's own prior, before any data —
    #: available only when ``prior_draws`` were supplied. A conclusion whose prior
    #: already clears the threshold was not established by this analysis.
    implied_prior_prob: float | None = None
    #: The posterior places *all* mass on the supported side, which for a
    #: positive-support media prior can be true by construction. Every unit of
    #: doubt in the output is then the bias prior's, not the data's.
    positivity_constrained: bool = False
    caveats: tuple[str, ...] = ()

    @property
    def is_fragile(self) -> bool:
        return self.verdict == "fragile"

    @property
    def is_assessable(self) -> bool:
        return self.verdict != "not_assessable"

    @property
    def measured_priors(self) -> tuple[BiasScenario, ...]:
        """Scenarios whose prior came from evidence rather than the named ladder."""
        return tuple(s for s in self.scenarios if s.prior.is_measured)

    def describe(self) -> str:
        """One plain sentence, phrased so it cannot be read as proof of causality."""
        if self.verdict == "not_assessable":
            return f"{self.label}: not assessable — no usable posterior draws."
        if self.verdict == "overturned":
            return (
                f"{self.label}: not supported even before any confounding is "
                f"assumed (P({self.reference_label}) = "
                f"{self.prob_at_zero_bias:.0%} < {self.threshold:.0%})."
            )
        return f"{self.label}: {self.tipping_mu.describe(units=self.units)}."

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "estimate": float(self.estimate),
            "sd": float(self.sd),
            "reference": float(self.reference),
            "reference_label": self.reference_label,
            "prob_at_zero_bias": float(self.prob_at_zero_bias),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "tipping_sigma": self.tipping_sigma.to_dict(),
            "tipping_mu": self.tipping_mu.to_dict(),
            "verdict": self.verdict,
            "scale": self.scale,
            "threshold": float(self.threshold),
            "interval_prob": float(self.interval_prob),
            "units": self.units,
            "is_fragile": bool(self.is_fragile),
            "is_assessable": bool(self.is_assessable),
            "implied_prior_prob": (
                None
                if self.implied_prior_prob is None
                else float(self.implied_prior_prob)
            ),
            "positivity_constrained": bool(self.positivity_constrained),
            "surface": None if self.surface is None else self.surface.to_dict(),
            "caveats": list(self.caveats),
            "description": self.describe(),
        }


def _verdict(
    prob_at_zero: float,
    tipping: TippingPoint,
    *,
    threshold: float,
    fragile_at: float,
) -> str:
    if not math.isfinite(prob_at_zero):
        return "not_assessable"
    if prob_at_zero <= threshold:
        return "overturned"
    if tipping.crossed and tipping.value is not None and tipping.value <= fragile_at:
        return "fragile"
    return "resilient"


def prior_dominance_caveat(
    contraction: float | None,
    verdict: str,
    *,
    threshold: float = 0.20,
) -> str | None:
    """Refuse to quote resilience that a tight prior bought.

    The tipping point rises as the posterior narrows, and in a Bayesian model
    that narrowness can come from the prior rather than from data — the same trap
    :func:`mmm_framework.validation.sensitivity_unobserved.prior_inflation_warning`
    guards on the robustness value. Returns ``None`` when there is nothing to warn
    about: contraction unknown (the check was not run — reported separately, never
    as a pass), contraction healthy, or the verdict already flags a problem so
    nobody is being over-reassured.
    """
    if contraction is None or not np.isfinite(contraction):
        return None
    if contraction >= threshold:
        return None
    if verdict in ("fragile", "overturned", "not_assessable"):
        return None
    return (
        f"prior-dominated (prior->posterior contraction {contraction:.2f} < "
        f"{threshold:.2f}): this posterior is mostly its prior, so the tipping "
        "point reflects prior tightness rather than evidence about confounding. "
        "Do not quote it as resilience, and do not compare it against a quantity "
        "whose prior is looser."
    )


def bias_sensitivity_report(
    draws: Sequence[float] | np.ndarray | None = None,
    *,
    reference: float,
    label: str = "estimate",
    reference_label: str = "above reference",
    estimate: float | None = None,
    se: float | None = None,
    priors: Sequence[BiasPrior] | None = None,
    scale: BiasScale = "fraction_of_mean",
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    interval_prob: float = DEFAULT_INTERVAL_PROB,
    fragile_at: float | None = None,
    units: str = "",
    include_surface: bool = True,
    prior_draws: Sequence[float] | np.ndarray | None = None,
    prior_contraction: float | None = None,
    extra_caveats: Sequence[str] = (),
) -> BiasSensitivity:
    """Full sensitivity report for one quantity.

    Takes **either** ``draws`` (a Bayesian posterior — the usual MMM case) **or**
    ``estimate`` + ``se`` (a single measurement — the usual experiment case). The
    second is handled as a one-component mixture carrying the standard error in
    ``base_sd``, so both paths run through identical arithmetic and cannot drift.

    ``prior_draws`` (draws of the same quantity from the model's *prior*) turn the
    undeclared-prior caveat into a number: see
    :attr:`BiasSensitivity.implied_prior_prob`.

    ``fragile_at`` defaults to :data:`FRAGILE_FRACTION` on the fraction scale; on
    the absolute scale it defaults to the same fraction of ``|estimate|``, which
    is the same statement in the estimand's units.
    """
    if draws is None and (estimate is None or se is None):
        raise ValueError(
            "bias_sensitivity needs either posterior `draws` or both `estimate` "
            "and `se`."
        )

    base_sd = 0.0
    if draws is None:
        x = np.array([float(estimate)], dtype=float)
        base_sd = float(se)  # type: ignore[arg-type]
        if not math.isfinite(base_sd) or base_sd <= 0:
            raise ValueError(f"se must be positive and finite, got {se!r}")
    else:
        x = _finite_draws(draws)

    point = float(x.mean()) if x.size else float("nan")
    magnitude = abs(point) if math.isfinite(point) else 0.0
    spread = float(math.hypot(float(x.std(ddof=1)), base_sd)) if x.size > 1 else base_sd

    zero = BiasPrior(mu=0.0, sigma=0.0, scale=scale, label="none", source="scan")
    m0, s0 = (
        _components(x, zero, magnitude=magnitude, base_sd=base_sd) if x.size else (x, x)
    )
    prob_zero = _mixture_sf(m0, s0, float(reference)) if x.size else float("nan")

    if fragile_at is None:
        fragile_at = (
            FRAGILE_FRACTION
            if scale == "fraction_of_mean"
            else FRAGILE_FRACTION * magnitude
        )

    ladder = list(priors) if priors is not None else named_prior_ladder(scale)
    scenarios: list[BiasScenario] = []
    for prior in ladder:
        m, s = _components(x, prior, magnitude=magnitude, base_sd=base_sd)
        mean, sd = mixture_moments(m, s)
        low, high = mixture_interval(m, s, interval_prob)
        p_above = _mixture_sf(m, s, float(reference))
        scenarios.append(
            BiasScenario(
                prior=prior,
                mean=mean,
                sd=sd,
                lower=low,
                upper=high,
                prob_above=p_above,
                supported=bool(p_above > threshold),
            )
        )

    common = {
        "reference": reference,
        "scale": scale,
        "threshold": threshold,
        "base_sd": base_sd,
        "magnitude": magnitude,
    }
    tip_sigma = tipping_point(x, **common)  # type: ignore[arg-type]
    tip_mu = tipping_point_mu(x, **common)  # type: ignore[arg-type]
    verdict = _verdict(
        prob_zero, tip_mu, threshold=threshold, fragile_at=float(fragile_at)
    )

    surface = None
    if include_surface and x.size:
        surface = sensitivity_surface(
            x,
            reference=reference,
            scale=scale,
            threshold=threshold,
            base_sd=base_sd,
            magnitude=magnitude,
        )

    implied_prior_prob: float | None = None
    if prior_draws is not None:
        pd = _finite_draws(prior_draws)
        if pd.size:
            implied_prior_prob = float((pd > float(reference)).mean())

    positivity = bool(math.isfinite(prob_zero) and prob_zero >= 1.0)

    caveats = list(extra_caveats)
    warning = prior_dominance_caveat(prior_contraction, verdict)
    if warning:
        caveats.append(warning)
    if positivity:
        caveats.append(
            "The posterior places no mass below the reference at all, which a "
            "positive-support media prior can guarantee before any data is seen. "
            "Every unit of doubt below therefore comes from the bias prior, not "
            "from the evidence."
        )
    if implied_prior_prob is not None and implied_prior_prob > threshold:
        caveats.append(
            f"The model's own prior already put P({reference_label}) = "
            f"{implied_prior_prob:.0%} before seeing any data, above the "
            f"{threshold:.0%} bar — so this conclusion is substantially the "
            "prior's, and the tipping point below inherits that."
        )
    if not any(s.prior.is_measured for s in scenarios):
        caveats.append(
            "Every bias prior here is a named guess, not a measurement. The "
            "tipping point is the useful output; the per-scenario probabilities "
            "are only as good as the commitment behind them."
        )
    if not tip_mu.monotone or not tip_sigma.monotone:
        caveats.append(
            "The decision probability did not fall monotonically across the "
            "scan — the posterior straddles the reference — so the tipping point "
            "is the FIRST crossing, not necessarily the only one. Read the "
            "surface rather than the single number."
        )

    return BiasSensitivity(
        label=label,
        estimate=point,
        sd=spread,
        reference=float(reference),
        reference_label=reference_label,
        prob_at_zero_bias=prob_zero,
        scenarios=tuple(scenarios),
        tipping_sigma=tip_sigma,
        tipping_mu=tip_mu,
        verdict=verdict,
        scale=scale,
        threshold=float(threshold),
        interval_prob=float(interval_prob),
        units=units,
        surface=surface,
        implied_prior_prob=implied_prior_prob,
        positivity_constrained=positivity,
        caveats=tuple(caveats),
    )
