"""Cost-per-conversion (CPA) power math — the client metric that CANNOT be
naively inverted from the lift.

CPA = cost / incremental conversions. The lift estimate is approximately
Gaussian, but its reciprocal is not: dividing a constant by a noisy quantity
whose interval approaches zero produces an extremely right-skewed ratio
distribution — the estimated CPA has a long expensive tail, its mean
overshoots the truth, and the symmetric "delta-method" interval
``cpa ± z·(cost·se/lift²)`` under-covers (it can even go negative). Three
consequences this module encodes:

* **Maximum detectable CPA** — the planning summary that DOES survive
  inversion. A design that can detect lift ≥ MDE can certify
  ``CPA ≤ cost/MDE``; larger (worse) CPAs are indistinguishable from "no
  effect". Invert the *detectable-lift bound*, never the point estimate.
* **Interval by inverting the lift bound** — the CPA interval is the image of
  the lift interval under ``x → cost/x`` (monotone on the positive half-line):
  asymmetric, and **unbounded above** whenever the lift interval touches zero
  (the data cannot rule out an infinite cost per conversion — reporting that
  honestly is the point).
* **Power on the CPA scale** — power to come away with a *bounded* CPA at
  all, or to certify ``CPA ≤ target`` (equivalent to detecting
  ``lift ≥ cost/target``).

numpy only (kernel-safe). Cost and lift can be on any per-unit basis as long
as they share it (e.g. ghost ads: cost per treated user vs per-user lift).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPS = 1e-12


def _z(p: float) -> float:
    """Standard-normal quantile via erfinv (no scipy)."""
    from statistics import NormalDist

    return float(NormalDist().inv_cdf(p))


def _phi_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def max_detectable_cpa(cost: float, lift_mde: float) -> float:
    """The WORST (highest) cost per conversion the design can still certify:
    ``cost / MDE``. A true CPA above this is indistinguishable from "the media
    did nothing" — the honest planning summary on the CPA scale."""
    if cost <= 0 or lift_mde <= 0:
        return float("nan")
    return float(cost / lift_mde)


def cpa_interval(
    lift: float,
    se_lift: float,
    cost: float,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """CPA point + interval by inverting the lift interval — and, for
    contrast, the naive symmetric delta-method interval that clients are
    usually shown.

    Returns ``status``:

    * ``"bounded"`` — the lift interval is strictly positive; the CPA interval
      is ``[cost/lift_hi, cost/lift_lo]`` (asymmetric: the upper arm is longer).
    * ``"upper_unbounded"`` — the lift interval touches zero; only a lower CPA
      bound exists. The data cannot rule out an arbitrarily bad CPA.
    * ``"undefined"`` — the whole lift interval is ≤ 0; no positive-lift
      evidence, no CPA statement at all.
    """
    z = _z(1.0 - alpha / 2.0)
    lo_l, hi_l = lift - z * se_lift, lift + z * se_lift
    point = float(cost / lift) if lift > _EPS else float("nan")

    if lo_l > _EPS:
        status, lo, hi = "bounded", cost / hi_l, cost / lo_l
    elif hi_l > _EPS:
        status, lo, hi = "upper_unbounded", cost / hi_l, float("inf")
    else:
        status, lo, hi = "undefined", float("nan"), float("nan")

    # the naive symmetric interval: cpa ± z * cost*se/lift² (delta method)
    if lift > _EPS:
        se_naive = cost * se_lift / lift**2
        naive_lo, naive_hi = point - z * se_naive, point + z * se_naive
    else:
        se_naive = naive_lo = naive_hi = float("nan")

    return {
        "cpa": point,
        "lo": float(lo),
        "hi": float(hi),
        "status": status,
        "lift_interval": (float(lo_l), float(hi_l)),
        "naive_se": float(se_naive),
        "naive_lo": float(naive_lo),
        "naive_hi": float(naive_hi),
        "alpha": float(alpha),
    }


def cpa_power(
    cost: float,
    se_lift: float,
    true_lift: float,
    *,
    target_cpa: float | None = None,
    alpha: float = 0.05,
) -> float:
    """Power on the CPA scale.

    With ``target_cpa=None``: the probability the test comes away with a
    **bounded** CPA interval at all (the lift interval clears zero) —
    ``Φ(true_lift/se − z)``. With a target: the probability the CPA interval's
    upper end lands below ``target_cpa``, i.e. the test *certifies*
    ``CPA ≤ target`` — equivalent to the estimated lift clearing
    ``cost/target + z·se``.
    """
    if se_lift <= 0 or not math.isfinite(se_lift):
        return float("nan")
    z = _z(1.0 - alpha / 2.0)
    if target_cpa is None:
        thresh = 0.0
    else:
        if target_cpa <= 0:
            return float("nan")
        thresh = cost / target_cpa
    return float(_phi_cdf((true_lift - thresh) / se_lift - z))


def simulate_cpa_distribution(
    true_lift: float,
    se_lift: float,
    cost: float,
    *,
    alpha: float = 0.05,
    n_sims: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Monte-Carlo the CPA readout a design of this precision would produce —
    the demonstration that the reciprocal is NOT well-behaved even when the
    lift is.

    Simulates ``lift̂ ~ N(true_lift, se_lift)``, forms ``cpâ = cost/lift̂``
    where the estimate is positive (non-positive draws are "no measurable
    conversions" — CPA undefined/infinite), and scores both interval
    constructions against the true CPA. Returns the draws (for plotting) plus
    skew diagnostics and empirical coverage.
    """
    rng = np.random.default_rng(seed)
    lifts = rng.normal(true_lift, se_lift, size=int(n_sims))
    pos = lifts > _EPS
    cpas = np.full(lifts.shape, np.nan)
    cpas[pos] = cost / lifts[pos]
    true_cpa = cost / true_lift if true_lift > _EPS else float("nan")

    d = cpas[pos]
    mean = float(d.mean()) if d.size else float("nan")
    med = float(np.median(d)) if d.size else float("nan")
    sd = float(d.std()) if d.size else float("nan")
    skew = float(np.mean(((d - mean) / sd) ** 3)) if d.size and sd > 0 else float("nan")

    # per-sim coverage of the two interval constructions
    z = _z(1.0 - alpha / 2.0)
    lo_l, hi_l = lifts - z * se_lift, lifts + z * se_lift
    # inverted-bound: covered when the (possibly one-sided) interval holds truth
    inv_lo = np.where(hi_l > _EPS, cost / hi_l, np.nan)
    inv_hi = np.where(lo_l > _EPS, cost / lo_l, np.inf)
    inv_defined = hi_l > _EPS
    inv_cover = inv_defined & (inv_lo <= true_cpa) & (true_cpa <= inv_hi)
    # naive symmetric delta-method interval (only defined for positive lift̂)
    se_naive = np.where(
        pos, cost * se_lift / np.square(np.where(pos, lifts, 1.0)), np.nan
    )
    naive_cover = (
        pos & (cpas - z * se_naive <= true_cpa) & (true_cpa <= cpas + z * se_naive)
    )

    return {
        "true_cpa": float(true_cpa),
        "cpa_draws": d,
        "lift_draws": lifts,
        "frac_no_positive_lift": float(1.0 - pos.mean()),
        "mean": mean,
        "median": med,
        "sd": sd,
        "skewness": skew,
        "mean_over_true": float(mean / true_cpa) if true_cpa else float("nan"),
        "p_over_2x_true": (
            float(np.mean(d > 2.0 * true_cpa)) if d.size else float("nan")
        ),
        "coverage_inverted": float(inv_cover.mean()),
        "coverage_naive": float(naive_cover.mean()),
        "nominal": 1.0 - alpha,
        "n_sims": int(n_sims),
    }


__all__ = [
    "max_detectable_cpa",
    "cpa_interval",
    "cpa_power",
    "simulate_cpa_distribution",
]
