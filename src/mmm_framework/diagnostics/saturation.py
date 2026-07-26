"""Is the saturation prior defensible, and did the data move it?

Saturation is **not identified from the sales likelihood**. That is settled, not a
property of this implementation: Jin et al. (Google, 2017) exhibit visually
identical Hill curves from very different parameters and call them "essentially
unidentifiable"; Dew et al. (2024) prove predictive fit — cross-validation
included — cannot arbitrate between observationally equivalent response
specifications. A transform search on this repo's own positive-control world
reproduces it: candidates whose out-of-sample error is within 10% of the best
disagree about the elbow across nearly the whole search range.

When a parameter is unidentified, **the prior is the answer**. Two consequences,
and this module is the pair of checks for them:

* :func:`saturation_prior_report` — *before* fitting, where does the prior put the
  curve's elbow relative to spend you have actually observed? Media reaches
  saturation normalized by the channel's training maximum, so the elbow is
  directly interpretable as a fraction of that maximum. Prior mass beyond 1.0 is
  mass on a region no observational data can move, and it comes back as posterior
  essentially unchanged.
* :func:`saturation_learning` — *after* fitting, how much did the posterior
  actually contract? A ``"prior-dominated"`` verdict here means the number in your
  report is the prior you chose, not something the data told you.

Neither check can make the parameter identifiable. What does that is **dose
spread** — a rank condition on the spend design, not a property of any estimator.
A single lift test at one spend level yields one equation in two unknowns;
:mod:`mmm_framework.planning.identification` reaches the same conclusion from a
Laplace bound and refuses to claim identification below three in-support levels.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..config.enums import SaturationType

if TYPE_CHECKING:  # pragma: no cover
    from ..model.base import BayesianMMM

__all__ = [
    "ELBOW_MASS_DIFFUSE",
    "ELBOW_MASS_UNANCHORED",
    "saturation_learning",
    "saturation_prior_report",
    "warn_if_saturation_prior_is_unanchored",
]

#: Prior mass beyond maximum observed spend above which the elbow prior is
#: reported as ``"diffuse"`` — a fifth of the answer is coming from a region the
#: data cannot speak to.
ELBOW_MASS_DIFFUSE = 0.05

#: ...and above which it is reported as ``"unanchored"`` and warned about.
ELBOW_MASS_UNANCHORED = 0.20

#: Saturation families whose parameter locates the curve's elbow, and how to get
#: the elbow (as a fraction of maximum observed spend) from it. ``root`` has no
#: asymptote and therefore no elbow, so it is absent by design rather than by
#: oversight.
_ELBOW_FROM: dict[SaturationType, tuple[str, Any]] = {
    SaturationType.LOGISTIC: ("sat_lam", lambda v: np.log(2.0) / np.maximum(v, 1e-12)),
    SaturationType.HILL: ("sat_half", lambda v: v),
    SaturationType.MICHAELIS_MENTEN: ("sat_half", lambda v: v),
    SaturationType.TANH: ("sat_half", lambda v: np.arctanh(0.5) * v),
}


def _elbow_verdict(mass_beyond: float) -> str:
    if not np.isfinite(mass_beyond):
        return "unknown"
    if mass_beyond >= ELBOW_MASS_UNANCHORED:
        return "unanchored"
    if mass_beyond >= ELBOW_MASS_DIFFUSE:
        return "diffuse"
    return "anchored"


def saturation_prior_report(
    model: "BayesianMMM", *, draws: int = 4000, random_seed: int | None = 0
) -> pd.DataFrame:
    """Where does the saturation prior put the curve's elbow? (pre-fit)

    Draws each channel's saturation parameter from its prior and converts it to
    the **half-saturation point as a fraction of the channel's maximum observed
    spend** — the scale saturation actually sees, since media is max-normalized
    before the transform.

    Args:
        model: A constructed :class:`~mmm_framework.model.base.BayesianMMM`. It
            need not be fitted; the graph is built if it has not been.
        draws: Prior draws per channel.
        random_seed: Seed for reproducibility.

    Returns:
        One row per channel with ``channel``, ``saturation``, ``parameter``,
        ``elbow_q05`` / ``elbow_median`` / ``elbow_q95`` (fractions of maximum
        observed spend), ``mass_beyond_support`` and ``verdict``
        (``anchored`` / ``diffuse`` / ``unanchored``).

        Channels whose family has no elbow (``root``, ``none``) are omitted.

    Note:
        A prior can be perfectly anchored and the parameter still unidentified —
        this reports whether the prior is *defensible*, not whether the data will
        move it. Pair it with :func:`saturation_learning` after fitting.
    """
    import pymc as pm

    graph = model.model  # lazy build
    rvs = {rv.name: rv for rv in graph.free_RVs}
    rows = []
    for channel in model.channel_names:
        cfg = model._get_saturation_config(channel)
        spec = _ELBOW_FROM.get(cfg.type)
        if spec is None:
            continue
        param, to_elbow = spec
        name = f"{param}_{channel}"
        # NB explicit `is None` checks: a TensorVariable has no truth value, so
        # `a or b` raises rather than falling through.
        rv = rvs.get(name)
        if rv is None:
            # An anchored Hill elbow is a Deterministic over a raw Beta.
            rv = {d.name: d for d in graph.deterministics}.get(name)
        if rv is None:
            continue
        samples = np.asarray(
            pm.draw(rv, draws=draws, random_seed=random_seed), dtype=float
        ).ravel()
        elbow = np.asarray(to_elbow(samples), dtype=float)
        elbow = elbow[np.isfinite(elbow)]
        if elbow.size == 0:
            continue
        beyond = float(np.mean(elbow > 1.0))
        rows.append(
            {
                "channel": channel,
                "saturation": cfg.type.value,
                "parameter": name,
                "elbow_q05": float(np.quantile(elbow, 0.05)),
                "elbow_median": float(np.median(elbow)),
                "elbow_q95": float(np.quantile(elbow, 0.95)),
                "mass_beyond_support": beyond,
                "verdict": _elbow_verdict(beyond),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "channel",
            "saturation",
            "parameter",
            "elbow_q05",
            "elbow_median",
            "elbow_q95",
            "mass_beyond_support",
            "verdict",
        ],
    )


def warn_if_saturation_prior_is_unanchored(
    model: "BayesianMMM", *, draws: int = 2000, random_seed: int | None = 0
) -> pd.DataFrame:
    """Emit a ``UserWarning`` per channel whose elbow prior is unanchored.

    Returns the same frame as :func:`saturation_prior_report` so a caller can act
    on it. Silent when every channel is anchored — this is meant to be safe to
    call unconditionally.
    """
    report = saturation_prior_report(model, draws=draws, random_seed=random_seed)
    for row in report.itertuples():
        if row.verdict != "unanchored":
            continue
        warnings.warn(
            f"Channel {row.channel!r}: {row.mass_beyond_support:.0%} of the "
            f"{row.saturation} saturation prior puts the curve's elbow BEYOND "
            "maximum observed spend, where no observational data can move it. "
            "Saturation is not identified from the sales likelihood, so that mass "
            "comes back as posterior essentially unchanged. Anchor the prior to "
            "observed spend (SaturationConfig.lam_prior, or "
            "anchor_kappa_to_data=True for Hill), or read the fitted curve as a "
            "prior assumption rather than a finding.",
            UserWarning,
            stacklevel=2,
        )
    return report


def saturation_learning(
    model: "BayesianMMM",
    results: Any = None,
    *,
    draws: int = 2000,
    random_seed: int | None = 0,
) -> pd.DataFrame:
    """Did the data move the saturation parameters? (post-fit)

    Thin wrapper over
    :func:`mmm_framework.diagnostics.learning.parameter_learning`, restricted to
    the saturation block and with the prior drawn for you.

    Args:
        model: A **fitted** model.
        results: Unused; accepted so the call reads like the other diagnostics.
        draws: Prior draws to compare the posterior against.
        random_seed: Seed for the prior draw.

    Returns:
        The ``parameter_learning`` frame for the saturation parameters, sorted
        least-learned first. A ``"prior-dominated"`` verdict means the fitted
        value is the prior you chose.

    Raises:
        ValueError: If the model has not been fitted.
    """
    import pymc as pm

    from .learning import parameter_learning

    if getattr(model, "_trace", None) is None:
        raise ValueError("Model not fitted. Call fit() first.")

    graph = model.model
    names = [
        rv.name
        for rv in graph.free_RVs
        if rv.name.startswith(("sat_lam_", "sat_half_", "sat_slope_", "sat_exponent_"))
    ]
    if not names:
        return pd.DataFrame()

    rvs = {rv.name: rv for rv in graph.free_RVs}
    prior = {
        n: np.asarray(
            pm.draw(rvs[n], draws=draws, random_seed=random_seed), dtype=float
        ).ravel()
        for n in names
    }
    posterior = {
        n: np.asarray(model._trace.posterior[n].values, dtype=float).ravel()
        for n in names
        if n in model._trace.posterior
    }
    common = [n for n in names if n in posterior]
    if not common:
        return pd.DataFrame()
    return parameter_learning(
        {n: prior[n] for n in common}, {n: posterior[n] for n in common}
    )
