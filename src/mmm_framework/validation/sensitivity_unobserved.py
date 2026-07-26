"""Sensitivity to unobserved confounding (Cinelli-Hazlett style).

The dominant threat to MMM causality is **unobserved demand**: budgets are set in
anticipation of demand, so spend correlates with a latent driver of the KPI that
no adjustment set can remove. No goodness-of-fit statistic detects this. The
honest response is to *quantify how strong such a confounder would have to be* to
overturn each channel's estimated effect, and to state plainly that identification
rests on a no-unobserved-confounding assumption.

This module computes, per channel, the **robustness value** (RV) and the partial
``R^2`` of the treatment with the outcome, following Cinelli & Hazlett (2020,
"Making Sense of Sensitivity"). The RV is the share of residual variance an
unobserved confounder would need to explain in *both* the channel's spend and the
KPI to reduce the estimated effect by a given fraction (to zero, by default).

Caveats (read before quoting numbers)
-------------------------------------
* This is an **OLS-analogy** robustness value. The MMM is Bayesian and
  hierarchical; ``t = posterior_mean(beta) / posterior_sd(beta)`` is used as a
  z-score analog and the degrees of freedom are *nominal* (``n_obs`` minus a
  small parameter count). Under pooling the effective parameter count differs, so
  the RV is approximate. Degrees of freedom are deliberately taken on the
  generous side, which *understates* the RV -- erring toward caution (flagging
  fragility) for a tool that must not overclaim robustness.
* A high RV does **not** prove the effect is causal; it only means a confounder
  would have to be implausibly strong to explain it away. The route to genuine
  causal anchoring is experiment calibration
  (:mod:`mmm_framework.calibration`), not this diagnostic.
* **A tight prior manufactures a high RV.** This is the caveat that bites hardest
  in a Bayesian setting and it deserves stating in full. :func:`robustness_value`
  is strictly increasing in ``|t|``, and ``t = posterior_mean / posterior_sd``.
  In OLS, ``posterior_sd``'s analogue is a standard error the *data* produced, so
  a large ``t`` really does mean the data pinned the effect. In a Bayesian MMM
  the posterior sd is a compromise between the likelihood and the prior, so
  tightening a channel's coefficient prior shrinks the posterior sd, inflates
  ``t``, and raises the reported RV — with no new evidence whatsoever. Precision
  that came from an assumption is then displayed as robustness to violating a
  *different* assumption, which is close to the opposite of what a sensitivity
  analysis is for.

  Two consequences follow. First, RV is only interpretable relative to a prior
  you are prepared to defend, so it should never be compared across channels
  whose priors differ in tightness (a calibrated channel and an uncalibrated one
  are not on the same scale). Second, the framework's default non-negative media
  prior already truncates and tightens; :func:`prior_inflation_warning` flags
  channels whose posterior is prior-dominated so a caller can refuse to quote the
  RV rather than quote it credulously. A channel with a high RV and low
  prior→posterior contraction has told you about your prior, not about your
  exposure to confounding.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .results import ChannelRobustness, UnobservedConfoundingSensitivity

logger = logging.getLogger(__name__)

# Below this robustness value a channel's effect is considered fragile to a
# plausible unobserved confounder (a confounder explaining <10% of residual
# variance in both spend and KPI would overturn it).
FRAGILE_RV_THRESHOLD = 0.10


def partial_r2_from_t(t_value: float, dof: int) -> float:
    """Partial ``R^2`` of treatment with outcome from a t-statistic."""
    if dof <= 0:
        return float("nan")
    t2 = float(t_value) ** 2
    return t2 / (t2 + dof)


def robustness_value(t_value: float, dof: int, q: float = 1.0) -> float:
    """Cinelli-Hazlett robustness value ``RV_q``.

    The partial ``R^2`` an unobserved confounder must have with *both* treatment
    and outcome (equal association) to reduce the point estimate by a fraction
    ``q`` (``q=1`` -> reduce to zero). Returns a value in ``[0, 1]``; larger means
    more robust.
    """
    if dof <= 0:
        return float("nan")
    f = abs(float(t_value)) / np.sqrt(dof)
    fq = q * f
    fq2 = fq**2
    rv = 0.5 * (np.sqrt(fq2**2 + 4.0 * fq2) - fq2)
    return float(np.clip(rv, 0.0, 1.0))


#: Below this prior->posterior contraction the coefficient's posterior sd is
#: mostly the prior's, so the robustness value it implies is an artifact of the
#: prior rather than evidence about confounding.
PRIOR_DOMINATED_CONTRACTION = 0.20


def prior_inflation_warning(
    contraction: float | None,
    robustness_value: float,
    *,
    threshold: float = PRIOR_DOMINATED_CONTRACTION,
) -> str | None:
    """Warn when a high RV was bought by a tight prior rather than by data.

    ``robustness_value`` is strictly increasing in ``|t| = |mean| / sd``. In OLS
    ``sd`` is a standard error the data produced; in a Bayesian MMM it is a
    compromise between likelihood and prior, so shrinking the prior shrinks
    ``sd``, inflates ``t``, and raises the RV with no new evidence. A coefficient
    whose posterior barely narrowed its prior therefore reports robustness it has
    not earned.

    Returns ``None`` when there is nothing to warn about (contraction unknown, or
    healthy, or the RV is already flagging fragility so nobody is being
    over-reassured).
    """
    if contraction is None or not np.isfinite(contraction):
        return None
    if contraction >= threshold:
        return None
    if not np.isfinite(robustness_value) or robustness_value < FRAGILE_RV_THRESHOLD:
        # Already reported as fragile — the RV is not being used to reassure.
        return None
    return (
        f"prior-dominated (contraction {contraction:.2f} < {threshold:.2f}): this "
        f"channel's posterior sd is mostly its prior's, so RV={robustness_value:.2f} "
        "reflects prior tightness, not evidence about confounding. Do not quote it "
        "as robustness, and do not compare it against a channel with a looser prior."
    )


def _prior_contraction(model: Any, beta_name: str, post_sd: float) -> float | None:
    """``1 - Var_post / Var_prior`` for one coefficient, or ``None``.

    Returns ``None`` whenever the trace carries no prior group — the check was
    not run, which is reported distinctly from having been run and passed.
    """
    trace = getattr(model, "_trace", None)
    if trace is None or not np.isfinite(post_sd) or post_sd <= 0:
        return None
    try:
        from mmm_framework.utils.arviz_compat import has_group

        if not has_group(trace, "prior"):
            return None
        prior = trace.prior
        if beta_name not in prior:
            return None
        prior_draws = np.asarray(prior[beta_name].values).reshape(-1)
        prior_draws = prior_draws[np.isfinite(prior_draws)]
        if prior_draws.size < 2:
            return None
        prior_sd = float(np.std(prior_draws, ddof=1))
        if not np.isfinite(prior_sd) or prior_sd <= 0:
            return None
        return float(1.0 - (post_sd**2) / (prior_sd**2))
    except Exception:  # noqa: BLE001 — a diagnostic must never sink the analysis
        logger.debug("prior contraction unavailable for %s", beta_name, exc_info=True)
        return None


class UnobservedConfoundingAnalysis:
    """Per-channel sensitivity of media effects to unobserved confounding.

    Parameters
    ----------
    model : BayesianMMM
        A fitted model exposing ``_trace.posterior`` with ``beta_<channel>``
        variables, ``channel_names`` and ``n_obs``.
    """

    def __init__(self, model: Any):
        if getattr(model, "_trace", None) is None:
            raise ValueError("Model must be fitted (no posterior trace found).")
        self.model = model

    def _nominal_dof(self) -> int:
        """Generous (caution-erring) nominal residual degrees of freedom."""
        n_obs = int(getattr(self.model, "n_obs", 0))
        n_media = len(getattr(self.model, "channel_names", []) or [])
        n_controls = int(getattr(self.model, "n_controls", 0) or 0)
        # Intentionally small parameter count: media + controls + intercept.
        p = n_media + n_controls + 1
        return max(1, n_obs - p)

    def run(self, q: float = 1.0) -> UnobservedConfoundingSensitivity:
        """Compute robustness values for every channel coefficient."""
        posterior = self.model._trace.posterior
        dof = self._nominal_dof()

        channels: list[ChannelRobustness] = []
        for channel in self.model.channel_names:
            beta_name = f"beta_{channel}"
            if beta_name not in posterior:
                logger.debug("No %s in posterior; skipping", beta_name)
                continue
            draws = np.asarray(posterior[beta_name].values).reshape(-1)
            mean = float(np.mean(draws))
            sd = float(np.std(draws, ddof=1)) if draws.size > 1 else float("nan")
            t_value = mean / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")

            rv = robustness_value(t_value, dof, q=q)
            contraction = _prior_contraction(self.model, beta_name, sd)
            warning = prior_inflation_warning(contraction, rv)
            if warning:
                logger.warning("Channel %s robustness value is %s", channel, warning)

            channels.append(
                ChannelRobustness(
                    channel=channel,
                    estimate=mean,
                    std_error=sd,
                    t_value=t_value,
                    dof=dof,
                    partial_r2=partial_r2_from_t(t_value, dof),
                    robustness_value=rv,
                    robustness_value_half=robustness_value(t_value, dof, q=0.5 * q),
                    fragile_threshold=FRAGILE_RV_THRESHOLD,
                    prior_contraction=contraction,
                    prior_dominated_threshold=PRIOR_DOMINATED_CONTRACTION,
                )
            )

        n_prior_dominated = sum(1 for c in channels if c.is_prior_dominated)
        n_unassessed = sum(1 for c in channels if c.prior_contraction is None)
        caveat = (
            "Identification rests on a NO-UNOBSERVED-CONFOUNDING assumption. "
            "The robustness value is the partial R^2 a hidden confounder would "
            "need with both a channel's spend and the KPI to nullify its "
            "estimated effect. It is an OLS-analogy approximation; anchor "
            "effects with randomized experiments (mmm_framework.calibration) "
            "for genuine causal validity. "
            "The RV rises with |estimate / standard error|, and in a Bayesian "
            "model a tighter prior shrinks that error — so a prior-dominated "
            "coefficient reports high robustness without new evidence. RVs are "
            "not comparable across channels whose priors differ in tightness."
        )
        if n_prior_dominated:
            caveat += (
                f" WARNING: {n_prior_dominated} channel(s) are prior-dominated "
                f"(prior->posterior contraction < {PRIOR_DOMINATED_CONTRACTION:.2f}); "
                "their robustness values describe the prior, not the evidence, and "
                "should not be quoted."
            )
        if n_unassessed:
            caveat += (
                f" {n_unassessed} channel(s) could not be checked for prior "
                "dominance (no prior group in the trace) — sample the prior "
                "predictive to enable the check rather than assuming they passed."
            )

        return UnobservedConfoundingSensitivity(
            channels=channels,
            dof=dof,
            q=q,
            caveat=caveat,
        )


__all__ = [
    "UnobservedConfoundingAnalysis",
    "robustness_value",
    "partial_r2_from_t",
    "prior_inflation_warning",
    "FRAGILE_RV_THRESHOLD",
    "PRIOR_DOMINATED_CONTRACTION",
]
