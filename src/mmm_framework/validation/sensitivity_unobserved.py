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

            channels.append(
                ChannelRobustness(
                    channel=channel,
                    estimate=mean,
                    std_error=sd,
                    t_value=t_value,
                    dof=dof,
                    partial_r2=partial_r2_from_t(t_value, dof),
                    robustness_value=robustness_value(t_value, dof, q=q),
                    robustness_value_half=robustness_value(t_value, dof, q=0.5 * q),
                    fragile_threshold=FRAGILE_RV_THRESHOLD,
                )
            )

        return UnobservedConfoundingSensitivity(
            channels=channels,
            dof=dof,
            q=q,
            caveat=(
                "Identification rests on a NO-UNOBSERVED-CONFOUNDING assumption. "
                "The robustness value is the partial R^2 a hidden confounder would "
                "need with both a channel's spend and the KPI to nullify its "
                "estimated effect. It is an OLS-analogy approximation; anchor "
                "effects with randomized experiments (mmm_framework.calibration) "
                "for genuine causal validity."
            ),
        )


__all__ = [
    "UnobservedConfoundingAnalysis",
    "robustness_value",
    "partial_r2_from_t",
    "FRAGILE_RV_THRESHOLD",
]
