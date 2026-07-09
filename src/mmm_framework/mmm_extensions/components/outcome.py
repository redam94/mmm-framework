"""Pluggable outcome (KPI) likelihood for the single-outcome extension models.

NestedMMM historically hard-coded a Normal observation on the standardized
outcome. This dispatch lets a spec-declared ``LikelihoodConfig`` swap in a
heavier-tailed Student-t (robust to outliers) while keeping the **Normal path
byte-identical** (same ``sigma_y`` RV, same call). It applies only to the
single-outcome, Gaussian-scale extension models (Nested); the multi-outcome
models fit a joint MvNormal-LKJ whose correlation structure a per-outcome
non-Gaussian family cannot join, so they do not use this.
"""

from __future__ import annotations

from typing import Any

import pymc as pm


def _family(likelihood_config: Any) -> str:
    if likelihood_config is None:
        return "normal"
    fam = getattr(likelihood_config, "family", likelihood_config)
    return str(getattr(fam, "value", fam)).lower()


def build_outcome_likelihood(
    name: str,
    mu_standardized: Any,
    observed: Any,
    likelihood_config: Any = None,
    dims: str = "obs",
) -> Any:
    """Register the outcome observation RV on the standardized scale.

    ``normal`` (default) reproduces the historical ``HalfNormal('sigma_y', 0.5)``
    + Normal exactly. ``student_t`` adds a ``nu_y ~ Gamma(2, 0.1)`` d.o.f. RV
    (Juárez–Steel) for heavy tails. Any other family raises — the extension's
    fixed effect/noise priors assume a standardized-Gaussian outcome, so a
    count/bounded family belongs to a model that owns its observation block.
    """
    family = _family(likelihood_config)
    if family in ("normal", ""):
        sigma = pm.HalfNormal("sigma_y", sigma=0.5)
        return pm.Normal(
            name, mu=mu_standardized, sigma=sigma, observed=observed, dims=dims
        )
    if family in ("student_t", "studentt", "t"):
        sigma = pm.HalfNormal("sigma_y", sigma=0.5)
        nu = pm.Gamma("nu_y", alpha=2.0, beta=0.1)
        return pm.StudentT(
            name, nu=nu, mu=mu_standardized, sigma=sigma, observed=observed, dims=dims
        )
    raise NotImplementedError(
        f"Outcome likelihood family '{family}' is not supported by the extension "
        "models (which assume a standardized-Gaussian outcome). Supported: "
        "normal, student_t. A count/bounded KPI needs a model that owns its "
        "observation block."
    )


__all__ = ["build_outcome_likelihood"]
