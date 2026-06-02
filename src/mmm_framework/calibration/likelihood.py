"""In-graph experiment likelihoods for the Bayesian MMM.

Where :mod:`mmm_framework.calibration.experiment` folds a lift test into an
*informative prior* on a channel's coefficient (a two-stage fit -> derive ->
refit), this module folds an experiment in as a **likelihood term inside the
PyMC graph**. The experiment's measured value becomes a data point whose model
expectation is the channel's estimand -- contribution, ROAS, or marginal ROAS --
expressed as a deterministic function of the *same* ``beta``, saturation, and
adstock parameters the time-series likelihood already estimates.

Why a likelihood (vs. a prior)
------------------------------
The prior route (:class:`~mmm_framework.calibration.ExperimentCalibrator`) holds
the first-stage saturation/adstock *shape* fixed when it inverts a measured lift
to a coefficient target ``beta_target = measured_lift / K_c`` -- so the derived
prior is marginally tighter than a fully joint treatment would justify, and it
can only encode a *contribution* (it has no notion of ROAS or mROAS). Adding the
experiment as a likelihood instead lets it update ``beta``, the s-curve, **and**
the adstock kernel *jointly*, and it generalises to any estimand that can be
written as a function of the graph::

    measured_value  ~  Normal( model_implied_estimand(theta), measured_se )

where ``theta`` are the channel's in-graph parameters and the estimand is one of:

* **contribution** -- total incremental KPI over the experiment window
  ``P``: ``y_std * sum_{t in P} beta_c * sat_c(adstock_c(x_{c,t}))`` (exactly the
  full-holdout counterfactual, since ``sat_c(0) = 0``);
* **ROAS** -- that contribution divided by the channel's observed spend over
  ``P`` (a known constant);
* **marginal ROAS** -- the incremental KPI from scaling the channel's spend over
  ``P`` by ``spend_lift_pct``, divided by the incremental spend; this re-evaluates
  the s-curve/adstock at the perturbed spend (a finite-difference matching how a
  geo *scaling* experiment is actually run).

The numpy-pure pieces (data structures, the lognormal moment conversion) live
here so they are unit-testable without PyMC; the model wires the per-channel
estimand tensors in :meth:`mmm_framework.model.base.BayesianMMM._add_experiment_likelihoods`
and calls :func:`attach_experiment_likelihood` to add the observed node.

Assumed semantics (read before use)
-----------------------------------
* ``value`` / ``se`` are on the experiment's natural scale: KPI units for
  ``contribution``; KPI-per-spend-dollar for ``roas`` / ``mroas``. The model
  estimand is converted to the same scale (via ``y_std`` and observed spend), so
  units must match -- a revenue KPI gives a true ROAS; a unit-volume KPI gives a
  cost-per-acquisition-inverse.
* The experiment window is summed at the model's aggregation level. With
  ``holdout_regions`` the obs mask is restricted to those geos so the estimand is
  the geo-restricted lift (the coefficient is still the pooled one); specifying
  ``holdout_regions`` on a model with no geo dimension is a configuration error.
* Carryover generated *during* ``P`` that lands *after* ``P`` is not counted (the
  estimand sums only over ``P``), matching
  :meth:`~mmm_framework.model.base.BayesianMMM.compute_marginal_contributions`.
* The time-series likelihood already sees the experiment window; adding the
  experiment as a second, independently-weighted measurement of the same period
  is the standard lift-calibration treatment (PyMC-Marketing, Meridian) -- it is
  not double counting in the pathological sense, but the experiment's ``se`` is
  what governs how hard it pulls the fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pytensor.tensor as pt

    from ..validation.results import LiftTestResult

# Floor for an estimand denominator / clip for log-scale stability.
_EPS = 1e-12


class ExperimentEstimand(str, Enum):
    """Unit in which an experiment's result is expressed.

    The three estimands are the same counterfactual contrast viewed three ways:
    ``CONTRIBUTION`` is the total incremental KPI over the window; ``ROAS``
    divides it by the window's spend; ``MROAS`` is the *marginal* version --
    incremental KPI from a small spend perturbation, per incremental dollar.
    """

    CONTRIBUTION = "contribution"
    ROAS = "roas"
    MROAS = "mroas"


@dataclass(frozen=True)
class ExperimentMeasurement:
    """A single experimental result to fold into the likelihood.

    Parameters
    ----------
    channel:
        Channel the experiment measured. Must be one of the model's channels.
    test_period:
        ``(start, end)`` of the experiment window, as dates (parsed against the
        panel) or integer period indices (as strings or ints).
    value:
        Measured point estimate, on the natural scale of ``estimand``.
    se:
        Standard error of ``value`` (the experiment's uncertainty). Must be > 0.
    estimand:
        Whether ``value`` is a contribution, a ROAS, or a marginal ROAS.
    spend_lift_pct:
        For ``MROAS`` only: the percentage by which the experiment scaled the
        channel's spend over the window (e.g. ``10.0`` for a +10% scaling cell).
        Required for ``MROAS``; ignored otherwise.
    spend:
        Optional override for the ``ROAS`` spend denominator. When ``None`` the
        channel's observed spend over the window is used. Not supported for
        ``MROAS`` -- its marginal-spend denominator is ``spend_lift_pct`` x the
        observed window spend (the same spend the perturbation scales), so an
        independent override would desynchronise numerator and denominator.
    holdout_regions:
        Geos the experiment was restricted to. Restricts the obs mask to those
        geos so the estimand is the geo-restricted lift. Requires a geo model.
    distribution:
        Measurement-error family. ``"normal"`` (default) places a Normal
        likelihood on ``value``; ``"lognormal"`` places a Normal on
        ``log(value)`` around ``log(estimand)`` with a moment-matched log-scale
        sd (median-matched; appropriate for strictly-positive ROAS).
    name:
        Optional explicit name for the likelihood node; auto-generated otherwise.
    outcome:
        For multi-outcome models (e.g. :class:`MultivariateMMM`,
        :class:`CombinedMMM`): which outcome the experiment measured. ``None`` for
        single-outcome models (the core model, :class:`NestedMMM`).
    """

    channel: str
    test_period: tuple[Any, Any]
    value: float
    se: float
    estimand: ExperimentEstimand = ExperimentEstimand.CONTRIBUTION
    spend_lift_pct: float | None = None
    spend: float | None = None
    holdout_regions: list[str] | None = None
    distribution: str = "normal"
    name: str | None = None
    outcome: str | None = None

    def __post_init__(self) -> None:
        estimand = ExperimentEstimand(self.estimand)
        object.__setattr__(self, "estimand", estimand)
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value!r}")
        if not math.isfinite(self.se) or self.se <= 0:
            raise ValueError(f"se must be a positive finite number, got {self.se!r}")
        if self.distribution not in ("normal", "lognormal"):
            raise ValueError(
                f"distribution must be 'normal' or 'lognormal', got "
                f"{self.distribution!r}"
            )
        if self.distribution == "lognormal" and self.value <= 0:
            raise ValueError(
                "lognormal measurement error requires a strictly positive value; "
                f"got {self.value!r}"
            )
        if estimand is ExperimentEstimand.MROAS:
            if self.spend_lift_pct is None or self.spend_lift_pct <= 0:
                raise ValueError(
                    "MROAS measurements require a positive spend_lift_pct (the "
                    "percentage the experiment scaled spend by)."
                )
            if self.spend is not None:
                raise ValueError(
                    "spend override is not supported for MROAS: the marginal-spend "
                    "denominator is derived from the observed window spend and "
                    "spend_lift_pct, so an independent override would desynchronise "
                    "the numerator and denominator. Use the ROAS estimand for a "
                    "custom spend denominator."
                )
        if self.spend is not None and self.spend <= 0:
            raise ValueError(f"spend override must be positive, got {self.spend!r}")

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "test_period": list(self.test_period),
            "value": self.value,
            "se": self.se,
            "estimand": self.estimand.value,
            "spend_lift_pct": self.spend_lift_pct,
            "spend": self.spend,
            "holdout_regions": (
                list(self.holdout_regions) if self.holdout_regions is not None else None
            ),
            "distribution": self.distribution,
            "name": self.name,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentMeasurement":
        period = data["test_period"]
        return cls(
            channel=data["channel"],
            test_period=(period[0], period[1]),
            value=data["value"],
            se=data["se"],
            estimand=ExperimentEstimand(data.get("estimand", "contribution")),
            spend_lift_pct=data.get("spend_lift_pct"),
            spend=data.get("spend"),
            holdout_regions=data.get("holdout_regions"),
            distribution=data.get("distribution", "normal"),
            name=data.get("name"),
            outcome=data.get("outcome"),
        )

    @classmethod
    def from_lift_test(cls, lift_test: "LiftTestResult") -> "ExperimentMeasurement":
        """Bridge a :class:`LiftTestResult` to a contribution measurement.

        A full-holdout lift test measures the channel's total incremental KPI
        over its window -- exactly the ``CONTRIBUTION`` estimand.
        """
        return cls(
            channel=lift_test.channel,
            test_period=tuple(lift_test.test_period),
            value=float(lift_test.measured_lift),
            se=float(lift_test.lift_se),
            estimand=ExperimentEstimand.CONTRIBUTION,
            holdout_regions=lift_test.holdout_regions,
        )

    def default_node_name(self, index: int) -> str:
        if self.name:
            return self.name
        parts = ["experiment", self.channel]
        if self.outcome:
            parts.append(self.outcome)
        parts.extend([self.estimand.value, str(index)])
        return "_".join(parts)


# =============================================================================
# Pure helpers (no PyMC -- directly unit-testable)
# =============================================================================


def lognormal_sigma_from_moments(value: float, se: float) -> float:
    """Log-scale sd of a lognormal whose natural-scale CV is ``se / value``.

    For ``X ~ LogNormal(mu, sigma)`` the coefficient of variation is
    ``sqrt(exp(sigma**2) - 1)``. Inverting at ``CV = se / value`` gives the
    log-scale spread that reproduces the experiment's relative uncertainty::

        sigma = sqrt( ln( 1 + (se / value)**2 ) )

    Used to place a (median-matched) lognormal measurement-error likelihood on a
    strictly-positive estimand such as ROAS.
    """
    if value <= 0:
        raise ValueError(f"lognormal value must be positive, got {value!r}")
    if se <= 0:
        raise ValueError(f"lognormal se must be positive, got {se!r}")
    cv2 = (se / value) ** 2
    return float(math.sqrt(math.log1p(cv2)))


# =============================================================================
# In-graph attachment (PyMC -- imported lazily so this module stays light)
# =============================================================================


def build_estimand_expr(
    estimand: ExperimentEstimand,
    *,
    contrib_window: "pt.TensorVariable",
    spend_window: float,
    scale: float = 1.0,
    contrib_window_pert: "pt.TensorVariable | None" = None,
    lift: float | None = None,
) -> "pt.TensorVariable":
    """Assemble a model-implied estimand from a window's contribution.

    Shared by the extension models (the core model has its own assembly with
    adstock). ``contrib_window`` is the summed per-obs contribution over the
    experiment window in *model* units; ``scale`` converts it to the estimand's
    natural scale (``y_std`` for a standardized model, ``1.0`` for a raw-``y``
    model). ``spend_window`` is the observed window spend (the denominator for
    ROAS / marginal spend for mROAS).

    For ``MROAS``, ``contrib_window_pert`` is the contribution under spend scaled
    by ``(1 + lift)`` within the window, and the result is the incremental KPI per
    incremental dollar.
    """
    contribution = contrib_window * scale
    if estimand is ExperimentEstimand.CONTRIBUTION:
        return contribution
    if estimand is ExperimentEstimand.ROAS:
        return contribution / spend_window
    if estimand is ExperimentEstimand.MROAS:
        if contrib_window_pert is None or lift is None:
            raise ValueError("MROAS estimand requires contrib_window_pert and lift")
        delta = (contrib_window_pert - contrib_window) * scale
        return delta / (lift * spend_window)
    raise ValueError(f"Unknown estimand: {estimand!r}")


def attach_experiment_likelihood(
    name: str,
    estimand_expr: "pt.TensorVariable",
    measurement: ExperimentMeasurement,
):
    """Add an observed likelihood comparing ``estimand_expr`` to a measurement.

    Reusable across model types: build the model-implied estimand tensor however
    your graph allows (it must be on the estimand's natural scale -- KPI units
    for a contribution, KPI-per-dollar for ROAS/mROAS), then call this *inside*
    the ``pm.Model`` context to fold the experiment into the joint posterior.

    Parameters
    ----------
    name:
        Name of the observed node (must be unique within the model).
    estimand_expr:
        PyTensor scalar: the model-implied estimand on the measurement's scale.
    measurement:
        The experimental result; its ``value``, ``se``, and ``distribution``
        define the likelihood.

    Returns
    -------
    The created observed random variable.
    """
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt

    # Expose the model-implied estimand as a named Deterministic so it can be
    # inspected alongside the measured value (diagnostics: "what the model thinks
    # the experiment should have measured") and used as an oracle in tests.
    pm.Deterministic(f"{name}_model_estimand", pt.as_tensor_variable(estimand_expr))

    if measurement.distribution == "lognormal":
        sigma_log = lognormal_sigma_from_moments(measurement.value, measurement.se)
        # log(value) ~ Normal(log(estimand), sigma_log): the estimand is the
        # median of the implied measurement distribution. Clip guards log(<=0).
        log_estimand = pt.log(pt.clip(estimand_expr, _EPS, np.inf))
        return pm.Normal(
            name,
            mu=log_estimand,
            sigma=sigma_log,
            observed=float(np.log(measurement.value)),
        )

    return pm.Normal(
        name,
        mu=estimand_expr,
        sigma=float(measurement.se),
        observed=float(measurement.value),
    )


__all__ = [
    "ExperimentEstimand",
    "ExperimentMeasurement",
    "attach_experiment_likelihood",
    "build_estimand_expr",
    "lognormal_sigma_from_moments",
]
