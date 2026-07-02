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

Off-panel calibration (experiment ran in a different period)
------------------------------------------------------------
When an experiment was run in a window the model was **not** fit on, set
``eval_spend`` (+ ``eval_periods`` / ``eval_units`` / ``adstock_state``) on the
measurement. The estimand is then built by evaluating the channel's *global*
response curve ``beta_c * sat_c(adstock_c(.))`` at the experiment's own spend
level -- a deterministic function of the same in-graph structural parameters --
**without** indexing any training row. This rests on one assumption made
explicit here:

* **Structural stationarity.** The channel's response-curve parameters
  (``beta_c``, the saturation shape, the adstock kernel) are assumed *stable*
  between the experiment period and the training period. The experiment is
  evidence about these global, time-invariant parameters, so a measurement from
  a non-overlapping window still constrains them -- *provided* the curve has not
  shifted. This is the standard (usually implicit) assumption behind seeding a
  future-period MMM with a past lift test; off-panel mode only makes it
  load-bearing and visible. If you have reason to believe the curve moved
  (a major creative/format change, a structural break), prefer an in-window
  test. Steady-state vs cold-start carryover at ``eval_spend`` is chosen per
  experiment via ``adstock_state``.
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


def _positive_int(value: Any, name: str) -> int:
    """Coerce ``value`` to a positive ``int``, rejecting non-integral numbers.

    Integer-valued floats (``8.0``) are accepted; genuinely fractional values
    (``8.9``) are rejected rather than silently truncated, because the off-panel
    estimand scales linearly in these counts (window length / treated units), so
    a silent ``8.9 -> 8`` would bias the calibration. Non-numeric input raises
    the same domain message instead of a bare ``int()`` error.
    """
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if as_int < 1 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return as_int


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
    eval_spend:
        Off-panel calibration. The channel's spend **per period, per treated
        unit**, on the raw dollar scale, *during the experiment* -- supply this
        when the experiment ran in a window the model was **not** fit on. When
        set, the estimand is built by evaluating the channel's *global* response
        curve (the same in-graph ``beta``, saturation and adstock parameters) at
        this spend level instead of summing training-matrix rows, so the
        experiment window no longer has to overlap the training period. This is
        valid under **structural stationarity** -- the response-curve parameters
        are assumed stable between the experiment period and the training period
        (see "Assumed semantics"). Leave ``None`` for the standard in-panel route
        (sum the contribution over the training rows inside the window).
    eval_periods:
        Off-panel calibration: the number of periods the experiment ran (the
        window length ``W``). Required when ``eval_spend`` is set.
    eval_units:
        Off-panel calibration: the number of treated units (e.g. geos) that ran
        at ``eval_spend`` per period. Defaults to ``1`` (a single national-level
        stream). For a multi-geo holdout, set this to the number of treated geos
        -- the estimand assumes those units ran at the *same* per-unit spend
        (homogeneous treatment). Ignored unless ``eval_spend`` is set.
    adstock_state:
        Off-panel calibration: carryover convention at ``eval_spend``.
        ``"steady_state"`` (default) assumes the channel had been running at
        ~that spend long enough for adstock to converge (right for always-on /
        sustained-holdout / scale tests); ``"cold_start"`` assumes spend turned
        on from zero at the window start and carryover builds over ``W`` (right
        for a burst/pulse launched from dark). Always validated, but only affects
        the estimand when ``eval_spend`` is set.
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
    eval_spend: float | None = None
    eval_periods: int | None = None
    eval_units: int = 1
    adstock_state: str = "steady_state"

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

        if self.adstock_state not in ("steady_state", "cold_start"):
            raise ValueError(
                "adstock_state must be 'steady_state' or 'cold_start', got "
                f"{self.adstock_state!r}"
            )

        # ``eval_units`` is always coerced/validated: it has a concrete int
        # default and is read on the off-panel path, so keep its type honest
        # regardless of whether this particular measurement is off-panel.
        object.__setattr__(
            self, "eval_units", _positive_int(self.eval_units, "eval_units")
        )

        # Off-panel calibration is triggered by supplying ``eval_spend``: the
        # estimand is then evaluated on the channel's global response curve at
        # that spend rather than on training rows, so the window need not overlap
        # the training period (valid under structural stationarity).
        if self.eval_spend is not None:
            if (
                not isinstance(self.eval_spend, (int, float))
                or not math.isfinite(self.eval_spend)
                or self.eval_spend <= 0
            ):
                raise ValueError(
                    f"eval_spend must be a positive finite number, got "
                    f"{self.eval_spend!r}"
                )
            if self.eval_periods is None:
                raise ValueError(
                    "eval_spend requires a positive integer eval_periods (the "
                    "experiment's window length in periods)."
                )
            object.__setattr__(
                self, "eval_periods", _positive_int(self.eval_periods, "eval_periods")
            )
            if estimand is ExperimentEstimand.MROAS:
                raise ValueError(
                    "off-panel calibration (eval_spend) does not yet support the "
                    "MROAS estimand; use CONTRIBUTION or ROAS, or run the "
                    "experiment within the model's fitted window for mROAS."
                )
            if self.holdout_regions:
                raise ValueError(
                    "off-panel calibration cannot also set holdout_regions (there "
                    "are no training rows to restrict). Use eval_units to set the "
                    "number of treated units (assumed homogeneous per-unit spend)."
                )
            if self.spend is not None:
                raise ValueError(
                    "off-panel calibration derives the ROAS denominator from "
                    "eval_spend x eval_periods x eval_units; do not also set the "
                    "spend override."
                )

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
            "eval_spend": self.eval_spend,
            "eval_periods": self.eval_periods,
            "eval_units": self.eval_units,
            "adstock_state": self.adstock_state,
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
            eval_spend=data.get("eval_spend"),
            eval_periods=data.get("eval_periods"),
            eval_units=data.get("eval_units", 1),
            adstock_state=data.get("adstock_state", "steady_state"),
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


@dataclass(frozen=True)
class ShareMeasurement:
    """An observed within-channel *share vector* to fold into the likelihood.

    Where :class:`ExperimentMeasurement` carries a scalar estimand (contribution
    / ROAS / mROAS) for one channel, this carries a **composition**: the observed
    mean shares of a virtual parent channel's breakout sub-streams (e.g. from a
    continuous-learning program's per-arm posterior, or a creative-level lift
    study). The model-implied counterpart is a simplex expression such as the
    breakout model's ``breakout_share_<C>`` Deterministic; the two are compared
    via :func:`attach_share_likelihood`.

    Two measurement-error families:

    * ``"logistic_normal"`` (default) -- an MvNormal on the ``K-1`` additive
      log-ratios (ALR) w.r.t. the **last** breakout as reference:
      ``z_k = log(share_k / share_K)``. Matches the simplex's ``K-1`` degrees of
      freedom exactly and preserves the correlation structure of the source
      draws. Requires ``log_ratio_cov``.
    * ``"dirichlet"`` -- ``shares ~ Dirichlet(concentration * model_share)``.
      A single-scalar-precision alternative with the Dirichlet's rigid
      negative-correlation structure. Requires ``concentration``.

    Parameters
    ----------
    channel:
        The VIRTUAL parent channel the shares decompose (must be one of the
        model's channels, e.g. a breakout group's parent name).
    breakouts:
        The parent's sub-stream column names (MFF media columns), in the
        explicit order the ``shares`` (and ``log_ratio_cov``) refer to.
        At least 2, unique. The ALR reference is the **last** entry.
    shares:
        Observed mean shares, one per breakout, same order. Must be
        non-negative and sum to ~1; they are floored at a tiny epsilon and
        renormalized to an exact simplex on construction.
    log_ratio_cov:
        ``(K-1, K-1)`` covariance of the ALR log-ratios (last breakout as
        reference). Must be symmetric and **strictly** positive definite
        (checked via Cholesky -- a singular/PSD matrix would make the
        ``MvNormal`` logp ``-inf`` deep inside the fit; add a small diagonal
        ridge, e.g. ``1e-9``, to an empirically-estimated covariance).
        Required for ``"logistic_normal"``; must be ``None`` for
        ``"dirichlet"``. NOTE: the ALR coordinates depend on the
        breakout ORDER -- reordering ``breakouts`` requires re-deriving the
        covariance, so consumers require an exact order match rather than
        reindexing.
    concentration:
        Dirichlet precision (larger = tighter). Required for ``"dirichlet"``;
        must be ``None`` for ``"logistic_normal"``.
    distribution:
        ``"logistic_normal"`` (default) or ``"dirichlet"``.
    name:
        Optional explicit name for the likelihood node; auto-generated
        otherwise.
    source:
        Optional provenance (e.g. ``{"program_id": ..., "wave": ...,
        "mode": ..., "spend_ref": ...}``) -- used by consumers to warn about
        double counting when the same program also produced a scalar
        parent-level readout.
    """

    channel: str
    breakouts: tuple[str, ...]
    shares: tuple[float, ...]
    log_ratio_cov: tuple[tuple[float, ...], ...] | None = None
    concentration: float | None = None
    distribution: str = "logistic_normal"
    name: str | None = None
    source: dict | None = None

    def __post_init__(self) -> None:
        if not self.channel or not isinstance(self.channel, str):
            raise ValueError(
                f"channel must be a non-empty string, got {self.channel!r}"
            )

        breakouts = tuple(str(b) for b in self.breakouts)
        if len(breakouts) < 2:
            raise ValueError(
                f"breakouts must list at least 2 sub-streams, got {breakouts!r}"
            )
        if len(set(breakouts)) != len(breakouts):
            raise ValueError(f"breakouts contains duplicates: {breakouts!r}")
        object.__setattr__(self, "breakouts", breakouts)

        k = len(breakouts)
        try:
            shares = tuple(float(s) for s in self.shares)
        except (TypeError, ValueError):
            raise ValueError(f"shares must be numeric, got {self.shares!r}") from None
        if len(shares) != k:
            raise ValueError(
                f"shares has {len(shares)} entries but breakouts has {k}; they "
                "must align one-to-one in the same order."
            )
        if any(not math.isfinite(s) or s < 0 for s in shares):
            raise ValueError(f"shares must be finite and non-negative, got {shares!r}")
        floored = tuple(max(s, _EPS) for s in shares)
        total = sum(floored)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=0.01):
            raise ValueError(
                f"shares must sum to ~1 (got {total:.6g}); pass the observed "
                "share composition, not raw responses."
            )
        object.__setattr__(self, "shares", tuple(s / total for s in floored))

        if self.distribution not in ("logistic_normal", "dirichlet"):
            raise ValueError(
                "distribution must be 'logistic_normal' or 'dirichlet', got "
                f"{self.distribution!r}"
            )

        if (self.log_ratio_cov is None) == (self.concentration is None):
            raise ValueError(
                "exactly one of log_ratio_cov (logistic_normal) or concentration "
                "(dirichlet) must be set."
            )
        if self.distribution == "logistic_normal" and self.log_ratio_cov is None:
            raise ValueError("distribution='logistic_normal' requires log_ratio_cov.")
        if self.distribution == "dirichlet" and self.concentration is None:
            raise ValueError("distribution='dirichlet' requires concentration.")

        if self.concentration is not None:
            conc = float(self.concentration)
            if not math.isfinite(conc) or conc <= 0:
                raise ValueError(
                    f"concentration must be a positive finite number, got "
                    f"{self.concentration!r}"
                )
            object.__setattr__(self, "concentration", conc)

        if self.log_ratio_cov is not None:
            import numpy as np

            cov = np.asarray(self.log_ratio_cov, dtype=float)
            if cov.shape != (k - 1, k - 1):
                raise ValueError(
                    f"log_ratio_cov must be ({k - 1}, {k - 1}) -- the ALR "
                    f"covariance w.r.t. the last breakout -- got shape "
                    f"{cov.shape}."
                )
            if not np.all(np.isfinite(cov)):
                raise ValueError("log_ratio_cov must be finite.")
            scale = max(float(np.abs(cov).max()), _EPS)
            if not np.allclose(cov, cov.T, atol=1e-8 * scale):
                raise ValueError("log_ratio_cov must be symmetric.")
            # Strict PD via Cholesky: a merely-PSD (singular) covariance passes
            # an eigenvalue-with-slack check but hands pm.MvNormal a matrix it
            # cannot factor -- surfacing only as a cryptic logp=-inf failure at
            # sampler init, deep inside the (expensive) fit job. Fail here,
            # at construction, with an actionable message instead.
            try:
                np.linalg.cholesky(0.5 * (cov + cov.T))
            except np.linalg.LinAlgError:
                raise ValueError(
                    "log_ratio_cov must be strictly positive definite; add a "
                    "small diagonal ridge (e.g. 1e-9)."
                ) from None
            object.__setattr__(
                self,
                "log_ratio_cov",
                tuple(tuple(float(v) for v in row) for row in cov),
            )

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "breakouts": list(self.breakouts),
            "shares": list(self.shares),
            "log_ratio_cov": (
                [list(row) for row in self.log_ratio_cov]
                if self.log_ratio_cov is not None
                else None
            ),
            "concentration": self.concentration,
            "distribution": self.distribution,
            "name": self.name,
            "source": dict(self.source) if self.source is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShareMeasurement":
        cov = data.get("log_ratio_cov")
        return cls(
            channel=data["channel"],
            breakouts=tuple(data["breakouts"]),
            shares=tuple(data["shares"]),
            log_ratio_cov=(
                tuple(tuple(row) for row in cov) if cov is not None else None
            ),
            concentration=data.get("concentration"),
            distribution=data.get("distribution", "logistic_normal"),
            name=data.get("name"),
            source=data.get("source"),
        )

    def default_node_name(self, index: int) -> str:
        if self.name:
            return self.name
        return f"share_{self.channel}_{index}"


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

# ``build_estimand_expr`` (the in-graph contribution/ROAS/mROAS algebra) now
# lives in mmm_framework.estimands.graph -- the single home of in-graph estimand
# realization -- and is re-exported here so existing callers keep working. The
# graph module imports nothing from mmm_framework at load (it normalizes the
# estimand to its .value), so this is cycle-free.
from ..estimands.graph import build_estimand_expr  # noqa: E402,F401


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


def attach_share_likelihood(
    name: str,
    share_expr: "pt.TensorVariable",
    measurement: ShareMeasurement,
):
    """Add an observed likelihood comparing a model simplex to measured shares.

    Mirrors :func:`attach_experiment_likelihood` for *compositional* evidence:
    ``share_expr`` is the model-implied share vector (a length-``K`` simplex
    tensor, e.g. the breakout model's ``breakout_share_<C>`` Deterministic) and
    ``measurement`` carries the observed shares in the SAME breakout order.
    Call inside the ``pm.Model`` context.

    ``"logistic_normal"`` places an MvNormal on the ``K-1`` additive log-ratios
    (last component as reference) -- matching the simplex's degrees of freedom
    exactly; ``"dirichlet"`` places a Dirichlet with ``a = concentration *
    model_share``. Both clip/floor at a tiny epsilon for numerical safety (the
    model share is strictly interior by construction in the breakout model).

    Parameters
    ----------
    name:
        Name of the observed node (must be unique within the model).
    share_expr:
        PyTensor vector: the model-implied shares, length ``K`` matching
        ``measurement.breakouts`` in order.
    measurement:
        The observed share composition and its measurement-error family.

    Returns
    -------
    The created observed random variable.
    """
    import numpy as np
    import pymc as pm
    import pytensor.tensor as pt

    # Expose the model-implied shares as a named Deterministic for diagnostics
    # ("what the model thinks the shares are" vs what was measured).
    pm.Deterministic(f"{name}_model_share", pt.as_tensor_variable(share_expr))

    shares = np.asarray(measurement.shares, dtype=float)

    if measurement.distribution == "logistic_normal":
        s = pt.clip(share_expr, _EPS, np.inf)
        z_model = pt.log(s[:-1] / s[-1])
        z_hat = np.log(shares[:-1] / shares[-1])
        return pm.MvNormal(
            name,
            mu=z_model,
            cov=np.asarray(measurement.log_ratio_cov, dtype=float),
            observed=z_hat,
        )

    # Dirichlet: a = concentration * model_share (floored to keep a > 0).
    a = float(measurement.concentration) * pt.clip(share_expr, _EPS, np.inf)
    return pm.Dirichlet(name, a=a, observed=shares)


__all__ = [
    "ExperimentEstimand",
    "ExperimentMeasurement",
    "ShareMeasurement",
    "attach_experiment_likelihood",
    "attach_share_likelihood",
    "build_estimand_expr",
    "lognormal_sigma_from_moments",
]
