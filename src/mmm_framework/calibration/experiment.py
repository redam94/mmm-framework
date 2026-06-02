"""Experiment-calibrated priors for the Bayesian MMM.

This module promotes lift / incrementality experiments from a *post-hoc
calibration check* (see :mod:`mmm_framework.validation`) to *informative priors*
on each channel's effect coefficient -- the single most important mechanism for
anchoring an observational MMM to randomized evidence and thereby attacking the
dominant MMM confounder (unobserved demand).

Two-stage workflow
------------------
1. Fit a :class:`~mmm_framework.model.base.BayesianMMM` with default priors.
2. ``ExperimentCalibrator(fitted_model).calibrate(lift_tests)`` derives an
   experiment-anchored prior on each tested channel's coefficient and refits.

The mapping (why it is valid)
-----------------------------
The core model is additive::

    mu = ... + sum_c beta_c * sat_c(adstock_c(x_c)) + ...

Zeroing channel ``c``'s spend sets ``sat_c(adstock_c(0)) = 0``, so the channel's
counterfactual contribution over a period ``P`` is exactly
``beta_c * sum_{t in P} sat_c(adstock_c(x_{c,t}))`` -- **linear in ``beta_c``**.
Define the (data- and shape-dependent) *design factor*

    K_c = E_posterior[ y_std * sum_{t in P} sat_c(adstock_c(x_{c,t})) ]
        = E_posterior[ contribution_c^(s) / beta_c^(s) ]   (computed per draw)

so the model-implied contribution equals ``beta_c * K_c``. A full-holdout lift
test measures that contribution directly as ``measured_lift +/- lift_se``.
Inverting gives an experiment-anchored prior on the coefficient::

    beta_target = measured_lift / K_c
    beta_sigma  = lift_se      / K_c

rendered as a positive ``Gamma`` prior matched in mean and standard deviation.

Assumed semantics (v1) -- read before use
------------------------------------------
* ``measured_lift`` is the channel's **total incremental** KPI over ``test_period``
  (a full holdout / channel-off experiment), **not** the marginal effect of a
  small spend change. The data model (:class:`LiftTestResult` has no spend-delta
  field) matches this interpretation.
* Carryover from the channel is assumed to be contained within ``test_period``
  (or already reflected in ``lift_se``).
* ``K_c`` is computed at the model's aggregation level. If a test specifies
  ``holdout_regions`` while the model pools across geographies, the national
  contribution may not correspond to the geo-restricted lift; the calibrator
  warns (or, with ``strict_geo=True``, refuses).
* ``beta_sigma`` propagates only the experiment's uncertainty; it treats ``K_c``
  (first-stage saturation / adstock shape) as fixed, so the resulting prior is
  marginally tighter than a fully joint treatment would justify.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..config import MFFConfig, PriorConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ..model.base import BayesianMMM
    from ..validation.results import LiftTestResult

logger = logging.getLogger(__name__)

# Floor for posterior beta draws when forming the per-draw ratio contribution/beta.
_MIN_BETA = 1e-6
# Floor for a standard error, to avoid division-by-zero in inverse-variance weights.
_MIN_SE = 1e-12


# =============================================================================
# Pure helpers (no model I/O -- directly unit-testable)
# =============================================================================


def mean_sd_to_gamma(mean: float, sd: float) -> tuple[float, float]:
    """Convert a target mean/sd into ``Gamma(alpha, beta=rate)`` parameters.

    Matches the moments of a Gamma distribution to ``mean`` and ``sd`` so the
    derived prior is centered at ``mean`` with spread ``sd``. ``pm.Gamma``'s
    ``beta`` is the *rate*, which is what is returned.

    Raises
    ------
    ValueError
        If ``mean`` is not strictly positive (a positive-coefficient model
        cannot be anchored at a non-positive effect).
    """
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(f"Gamma mean must be positive and finite, got {mean!r}")
    sd = float(max(sd, 1e-9))
    alpha = (mean / sd) ** 2
    rate = mean / (sd**2)
    return float(alpha), float(rate)


def combine_inverse_variance(
    targets: Sequence[float], ses: Sequence[float]
) -> tuple[float, float]:
    """Inverse-variance (fixed-effect meta-analytic) combination of estimates.

    Parameters
    ----------
    targets, ses
        Per-observation point estimates and their standard errors.

    Returns
    -------
    tuple[float, float]
        Combined ``(mean, sd)``.
    """
    t = np.asarray(targets, dtype=float)
    s = np.asarray(ses, dtype=float)
    if t.size == 0:
        raise ValueError("Cannot combine an empty set of estimates")
    s = np.maximum(s, _MIN_SE)
    w = 1.0 / s**2
    mean = float(np.sum(w * t) / np.sum(w))
    sd = float(np.sqrt(1.0 / np.sum(w)))
    return mean, sd


def design_factor(
    contribution_samples: np.ndarray,
    beta_samples: np.ndarray,
    *,
    min_beta: float = _MIN_BETA,
) -> float:
    """Posterior-mean design factor ``K_c = E[contribution / beta]``.

    Computed *per draw* (then averaged) rather than as a ratio of means, so the
    beta<->saturation posterior covariance does not bias the factor. ``K_c`` is
    the original-scale contribution that one unit of ``beta`` produces over the
    period the samples were drawn for.
    """
    contribution_samples = np.asarray(contribution_samples, dtype=float).reshape(-1)
    beta_samples = np.asarray(beta_samples, dtype=float).reshape(-1)
    if contribution_samples.shape != beta_samples.shape:
        raise ValueError(
            "contribution_samples and beta_samples must have the same length"
        )
    valid = np.abs(beta_samples) > min_beta
    if not valid.any():
        raise ValueError(
            "All posterior beta draws are ~0; cannot derive a design factor."
        )
    return float(np.mean(contribution_samples[valid] / beta_samples[valid]))


# =============================================================================
# Result containers
# =============================================================================


@dataclass(frozen=True)
class LiftObservation:
    """A single lift test reduced to the coefficient scale.

    ``measured_lift`` and ``lift_se`` are the experiment's original-scale values;
    ``design_factor`` is the period-specific ``K_c``. The coefficient-scale
    target is ``measured_lift / design_factor``.
    """

    test_period: tuple[str, str]
    measured_lift: float
    lift_se: float
    design_factor: float
    usable: bool
    note: str = ""

    @property
    def beta_target(self) -> float | None:
        if not self.usable or self.design_factor <= 0:
            return None
        return self.measured_lift / self.design_factor

    @property
    def beta_se(self) -> float | None:
        if not self.usable or self.design_factor <= 0:
            return None
        return self.lift_se / self.design_factor


@dataclass(frozen=True)
class ChannelCalibration:
    """Derived experiment-calibrated prior for one channel."""

    channel: str
    roi_prior: PriorConfig | None
    beta_target: float | None
    beta_sigma: float | None
    beta_fit_mean: float | None
    observations: list[LiftObservation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped_reason: str | None = None

    @property
    def calibrated(self) -> bool:
        return self.roi_prior is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "calibrated": self.calibrated,
            "beta_target": self.beta_target,
            "beta_sigma": self.beta_sigma,
            "beta_fit_mean": self.beta_fit_mean,
            "skipped_reason": self.skipped_reason,
            "notes": list(self.notes),
            "roi_prior": (
                self.roi_prior.model_dump() if self.roi_prior is not None else None
            ),
            "observations": [
                {
                    "test_period": list(o.test_period),
                    "measured_lift": o.measured_lift,
                    "lift_se": o.lift_se,
                    "design_factor": o.design_factor,
                    "usable": o.usable,
                    "note": o.note,
                }
                for o in self.observations
            ],
        }


@dataclass
class CalibrationReport:
    """Per-channel derivation of experiment-calibrated priors."""

    channel_calibrations: list[ChannelCalibration] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def priors(self) -> dict[str, PriorConfig]:
        """Map of channel -> derived ``roi_prior`` (only calibrated channels)."""
        return {
            c.channel: c.roi_prior
            for c in self.channel_calibrations
            if c.roi_prior is not None
        }

    @property
    def calibrated_channels(self) -> list[str]:
        return list(self.priors().keys())

    def summary(self) -> "pd.DataFrame":
        import pandas as pd

        rows = []
        for c in self.channel_calibrations:
            rows.append(
                {
                    "Channel": c.channel,
                    "Calibrated": "Yes" if c.calibrated else "No",
                    "Beta target": (
                        f"{c.beta_target:.4f}" if c.beta_target is not None else "-"
                    ),
                    "Beta sigma": (
                        f"{c.beta_sigma:.4f}" if c.beta_sigma is not None else "-"
                    ),
                    "Prior-fit mean": (
                        f"{c.beta_fit_mean:.4f}" if c.beta_fit_mean is not None else "-"
                    ),
                    "N tests": sum(o.usable for o in c.observations),
                    "Note": c.skipped_reason or "; ".join(c.notes),
                }
            )
        for ch, reason in self.skipped:
            rows.append(
                {
                    "Channel": ch,
                    "Calibrated": "No",
                    "Beta target": "-",
                    "Beta sigma": "-",
                    "Prior-fit mean": "-",
                    "N tests": 0,
                    "Note": reason,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_calibrations": [c.to_dict() for c in self.channel_calibrations],
            "skipped": [{"channel": ch, "reason": r} for ch, r in self.skipped],
            "calibrated_channels": self.calibrated_channels,
        }


@dataclass
class CalibrationOutcome:
    """Result of a calibration run, optionally including the refit model."""

    report: CalibrationReport
    config: MFFConfig
    model: "BayesianMMM | None" = None
    results: Any | None = None


# =============================================================================
# Channel-level derivation (pure given the reduced observations)
# =============================================================================


def derive_channel_prior(
    channel: str,
    observations: Sequence[LiftObservation],
) -> ChannelCalibration:
    """Derive a channel's coefficient prior from reduced lift observations.

    Pure: takes pre-computed :class:`LiftObservation` records (each carrying its
    own period-specific ``design_factor``) and returns the combined prior. Lift
    tests with non-positive measured lift or design factor are excluded with a
    note (a positive-coefficient model cannot be anchored to them).
    """
    observations = list(observations)
    notes: list[str] = []
    targets: list[float] = []
    ses: list[float] = []
    beta_fit_mean: float | None = None

    for obs in observations:
        if obs.note:
            notes.append(f"{obs.test_period}: {obs.note}")
        target = obs.beta_target
        se = obs.beta_se
        if target is None or se is None:
            continue
        if target <= 0:
            notes.append(
                f"{obs.test_period}: coefficient target {target:.4g} <= 0 excluded"
            )
            continue
        targets.append(target)
        ses.append(se)

    if not targets:
        return ChannelCalibration(
            channel=channel,
            roi_prior=None,
            beta_target=None,
            beta_sigma=None,
            beta_fit_mean=beta_fit_mean,
            observations=observations,
            notes=notes,
            skipped_reason="no usable (positive-lift, positive-design) tests",
        )

    beta_target, beta_sigma = combine_inverse_variance(targets, ses)
    try:
        alpha, rate = mean_sd_to_gamma(beta_target, beta_sigma)
    except ValueError as exc:  # pragma: no cover - guarded above, defensive
        return ChannelCalibration(
            channel=channel,
            roi_prior=None,
            beta_target=beta_target,
            beta_sigma=beta_sigma,
            beta_fit_mean=beta_fit_mean,
            observations=observations,
            notes=notes + [str(exc)],
            skipped_reason=str(exc),
        )

    roi_prior = PriorConfig.gamma(alpha=alpha, beta=rate)
    return ChannelCalibration(
        channel=channel,
        roi_prior=roi_prior,
        beta_target=beta_target,
        beta_sigma=beta_sigma,
        beta_fit_mean=beta_fit_mean,
        observations=observations,
        notes=notes,
    )


# =============================================================================
# Calibrator (model I/O + orchestration)
# =============================================================================


class ExperimentCalibrator:
    """Turn lift / incrementality experiments into informative channel priors.

    Parameters
    ----------
    model : BayesianMMM
        A **fitted** model (its posterior supplies the per-channel design
        factor). Call :meth:`~mmm_framework.model.base.BayesianMMM.fit` first.
    results : optional
        Fit results container (passed through to the internal validator helper).
    """

    def __init__(self, model: "BayesianMMM", results: Any | None = None):
        if getattr(model, "_trace", None) is None:
            raise ValueError(
                "ExperimentCalibrator requires a fitted model. Call model.fit() "
                "before deriving experiment-calibrated priors."
            )
        self.model = model
        self.results = results
        self._validator = None  # lazily constructed ModelValidator for period parsing

    # -- extraction -------------------------------------------------------

    def _period_indices(self, test_period: tuple[str, str]) -> tuple[int, int]:
        from ..validation.validator import ModelValidator

        if self._validator is None:
            self._validator = ModelValidator(self.model, self.results)
        return self._validator._parse_period_to_indices(test_period)

    def _contribution_and_beta_samples(
        self, channel: str, test_period: tuple[str, str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-draw (original-scale contribution, coefficient) over ``test_period``."""
        model = self.model
        posterior = model._trace.posterior

        if channel not in model.channel_names:
            raise ValueError(f"Unknown channel: {channel!r}")
        ch_idx = model.channel_names.index(channel)

        start_idx, end_idx = self._period_indices(test_period)
        mask = np.asarray(model._get_time_mask((start_idx, end_idx)))

        if "channel_contributions" not in posterior:
            raise KeyError(
                "Posterior is missing 'channel_contributions'; cannot calibrate."
            )
        cc = posterior["channel_contributions"]
        if "channel" in cc.dims:
            cc = cc.isel(channel=ch_idx)
        arr = np.asarray(cc.values)  # (chain, draw, obs)
        contrib = arr[:, :, mask].sum(axis=-1) * float(model.y_std)
        contrib_samples = contrib.reshape(-1)

        beta_var = f"beta_{channel}"
        if beta_var not in posterior:
            raise KeyError(
                f"Posterior is missing '{beta_var}'; cannot derive a design factor."
            )
        beta_samples = np.asarray(posterior[beta_var].values).reshape(-1)
        return contrib_samples, beta_samples

    def _geo_warning(
        self, tests: Sequence["LiftTestResult"], strict: bool
    ) -> str | None:
        model = self.model
        has_geo = bool(getattr(model, "has_geo", False))
        pooled = bool(
            getattr(
                getattr(model, "hierarchical_config", None), "pool_across_geo", False
            )
        )
        any_holdout = any(getattr(t, "holdout_regions", None) for t in tests)
        if any_holdout and has_geo and pooled:
            msg = (
                "lift test specifies holdout_regions but the model pools across "
                "geographies; the model contribution is national/pooled and may not "
                "correspond to the geo-restricted measured lift -- derived prior is "
                "approximate"
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=3)
            return msg
        return None

    # -- public API -------------------------------------------------------

    def derive_priors(
        self,
        lift_tests: Sequence["LiftTestResult"],
        *,
        strict_geo: bool = False,
    ) -> CalibrationReport:
        """Derive experiment-calibrated priors without refitting."""
        from collections import defaultdict

        by_channel: dict[str, list["LiftTestResult"]] = defaultdict(list)
        for lt in lift_tests:
            by_channel[lt.channel].append(lt)

        report = CalibrationReport()
        for channel, tests in by_channel.items():
            if channel not in self.model.channel_names:
                report.skipped.append((channel, "unknown channel"))
                continue

            geo_msg = self._geo_warning(tests, strict_geo)

            observations: list[LiftObservation] = []
            beta_fit_mean: float | None = None
            for lt in tests:
                try:
                    contrib, beta = self._contribution_and_beta_samples(
                        channel, lt.test_period
                    )
                except (KeyError, ValueError) as exc:
                    observations.append(
                        LiftObservation(
                            test_period=tuple(lt.test_period),
                            measured_lift=float(lt.measured_lift),
                            lift_se=float(lt.lift_se),
                            design_factor=0.0,
                            usable=False,
                            note=f"extraction failed: {exc}",
                        )
                    )
                    continue

                if beta_fit_mean is None:
                    beta_fit_mean = float(np.mean(beta))

                try:
                    k_c = design_factor(contrib, beta)
                except ValueError as exc:
                    observations.append(
                        LiftObservation(
                            test_period=tuple(lt.test_period),
                            measured_lift=float(lt.measured_lift),
                            lift_se=float(lt.lift_se),
                            design_factor=0.0,
                            usable=False,
                            note=str(exc),
                        )
                    )
                    continue

                note = ""
                usable = True
                if k_c <= 0:
                    usable = False
                    note = f"non-positive design factor ({k_c:.4g})"
                elif float(lt.measured_lift) <= 0:
                    usable = False
                    note = f"non-positive measured lift ({lt.measured_lift})"
                elif float(lt.lift_se) < 0:
                    usable = False
                    note = "negative lift_se"

                observations.append(
                    LiftObservation(
                        test_period=tuple(lt.test_period),
                        measured_lift=float(lt.measured_lift),
                        lift_se=float(lt.lift_se),
                        design_factor=float(k_c),
                        usable=usable,
                        note=note,
                    )
                )

            channel_cal = derive_channel_prior(channel, observations)
            # attach the posterior-fit mean and any geo caveat
            notes = list(channel_cal.notes)
            if geo_msg:
                notes.append(geo_msg)
            channel_cal = ChannelCalibration(
                channel=channel_cal.channel,
                roi_prior=channel_cal.roi_prior,
                beta_target=channel_cal.beta_target,
                beta_sigma=channel_cal.beta_sigma,
                beta_fit_mean=beta_fit_mean,
                observations=channel_cal.observations,
                notes=notes,
                skipped_reason=channel_cal.skipped_reason,
            )
            report.channel_calibrations.append(channel_cal)

        return report

    def calibrated_config(self, report: CalibrationReport) -> MFFConfig:
        """Deep-copy the model's MFFConfig with derived ``roi_prior`` applied."""
        priors = report.priors()
        new_config = self.model.mff_config.model_copy(deep=True)
        for media in new_config.media_channels:
            if media.name in priors:
                media.roi_prior = priors[media.name]
        return new_config

    def _clone_panel_with_config(self, config: MFFConfig) -> Any:
        from ..data_loader import PanelDataset

        panel = self.model.panel
        return PanelDataset(
            y=panel.y,
            X_media=panel.X_media,
            X_controls=panel.X_controls,
            index=panel.index,
            config=config,
            coords=panel.coords,
        )

    def calibrate(
        self,
        lift_tests: Sequence["LiftTestResult"],
        *,
        refit: bool = True,
        draws: int | None = None,
        tune: int | None = None,
        chains: int | None = None,
        random_seed: int | None = None,
        strict_geo: bool = False,
    ) -> CalibrationOutcome:
        """Derive experiment-calibrated priors and (optionally) refit.

        Returns a :class:`CalibrationOutcome` with the derivation ``report``, the
        ``config`` carrying the new priors, and -- when ``refit`` -- a freshly
        fitted ``model`` and its ``results``.
        """
        report = self.derive_priors(lift_tests, strict_geo=strict_geo)
        config = self.calibrated_config(report)

        if not refit or not report.calibrated_channels:
            if refit and not report.calibrated_channels:
                logger.warning(
                    "No channels were calibrated; returning without refitting."
                )
            return CalibrationOutcome(report=report, config=config)

        from ..model.base import BayesianMMM

        new_panel = self._clone_panel_with_config(config)
        new_model = BayesianMMM(
            panel=new_panel,
            model_config=self.model.model_config,
            trend_config=self.model.trend_config,
            adstock_alphas=self.model.adstock_alphas,
        )

        fit_kwargs: dict[str, Any] = {}
        if draws is not None:
            fit_kwargs["draws"] = draws
        if tune is not None:
            fit_kwargs["tune"] = tune
        if chains is not None:
            fit_kwargs["chains"] = chains
        if random_seed is not None:
            fit_kwargs["random_seed"] = random_seed

        logger.info(
            "Refitting with experiment-calibrated priors on %s",
            report.calibrated_channels,
        )
        new_results = new_model.fit(**fit_kwargs)
        return CalibrationOutcome(
            report=report, config=config, model=new_model, results=new_results
        )


def calibrate_with_experiments(
    model: "BayesianMMM",
    lift_tests: Sequence["LiftTestResult"],
    *,
    refit: bool = True,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    random_seed: int | None = None,
    strict_geo: bool = False,
) -> CalibrationOutcome:
    """Convenience wrapper: derive experiment-calibrated priors and refit.

    Examples
    --------
    >>> from mmm_framework.calibration import calibrate_with_experiments
    >>> from mmm_framework.validation import LiftTestResult
    >>> base = BayesianMMM(panel, model_config); base.fit()  # doctest: +SKIP
    >>> tests = [LiftTestResult("TV", ("2023-01-01", "2023-03-31"), 1.2e5, 2e4)]
    >>> outcome = calibrate_with_experiments(base, tests)     # doctest: +SKIP
    >>> outcome.model  # the experiment-anchored refit                # doctest: +SKIP
    """
    return ExperimentCalibrator(model).calibrate(
        lift_tests,
        refit=refit,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        strict_geo=strict_geo,
    )
