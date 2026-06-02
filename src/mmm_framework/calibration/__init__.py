"""Experiment calibration for the MMM framework.

Anchors the observational model to randomized incrementality evidence via two
complementary routes:

* **Priors** (:class:`ExperimentCalibrator`) -- a two-stage fit -> derive an
  experiment-anchored coefficient prior -> refit. Simple and robust; encodes a
  *contribution* only, holding the saturation/adstock shape fixed.
* **Likelihood** (:class:`ExperimentMeasurement` +
  :meth:`BayesianMMM.add_experiment_calibration`) -- folds the experiment into
  the PyMC graph as a likelihood term on the model-implied *estimand*
  (contribution, ROAS, or marginal ROAS), updating ``beta``, the s-curve, and
  the adstock kernel jointly. More general; this is the route to use when the
  experiment is reported in ROAS / mROAS terms.

Examples
--------
Prior route::

>>> from mmm_framework.calibration import ExperimentCalibrator
>>> calibrator = ExperimentCalibrator(fitted_model)        # doctest: +SKIP
>>> outcome = calibrator.calibrate(lift_tests)             # doctest: +SKIP

Likelihood route::

>>> from mmm_framework.calibration import (
...     ExperimentMeasurement, ExperimentEstimand)
>>> exp = ExperimentMeasurement(                            # doctest: +SKIP
...     "TV", ("2023-01-01", "2023-03-31"), value=2.5, se=0.4,
...     estimand=ExperimentEstimand.ROAS)
>>> model.add_experiment_calibration([exp]).fit()          # doctest: +SKIP
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .experiment import (
    CalibrationOutcome,
    CalibrationReport,
    ChannelCalibration,
    ExperimentCalibrator,
    LiftObservation,
    calibrate_with_experiments,
    combine_inverse_variance,
    derive_channel_prior,
    design_factor,
    mean_sd_to_gamma,
)
from .likelihood import (
    ExperimentEstimand,
    ExperimentMeasurement,
    attach_experiment_likelihood,
    build_estimand_expr,
    lognormal_sigma_from_moments,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..validation.results import LiftTestResult

__all__ = [
    "ExperimentCalibrator",
    "calibrate_with_experiments",
    "CalibrationOutcome",
    "CalibrationReport",
    "ChannelCalibration",
    "LiftObservation",
    "derive_channel_prior",
    "design_factor",
    "combine_inverse_variance",
    "mean_sd_to_gamma",
    "LiftTestResult",
    # Likelihood-based (in-graph) experiment calibration
    "ExperimentMeasurement",
    "ExperimentEstimand",
    "attach_experiment_likelihood",
    "build_estimand_expr",
    "lognormal_sigma_from_moments",
]


def __getattr__(name: str):
    # Lazily re-export LiftTestResult so importing this package does not eagerly
    # pull in the (heavier) validation package.
    if name == "LiftTestResult":
        from ..validation.results import LiftTestResult

        return LiftTestResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
