"""
Base class for extended MMM models.

Provides common functionality for NestedMMM, MultivariateMMM, and CombinedMMM.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd
import pymc as pm

if TYPE_CHECKING:
    import arviz as az
    import pytensor.tensor as pt

    from ...calibration.likelihood import ExperimentMeasurement

from ..results import ModelResults

# Adstock kernel length (lags) for the extension models' media transform.
_ADSTOCK_L_MAX = 8


class BaseExtendedMMM:
    """Base class for extended MMM models.

    Provides common functionality for all extended model types:
    - Data storage and validation
    - Model building lifecycle
    - MCMC fitting
    - Result extraction

    Subclasses must implement:
    - _build_coords(): Return PyMC coordinate dict
    - _build_model(): Return built PyMC model
    """

    def __init__(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        index: pd.Index | None = None,
    ):
        """
        Initialize the base model.

        Parameters
        ----------
        X_media : np.ndarray
            Media variable matrix (n_obs, n_channels)
        y : np.ndarray
            Target variable (n_obs,)
        channel_names : list[str]
            Names of media channels
        index : pd.Index | None
            Optional time index for the data
        """
        self.X_media = X_media
        self.y = y
        self.channel_names = channel_names
        self.index = index if index is not None else pd.RangeIndex(len(y))

        self.n_obs = len(y)
        self.n_channels = len(channel_names)

        # Outcome standardization. ``self.y`` stays on the caller's raw scale;
        # subclasses fit the likelihood on ``(y - y_mean) / y_std`` so the fixed
        # effect/noise priors (Normal(0, ~0.5-2)) are well-calibrated regardless
        # of the KPI's units. Report-consumed deterministics are registered back
        # in original units.
        y_arr = np.asarray(y, dtype=float)
        self.y_mean = float(y_arr.mean())
        self.y_std = float(y_arr.std()) + 1e-8

        # Per-channel scale (raw spend max) used to normalize media before the
        # adstock+saturation transform so the logistic curve operates in a
        # meaningful range rather than the flat (~1) tail.
        self._media_scale = np.asarray(self.X_media, dtype=float).max(axis=0) + 1e-8

        # Experiment likelihood terms (set via add_experiment_calibration()).
        self.experiments: list["ExperimentMeasurement"] = []

        self._model: pm.Model | None = None
        self._trace: az.InferenceData | None = None

    # -- pickling: keep the (unpicklable) PyMC graph / trace out of state ------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # The built graph holds PyTensor objects and the trace is large; both are
        # rebuilt / reattached on load (the trace is saved separately by save()).
        state["_model"] = None
        state["_trace"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._model = None
        self._trace = None

    def _build_coords(self) -> dict:
        """Build PyMC coordinates. Override in subclasses."""
        return {
            "obs": np.arange(self.n_obs),
            "channel": self.channel_names,
        }

    def _build_model(self) -> pm.Model:
        """Build the PyMC model. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _build_model")

    @property
    def model(self) -> pm.Model:
        """Get or build the PyMC model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    # =====================================================================
    # Media transform (shared)
    # =====================================================================

    def _media_transform_apply(
        self,
        channel_idx: int,
        alpha: "pt.TensorVariable",
        lam: "pt.TensorVariable",
    ) -> "Callable[[pt.TensorVariable], pt.TensorVariable]":
        """Return the per-channel media transform: normalize -> adstock -> saturate.

        Normalizing the raw spend by its per-channel max (a data-fixed constant)
        keeps the input in roughly ``[0, 1]`` so the geometric adstock (with
        ``normalize=True``, hence scale-preserving) and the logistic saturation
        operate with real curvature instead of saturating immediately. The
        returned closure reuses the channel's ``alpha`` (carryover) and ``lam``
        (saturation) RVs, so it can transform both the observed spend and a
        perturbed (experiment-scaled) spend with shared parameters.
        """
        import pytensor.tensor as pt

        from ...transforms.adstock_pt import parametric_adstock_pt
        from ..components.transforms import logistic_saturation_pt

        scale = float(self._media_scale[channel_idx])
        # Geometric weights are ``alpha ** lag``; the gradient of the ``lag == 0``
        # term is ``0 * alpha ** -1``, singular at alpha == 0. Clip alpha off the
        # boundary so the carryover gradient stays finite (Beta(2,2) sampling
        # lives strictly inside (0, 1), so this does not change the model).
        alpha_safe = pt.clip(alpha, 1e-6, 1.0 - 1e-6)

        def apply(x: "pt.TensorVariable") -> "pt.TensorVariable":
            x_adstocked = parametric_adstock_pt(
                x / scale, "geometric", _ADSTOCK_L_MAX, alpha=alpha_safe, normalize=True
            )
            return logistic_saturation_pt(x_adstocked, lam)

        return apply

    # =====================================================================
    # Serialization (save/load the model + trace + experiments)
    # =====================================================================

    def save(self, path: str | Path) -> None:
        """Persist the model (config, data, experiments) and its fitted trace.

        The PyMC graph is dropped (rebuilt on demand) and the trace is written
        separately as NetCDF; everything else -- including any registered
        :class:`ExperimentMeasurement` calibrations -- is pickled, so a reloaded
        model can be inspected, predicted from, or re-fit with the same
        experiment anchoring.
        """
        import cloudpickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            cloudpickle.dump(self, f)
        if self._trace is not None:
            self._trace.to_netcdf(str(path / "trace.nc"))

    @classmethod
    def load(cls, path: str | Path) -> "BaseExtendedMMM":
        """Load a model saved with :meth:`save`, reattaching its trace."""
        import cloudpickle

        path = Path(path)
        with open(path / "model.pkl", "rb") as f:
            obj = cloudpickle.load(f)
        trace_path = path / "trace.nc"
        if trace_path.exists():
            import arviz as az

            obj._trace = az.from_netcdf(str(trace_path))
        return obj

    # =====================================================================
    # Experiment (incrementality / lift / ROAS) calibration likelihoods
    # =====================================================================

    def add_experiment_calibration(
        self, experiments: "Sequence[ExperimentMeasurement]"
    ) -> "BaseExtendedMMM":
        """Register experiment likelihood terms and invalidate the built graph.

        The experiments are folded into the model as likelihood terms on the
        model-implied estimand (contribution / ROAS / marginal ROAS) the next
        time the graph is built, so call this before :meth:`fit`. Returns ``self``
        for chaining.

        Multi-outcome models (:class:`MultivariateMMM`, :class:`CombinedMMM`)
        require each measurement's ``outcome`` to be set.
        """
        self.experiments = list(experiments)
        self._model = None  # force a rebuild that includes the new likelihoods
        return self

    def _experiment_period_mask(
        self, exp: "ExperimentMeasurement"
    ) -> np.ndarray | None:
        """Boolean obs mask for an experiment window over ``self.index``.

        Extension models are single time series (one observation per period, no
        geo dimension), so the window selects observations directly. Returns
        ``None`` (with a warning) when the window is out of range or specifies
        geo holdout regions the model cannot represent.
        """
        if exp.holdout_regions:
            warnings.warn(
                f"Experiment on {exp.channel!r} skipped: holdout_regions "
                f"{exp.holdout_regions} require a geo model, which the extension "
                "models do not support.",
                stacklevel=3,
            )
            return None

        start, end = exp.test_period
        if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)):
            positions = np.arange(self.n_obs)
            mask = (positions >= int(start)) & (positions <= int(end))
            return mask if mask.any() else None

        try:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            index_dt = pd.to_datetime(self.index)
        except (ValueError, TypeError):
            warnings.warn(
                f"Experiment on {exp.channel!r} skipped: cannot parse period "
                f"{exp.test_period!r}.",
                stacklevel=3,
            )
            return None

        mask = np.asarray((index_dt >= start_date) & (index_dt <= end_date))
        if not mask.any():
            warnings.warn(
                f"Experiment on {exp.channel!r} skipped: window "
                f"{exp.test_period!r} falls outside the data's period range.",
                stacklevel=3,
            )
            return None
        return mask

    def _add_experiment_likelihoods(self, handles: dict, *, scale: float = 1.0) -> None:
        """Fold registered experiments into the graph as likelihood terms.

        Called inside the ``pm.Model`` context of a subclass ``_build_model``.
        ``handles`` maps a channel (or ``(channel, outcome)`` for multi-outcome
        models) to ``{"coef", "x_sat", "apply", "x_input", "spend_obs"}`` where
        the per-obs contribution is ``coef * x_sat``, ``apply`` re-runs the
        channel's media transform (for the marginal-ROAS perturbation),
        ``x_input`` is the channel's raw spend tensor, and ``spend_obs`` is its
        raw spend series. ``scale`` converts the contribution to the estimand's
        natural (original-KPI) scale — the outcome's ``y_std`` when the model
        fits standardized outcomes. A handle may carry its own ``"scale"`` entry
        (multi-outcome models, where each outcome has its own std), which
        overrides the argument.
        """
        import pytensor.tensor as pt

        from ...calibration.likelihood import (
            ExperimentEstimand,
            attach_experiment_likelihood,
            build_estimand_expr,
        )

        used_names: set[str] = set()
        for i, exp in enumerate(self.experiments):
            key = (exp.channel, exp.outcome) if exp.outcome else exp.channel
            handle = handles.get(key)
            if handle is None:
                label = f"{exp.channel!r}" + (
                    f"/outcome {exp.outcome!r}" if exp.outcome else ""
                )
                warnings.warn(
                    f"Experiment skipped: no model handle for channel {label}.",
                    stacklevel=2,
                )
                continue

            mask = self._experiment_period_mask(exp)
            if mask is None:
                continue
            mask_idx = np.flatnonzero(mask)

            handle_scale = float(handle.get("scale", scale))
            coef = handle["coef"]
            contrib_window = (coef * handle["x_sat"])[mask_idx].sum()

            if exp.spend is not None:
                spend_window = float(exp.spend)
            else:
                spend_window = float(np.asarray(handle["spend_obs"])[mask_idx].sum())

            if exp.estimand is ExperimentEstimand.MROAS:
                if spend_window <= 0:
                    warnings.warn(
                        f"mROAS experiment on {exp.channel!r} skipped: window "
                        "spend is zero.",
                        stacklevel=2,
                    )
                    continue
                lift = exp.spend_lift_pct / 100.0
                pert_mult = np.ones(self.n_obs, dtype=np.float64)
                pert_mult[mask] = 1.0 + lift
                # Re-run the full media transform (normalize -> adstock ->
                # saturate) at the perturbed spend, reusing the channel's RVs.
                x_sat_pert = handle["apply"](
                    handle["x_input"] * pt.as_tensor_variable(pert_mult)
                )
                contrib_pert_window = (coef * x_sat_pert)[mask_idx].sum()
                estimand = build_estimand_expr(
                    ExperimentEstimand.MROAS,
                    contrib_window=contrib_window,
                    spend_window=spend_window,
                    scale=handle_scale,
                    contrib_window_pert=contrib_pert_window,
                    lift=lift,
                )
            elif exp.estimand is ExperimentEstimand.ROAS:
                if spend_window <= 0:
                    warnings.warn(
                        f"ROAS experiment on {exp.channel!r} skipped: window "
                        "spend is zero.",
                        stacklevel=2,
                    )
                    continue
                estimand = build_estimand_expr(
                    ExperimentEstimand.ROAS,
                    contrib_window=contrib_window,
                    spend_window=spend_window,
                    scale=handle_scale,
                )
            else:
                estimand = build_estimand_expr(
                    ExperimentEstimand.CONTRIBUTION,
                    contrib_window=contrib_window,
                    spend_window=spend_window,
                    scale=handle_scale,
                )

            base_name = exp.default_node_name(i)
            name = base_name
            bump = 2
            while name in used_names or f"{name}_model_estimand" in used_names:
                name = f"{base_name}_{bump}"
                bump += 1
            used_names.add(name)
            used_names.add(f"{name}_model_estimand")
            attach_experiment_likelihood(name, estimand, exp)

    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        nuts_sampler: str = "pymc",
        **kwargs,
    ) -> ModelResults:
        """
        Fit the model using MCMC.

        Args:
            draws: Number of posterior draws per chain.
            tune: Number of tuning iterations.
            chains: Number of MCMC chains.
            target_accept: Target acceptance rate for NUTS.
            random_seed: Random seed for reproducibility.
            nuts_sampler: NUTS sampler to use ("pymc", "numpyro", "nutpie").
            **kwargs: Additional arguments passed to pm.sample.

        Returns:
            Container with trace and model.
        """
        with self.model:
            self._trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                **kwargs,
            )
        return ModelResults(
            trace=self._trace,
            model=self.model,
            config=getattr(self, "config", None),
        )

    def _check_fitted(self):
        """Check that model has been fitted."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

    @property
    def trace(self) -> az.InferenceData:
        """Get the fitted trace."""
        self._check_fitted()
        return self._trace

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary statistics."""
        import arviz as az

        self._check_fitted()
        return az.summary(self._trace, var_names=var_names)

    def sample_prior_predictive(
        self, samples: int = 500, random_seed: int | None = None
    ) -> az.InferenceData:
        """Sample from the prior predictive distribution."""
        import pymc as pm

        with self.model:
            return pm.sample_prior_predictive(samples=samples, random_seed=random_seed)

    def compute_parameter_learning(
        self,
        var_names: list[str] | None = None,
        *,
        prior_samples: int = 1000,
        random_seed: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Quantify how much the data updated each parameter relative to its prior.

        Draws prior samples and compares them to the fitted posterior, returning the
        prior-to-posterior **contraction**, **overlap**, and location **shift** for every
        parameter. Particularly useful for sign-constrained effects such as a
        cannibalization cross-effect (``psi = -HalfNormal``), where a posterior "entirely
        below zero" can simply restate the prior: contraction/overlap reveal whether the
        data actually learned it. See
        :func:`mmm_framework.diagnostics.parameter_learning`.

        Parameters
        ----------
        var_names:
            Parameters to diagnose. ``None`` (default) uses the model's free random
            variables (which include the ``psi_*_raw`` cross-effect magnitudes).
        prior_samples:
            Number of prior draws used to estimate the prior moments/overlap.
        random_seed:
            Seed for the prior draw (reproducibility).
        **kwargs:
            Forwarded to :func:`~mmm_framework.diagnostics.parameter_learning`.

        Returns
        -------
        pandas.DataFrame
            One row per parameter, sorted by ``contraction`` ascending (most
            prior-dominated first).
        """
        from ...diagnostics import parameter_learning

        self._check_fitted()
        prior = self.sample_prior_predictive(
            samples=prior_samples, random_seed=random_seed
        )
        if var_names is None:
            var_names = [rv.name for rv in self.model.free_RVs]
        return parameter_learning(prior, self._trace, var_names=var_names, **kwargs)


__all__ = ["BaseExtendedMMM"]
