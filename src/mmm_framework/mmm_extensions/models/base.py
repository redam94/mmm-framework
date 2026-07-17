"""
Base class for extended MMM models.

Provides common functionality for NestedMMM, MultivariateMMM, and CombinedMMM.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import pandas as pd
import pymc as pm

if TYPE_CHECKING:
    import arviz as az
    import pytensor.tensor as pt

    from ...calibration.likelihood import ExperimentMeasurement
    from ...config.enums import FitMethod

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

    # Saved-model format version for the extension family (read by
    # MMMSerializer's extended branch; the core BayesianMMM has its own).
    _VERSION = "1.0"

    def __init__(
        self,
        X_media: np.ndarray,
        y: np.ndarray,
        channel_names: list[str],
        index: pd.Index | None = None,
        model_config: Any | None = None,
        trend_config: Any | None = None,
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
        model_config : ModelConfig | None
            Optional core model configuration carrying the seasonality settings
            and the outcome likelihood family. When ``None`` the extension keeps
            its historical hard-coded baseline (no seasonality, Normal outcome) —
            so a directly-constructed model is byte-identical to before.
        trend_config : TrendConfig | None
            Optional trend configuration. ``None`` or type ``none`` → no trend
            term (historical behavior); ``linear`` adds a standardized-scale
            trend so a real drift does not contaminate the media coefficients.
        """
        self.X_media = X_media
        self.y = y
        self.channel_names = channel_names
        self.index = index if index is not None else pd.RangeIndex(len(y))
        # Optional core config — carries seasonality + outcome likelihood; None
        # keeps the historical hard-coded baseline (see docstring).
        self.model_config = model_config
        self.trend_config = trend_config

        # The multiplicative (semi-log) specification is a core-BayesianMMM feature;
        # the extension families (nested / multivariate / combined / structural)
        # build their own additive graphs and do not implement a log link.
        from ...config import ModelSpecification as _ModelSpec

        if (
            getattr(model_config, "specification", _ModelSpec.ADDITIVE)
            == _ModelSpec.MULTIPLICATIVE
        ):
            raise NotImplementedError(
                "Multiplicative (semi-log) specification is not supported for "
                f"extension models ({type(self).__name__}); it is available on the "
                "core BayesianMMM. Use the additive form for this model family."
            )

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

    def _build_baseline_dynamics(
        self,
        outcomes: Sequence,
        share_trend: bool,
        share_seasonality: bool,
    ) -> tuple[dict, dict]:
        """Build per-outcome trend + seasonality terms (standardized scale) for
        the multi-outcome models.

        Returns ``(trend_terms, seasonality_terms)`` — dicts keyed by outcome
        index, value None when that component is off for the outcome. A trend
        (resp. seasonality) is added for outcome ``k`` only when a trend_config
        (resp. a seasonality config) is present AND the outcome's
        ``include_trend`` (resp. ``include_seasonality``) is True. When the
        corresponding ``share_*`` flag is set, one RV is shared across outcomes;
        otherwise each outcome gets its own (name-prefixed) RV. Must be called
        inside the model context (it creates RVs).
        """
        from ..components.temporal import (
            build_seasonality_contribution,
            build_trend_contribution,
        )

        trend_terms: dict = {}
        seasonality_terms: dict = {}
        season_cfg = getattr(self.model_config, "seasonality", None)

        shared_trend = (
            build_trend_contribution("", self.n_obs, self.trend_config)
            if (self.trend_config is not None and share_trend)
            else None
        )
        shared_season = (
            build_seasonality_contribution("", self.index, self.n_obs, season_cfg)
            if (season_cfg is not None and share_seasonality)
            else None
        )
        for k, oc in enumerate(outcomes):
            if self.trend_config is not None and getattr(oc, "include_trend", True):
                trend_terms[k] = (
                    shared_trend
                    if share_trend
                    else build_trend_contribution(
                        f"{oc.name}_", self.n_obs, self.trend_config
                    )
                )
            if season_cfg is not None and getattr(oc, "include_seasonality", True):
                seasonality_terms[k] = (
                    shared_season
                    if share_seasonality
                    else build_seasonality_contribution(
                        f"{oc.name}_", self.index, self.n_obs, season_cfg
                    )
                )
        return trend_terms, seasonality_terms

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
        """Load a model saved with :meth:`save`, reattaching its trace.

        Also reads :class:`~mmm_framework.serialization.MMMSerializer`'s
        extended-flavor saves (which gzip the trace to ``trace.nc.gz`` by
        default) — the trace loader handles both layouts, so a cross-flavor
        load never silently drops the posterior.
        """
        import cloudpickle

        path = Path(path)
        with open(path / "model.pkl", "rb") as f:
            obj = cloudpickle.load(f)
        trace_path = path / "trace.nc"
        if trace_path.exists():
            import arviz as az

            obj._trace = az.from_netcdf(str(trace_path))
        elif (path / "trace.nc.gz").exists():
            from ...serialization import MMMSerializer

            obj._trace = MMMSerializer._load_trace(path)
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
        method: "FitMethod | str" = "nuts",
        **kwargs,
    ) -> ModelResults:
        """Fit the model with NUTS (default), SMC, or a fast **approximate** method.

        ``method`` ∈ {``nuts``, ``smc``, ``map``, ``laplace``, ``advi``,
        ``fullrank_advi``, ``pathfinder``}. NUTS is full MCMC for real
        inference. ``smc`` is tempered Sequential Monte Carlo — also **exact**
        (``ModelResults.approximate`` stays ``False``): slower than NUTS but
        robust to the multimodal geometries extension models are prone to
        (reflected factor modes, label switching, adstock↔AR ridges) and it
        estimates the log marginal likelihood for model comparison. The
        approximate methods fit in **seconds** for quick model checks — the
        returned ``ModelResults.approximate`` is ``True``, R-hat/ESS are
        ``None`` and the uncertainty is **not calibrated** (re-fit with NUTS
        before trusting intervals/decisions). The approximate posterior is a
        drop-in for the NUTS trace (deterministics included), so the extended
        reports and pathway analysis work off it. Extension models share the
        base model's engines (:func:`~mmm_framework.model.base.run_approximate_fit`,
        :func:`~mmm_framework.model.base.run_smc_fit`).

        Args:
            draws: Posterior draws per chain (NUTS), particles per SMC run,
                or number of approximate draws (ADVI/Laplace/Pathfinder; MAP
                is a single point and ignores it).
            tune / target_accept / nuts_sampler: NUTS controls (unused by SMC
                and the approximate methods). ``chains`` also sets the number
                of independent SMC runs (R-hat is computed across them).
            random_seed: Random seed for reproducibility.
            method: Inference method (see above).
            **kwargs: forwarded to ``pm.sample`` (NUTS), ``pm.sample_smc``
                (SMC), or the approximate fitter.

        Returns:
            Container with trace, model and diagnostics.
        """
        from ...config.enums import FitMethod

        fit_method = (
            method if isinstance(method, FitMethod) else FitMethod(str(method).lower())
        )

        if fit_method is FitMethod.NUTS:
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

            # Compute + stamp convergence diagnostics and WARN on non-convergence.
            # Extended models (Nested/Multivariate/Combined) previously recorded
            # NO diagnostics -- the most divergence-prone geometries shipped with
            # no convergence signal. Best-effort: never let diagnostics fail a fit.
            diagnostics: dict = {"approximate": False, "fit_method": "nuts"}
            try:
                from ...diagnostics import convergence as _conv

                diagnostics.update(_conv.compute_convergence(self._trace))
                _conv.annotate(diagnostics)
                _conv.warn_if_not_converged(diagnostics, label=type(self).__name__)
            except Exception:  # noqa: BLE001 - diagnostics are best-effort
                pass

            # Stamped on the instance so a serialized model is self-describing
            # (MMMSerializer records fit_method/approximate from here).
            self._fit_diagnostics = dict(diagnostics)

            return ModelResults(
                trace=self._trace,
                model=self.model,
                config=getattr(self, "config", None),
                diagnostics=diagnostics,
            )

        if fit_method is FitMethod.SMC:
            # ---- Exact inference via tempered SMC (shared engine) ----
            from ...model.base import run_smc_fit
            from ...utils import arviz_compat

            trace, extra = run_smc_fit(
                self.model,
                draws=draws,
                chains=chains,
                random_seed=random_seed,
                **kwargs,
            )
            try:
                prior = self.sample_prior_predictive(
                    samples=1000, random_seed=random_seed
                )
                trace = arviz_compat.attach_prior(trace, prior)
            except Exception:  # noqa: BLE001 - prior is best-effort
                pass
            self._trace = trace

            diagnostics = {"fit_method": "smc", "approximate": False}
            try:
                from ...diagnostics import convergence as _conv

                diagnostics.update(_conv.compute_convergence(self._trace))
                diagnostics.update(extra)
                _conv.annotate(diagnostics)
                # Disagreeing SMC runs (high R-hat) = the multimodality signal.
                _conv.warn_if_not_converged(
                    diagnostics, label=f"{type(self).__name__} (SMC)"
                )
            except Exception:  # noqa: BLE001 - diagnostics are best-effort
                diagnostics.update(extra)

            self._fit_diagnostics = dict(diagnostics)

            return ModelResults(
                trace=self._trace,
                model=self.model,
                config=getattr(self, "config", None),
                diagnostics=diagnostics,
            )

        # ---- Approximate inference (MAP / Laplace / ADVI / full-rank ADVI /
        # Pathfinder) --
        from ...model.base import run_approximate_fit
        from ...utils import arviz_compat

        trace, extra = run_approximate_fit(
            self.model, fit_method, draws, random_seed, **kwargs
        )
        # Attach the prior so prior-vs-posterior tooling still works (best-effort).
        try:
            prior = self.sample_prior_predictive(samples=1000, random_seed=random_seed)
            trace = arviz_compat.attach_prior(trace, prior)
        except Exception:  # noqa: BLE001 - prior is best-effort for approx fits
            pass
        self._trace = trace

        diagnostics = {
            "fit_method": fit_method.value,
            "approximate": True,
            # R-hat / ESS are undefined for a single-path approximation.
            "rhat_max": None,
            "ess_bulk_min": None,
        }
        diagnostics.update(extra)
        try:
            from ...diagnostics import convergence as _conv

            # converged -> None ("not assessable"); never warns for approx fits.
            _conv.annotate(diagnostics)
        except Exception:  # noqa: BLE001
            pass

        self._fit_diagnostics = dict(diagnostics)

        return ModelResults(
            trace=self._trace,
            model=self.model,
            config=getattr(self, "config", None),
            diagnostics=diagnostics,
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

    # -- posterior-predictive outcome (BayesianMMM.predict parity) ------------

    @contextmanager
    def _swapped_data(
        self,
        X_media: np.ndarray | None = None,
        X_controls: np.ndarray | None = None,
    ):
        """Temporarily swap the graph's ``pm.Data`` containers for a
        counterfactual pass, restoring the training data afterwards — the same
        in-graph pattern :class:`StructuralNestedMMM` uses for its exact
        counterfactual mediation effects. ``X_media`` is raw-scale (the graph
        normalizes internally); ``X_controls`` is raw-scale and re-standardized
        with the training moments when the model standardizes controls."""
        model = self.model
        updates: dict[str, np.ndarray] = {}
        restores: dict[str, np.ndarray] = {}
        if X_media is not None:
            if "X_media" not in model.named_vars:
                raise ValueError(
                    "This model's graph has no 'X_media' data container; "
                    "counterfactual media prediction is not supported."
                )
            updates["X_media"] = np.asarray(X_media, dtype=float)
            restores["X_media"] = np.asarray(self.X_media, dtype=float)
        if X_controls is not None:
            if "X_controls" not in model.named_vars:
                raise ValueError("This model has no control-variable data container.")
            xc = np.asarray(X_controls, dtype=float)
            c_mean = getattr(self, "_controls_mean", None)
            c_std = getattr(self, "_controls_std", None)
            if c_mean is not None and c_std is not None:
                xc = (xc - c_mean) / c_std
            updates["X_controls"] = xc
            restores["X_controls"] = np.asarray(self.X_controls, dtype=float)
        if not updates:
            yield
            return
        with model:
            try:
                pm.set_data(updates)
                yield
            finally:
                pm.set_data(restores)

    def predict(
        self,
        X_media: np.ndarray | None = None,
        X_controls: np.ndarray | None = None,
        return_original_scale: bool = True,
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
    ):
        """Posterior-predictive outcome draws, mirroring
        :meth:`mmm_framework.model.base.BayesianMMM.predict`.

        Resamples the outcome likelihood (``y_obs``) under the posterior —
        optionally with counterfactual raw-scale ``X_media`` / ``X_controls``
        swapped into the graph — and returns a
        :class:`~mmm_framework.model.results.PredictionResults` whose
        ``y_pred_samples`` has shape ``(n_draws, n_obs)``. Latent states
        (mediators, factors) keep their posterior draws, so counterfactuals
        are structure-preserving.

        Single-outcome models (:class:`NestedMMM`,
        :class:`StructuralNestedMMM`) predict their one outcome; the
        multi-outcome models (:class:`MultivariateMMM`, :class:`CombinedMMM`)
        predict their **primary** outcome (see :meth:`_primary_outcome_index`)
        from the joint ``Y_obs`` likelihood, so the single-KPI reporting and
        goodness-of-fit tooling work on them too.
        """
        self._check_fitted()
        model = self.model
        # Single-outcome models expose a 1-D ``y_obs``; the multi-outcome models
        # a joint ``Y_obs`` of shape (obs, outcome) from which we take the
        # primary outcome column.
        if "y_obs" in model.named_vars and model.named_vars["y_obs"].ndim == 1:
            outcome_var, multi = "y_obs", False
        elif "Y_obs" in model.named_vars:
            outcome_var, multi = "Y_obs", True
        else:
            raise NotImplementedError(
                f"{type(self).__name__} has no recognized outcome node "
                "('y_obs' or 'Y_obs')."
            )

        with self._swapped_data(X_media, X_controls):
            with model:
                with warnings.catch_warnings():
                    # Multi-outcome graphs carry coord-bearing params; freezing
                    # them at their posterior draws while resampling only the
                    # likelihood is exactly what a predictive pass wants.
                    warnings.filterwarnings("ignore", message="The following trace")
                    pp = pm.sample_posterior_predictive(
                        self._trace,
                        var_names=[outcome_var],
                        random_seed=random_seed,
                        progressbar=False,
                    )

        y_samples = pp.posterior_predictive[outcome_var].values
        if multi:
            # (chain, draw, obs, outcome) -> primary outcome (chain, draw, obs).
            y_samples = y_samples[..., self._primary_outcome_index()]
        y_samples = y_samples.reshape(-1, y_samples.shape[-1]).astype(float)
        # The extension likelihoods observe the STANDARDIZED outcome; bridge
        # back to the caller's units exactly as the base model does. For the
        # multi-outcome models ``self.y_std``/``y_mean`` are the primary
        # outcome's moments (the first outcome), matching the selected column.
        if return_original_scale:
            y_samples = y_samples * self.y_std + self.y_mean

        from ...model.results import PredictionResults
        from ...utils import compute_hdi_bounds

        y_hdi_low, y_hdi_high = compute_hdi_bounds(y_samples, hdi_prob=hdi_prob, axis=0)
        return PredictionResults(
            posterior_predictive=pp,
            y_pred_mean=y_samples.mean(axis=0),
            y_pred_std=y_samples.std(axis=0),
            y_pred_hdi_low=y_hdi_low,
            y_pred_hdi_high=y_hdi_high,
            y_pred_samples=y_samples,
        )

    # -- interactive-report / response-curve surface (BayesianMMM parity) -----
    #
    # The reporting pipeline (``reporting.interactive`` /
    # ``reporting.helpers.measurement``) reads a small, duck-typed surface off
    # any fitted model: the raw media/outcome arrays, a period ``time_idx``, and
    # ``sample_channel_contributions``. The extension family stores its data as
    # ``self.X_media`` / ``self.y`` (raw) with one observation per period, so the
    # aliases below expose it under the names the report expects without
    # duplicating storage.

    @property
    def X_media_raw(self) -> np.ndarray:
        """Raw (pre-transform) media matrix ``(n_obs, n_channels)``."""
        return np.asarray(self.X_media, dtype=float)

    @property
    def y_raw(self) -> np.ndarray:
        """Observed outcome on the caller's original scale ``(n_obs,)``.

        Multi-outcome models (:class:`MultivariateMMM` / :class:`CombinedMMM`)
        report the **primary** outcome (see :meth:`_primary_outcome_index`); a
        2-D ``self.y`` is sliced to that column.
        """
        y = np.asarray(self.y, dtype=float)
        if y.ndim == 2:
            return y[:, self._primary_outcome_index()]
        return y

    @property
    def time_idx(self) -> np.ndarray:
        """Period index per observation. Extension models are a single national
        series (one obs per period), so this is a plain ``0..n_obs-1`` range."""
        return np.arange(self.n_obs)

    def _primary_outcome_index(self) -> int:
        """Index of the outcome the single-KPI report is built for.

        Single-outcome models (Nested / Structural) have exactly one, so this is
        0. Multi-outcome models override / honor a configured primary outcome.
        """
        return 0

    def sample_channel_contributions(
        self,
        X_media: np.ndarray | None = None,
        max_draws: int | None = None,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Posterior draws of per-channel contributions to the (primary) outcome.

        Evaluates the graph's ``channel_contributions`` deterministic — which
        each extension registers in ORIGINAL KPI units — with ``X_media`` (raw
        scale) optionally swapped in for a counterfactual scenario, returning
        ``(n_draws, n_obs, n_channels)``. Mirrors
        :meth:`mmm_framework.model.base.BayesianMMM.sample_channel_contributions`
        so the interactive report and response-curve tooling work unchanged.

        For the mediation / structural models the contribution is the channel's
        total (direct + mediated) linear effect on the outcome (linearized for
        the nonlinear structural paths, consistent with the pathway table); for
        the multi-outcome models it is the contribution to the primary outcome.
        Unlike the base model this returns ORIGINAL-scale contributions directly
        (the deterministic already carries ``y_std``), so callers must NOT
        rescale.
        """
        self._check_fitted()
        model = self.model
        if "channel_contributions" not in model.named_vars:
            raise NotImplementedError(
                f"{type(self).__name__} does not register a "
                "'channel_contributions' deterministic; per-channel report "
                "facts are unavailable for this model."
            )
        trace = self._trace
        if max_draws is not None:
            n_chains = trace.posterior.sizes["chain"]
            per_chain = max(1, int(np.ceil(max_draws / n_chains)))
            step = max(1, trace.posterior.sizes["draw"] // per_chain)
            trace = trace.sel(draw=slice(None, None, step))

        with self._swapped_data(X_media):
            with model:
                with warnings.catch_warnings():
                    # Posterior params are (correctly) held at their posterior
                    # draws while only the deterministic is re-evaluated under
                    # the swapped media — the "implicitly frozen" note is the
                    # intended behavior here, not a problem.
                    warnings.filterwarnings("ignore", message="The following trace")
                    pp = pm.sample_posterior_predictive(
                        trace,
                        var_names=["channel_contributions"],
                        random_seed=random_seed,
                        progressbar=False,
                    )
        contrib = pp.posterior_predictive["channel_contributions"].values
        # (chain, draw, obs, channel) -> (draws, obs, channel); already original
        # units (the deterministic carries y_std), so no rescale here.
        return contrib.reshape(-1, *contrib.shape[2:])

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary statistics."""

        self._check_fitted()
        from mmm_framework.utils.arviz_compat import summary as az_summary

        return az_summary(self._trace, var_names=var_names)

    def sample_prior_predictive(
        self, samples: int = 500, random_seed: int | None = None
    ) -> az.InferenceData:
        """Sample from the prior predictive distribution."""
        from ...utils import arviz_compat

        with self.model:
            return arviz_compat.sample_prior_predictive(samples, random_seed)

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
