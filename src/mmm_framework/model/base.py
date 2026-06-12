"""
BayesianMMM - Main model class.

This module contains the BayesianMMM class which orchestrates
model building, fitting, and prediction.

Identifiability note (equifinality, critique.md §3.6)
-----------------------------------------------------
The core model uses logistic saturation (a single ``sat_lam`` per channel) by
default, honoring each channel's configured :class:`SaturationConfig` type
(logistic / hill / michaelis_menten / tanh / none) and,
by default, normalized adstock (``AdstockConfig.normalize=True``). Normalization
folds the total carryover magnitude into the channel coefficient ``beta``, so a
channel's adstock decay, saturation strength, and ``beta`` trade off against one
another: visibly different parameter combinations can fit the data almost
equally well. This is intrinsic to additive MMM, not a defect, but it means the
per-channel decomposition is only weakly identified from observational spend
alone. The framework's primary remedy is experiment-calibrated coefficient
priors (:mod:`mmm_framework.calibration`), which anchor ``beta`` to randomized
evidence and thereby break the trade-off; weakly-informative priors and (for the
Hill path) data-anchored ``kappa`` bounds help at the margin. See
:func:`mmm_framework.transforms.adstock.adstock_weights` for the kernel-level
discussion.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from ..config import (
    AdstockConfig,
    AdstockType,
    CausalControlRole,
    ModelConfig,
    PriorConfig,
    PriorType,
    SaturationConfig,
    SaturationType,
)
from ..data_loader import PanelDataset
from ..utils import compute_hdi_bounds
from ..transforms import (
    adstock_weights,
    geometric_adstock_2d,
    create_fourier_features,
    create_bspline_basis,
    create_piecewise_trend_matrix,
)
from ..transforms.adstock_pt import (
    parametric_adstock_panel_pt,
    parametric_adstock_pt,
)

from .results import (
    MMMResults,
    PredictionResults,
    ContributionResults,
    ComponentDecomposition,
)
from .trend_config import TrendType, TrendConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..calibration.likelihood import ExperimentMeasurement

# Map an AdstockType to the kernel name used by the parametric adstock kernels.
_ADSTOCK_KIND = {
    AdstockType.GEOMETRIC: "geometric",
    AdstockType.DELAYED: "delayed",
    AdstockType.WEIBULL: "weibull",
    AdstockType.NONE: "none",
}

# Control-coefficient prior widths (on standardized data), keyed by causal role.
# A *confounder* must not be shrunk toward zero: the project README proves that
# shrinking a confounder biases the media coefficient by (1 - s)*gamma*Cov/Var,
# so confounders get a wide, weakly-informative prior. Every other control --
# including any left unmarked -- keeps the model's historical Normal(0, 0.5),
# which is mildly regularizing (a precision-control choice). These are fixed
# widths by role, not the per-control ``coefficient_prior`` (which the core
# model has never honored for controls; honoring it now would silently change
# existing fits).
_CONFOUNDER_PRIOR_SIGMA = 2.0
_PRECISION_CONTROL_PRIOR_SIGMA = 0.5

# Causal roles that must never be conditioned on as controls (doing so induces
# bias for a total-effect estimate). Refused at model-construction time.
_BLOCKED_CONTROL_ROLES = (CausalControlRole.MEDIATOR, CausalControlRole.COLLIDER)


def _hdi_finite(samples: "NDArray", hdi_prob: float) -> tuple[float, float]:
    """HDI bounds over the finite entries of ``samples``.

    A pathological posterior can put non-finite values into the predictive draws;
    ``np.percentile`` would then return NaN and silently poison the interval.
    Filtering to finite draws keeps a few bad samples from corrupting the bound,
    and only returns ``(nan, nan)`` when there is nothing usable to summarize.
    """
    finite = samples[np.isfinite(samples)]
    if finite.size < 2:
        return float("nan"), float("nan")
    low, high = compute_hdi_bounds(finite, hdi_prob=hdi_prob, axis=0)
    return float(low), float(high)


def _sample_from_prior_config(
    name: str,
    prior: PriorConfig | None,
    default: Callable[[], "pt.TensorVariable"],
) -> "pt.TensorVariable":
    """Sample a PyMC variable from a :class:`PriorConfig`, or a default.

    Honors the common prior distributions configured on an
    :class:`AdstockConfig` so the core model's parametric adstock respects
    user-specified priors. Falls back to ``default()`` when ``prior`` is None
    or its distribution is not recognized.
    """
    if prior is None:
        return default()

    p = prior.params
    dist = prior.distribution
    if dist == PriorType.BETA:
        return pm.Beta(name, alpha=p.get("alpha", 2.0), beta=p.get("beta", 2.0))
    if dist == PriorType.GAMMA:
        return pm.Gamma(name, alpha=p.get("alpha", 2.0), beta=p.get("beta", 1.0))
    if dist == PriorType.HALF_NORMAL:
        return pm.HalfNormal(name, sigma=p.get("sigma", 1.0))
    if dist == PriorType.NORMAL:
        return pm.Normal(name, mu=p.get("mu", 0.0), sigma=p.get("sigma", 1.0))
    if dist == PriorType.LOG_NORMAL:
        return pm.LogNormal(name, mu=p.get("mu", 0.0), sigma=p.get("sigma", 1.0))
    if dist == PriorType.TRUNCATED_NORMAL:
        return pm.TruncatedNormal(
            name,
            mu=p.get("mu", 0.0),
            sigma=p.get("sigma", 1.0),
            lower=p.get("lower"),
            upper=p.get("upper"),
        )
    if dist == PriorType.HALF_STUDENT_T:
        return pm.HalfStudentT(name, nu=p.get("nu", 3.0), sigma=p.get("sigma", 1.0))
    return default()


def _apply_saturation_pt(
    x: "pt.TensorVariable",
    kind: SaturationType,
    params: dict[str, Any],
) -> "pt.TensorVariable":
    """Apply a channel's saturation transform to a pytensor input.

    This is the SINGLE place the saturation formula lives: the model build
    (:meth:`BayesianMMM._build_model`) and every re-evaluation of a perturbed
    spend series (:meth:`BayesianMMM._perturbed_contribution_sum`) route
    through it, so the likelihood and the marginal/counterfactual estimands
    cannot drift apart.

    Args:
        x: Adstocked, normalized (roughly ``[0, 1]``) spend tensor.
        kind: The channel's configured :class:`SaturationType`.
        params: The channel's saturation RVs as created by
            :meth:`BayesianMMM._build_channel_saturation` (``sat_lam`` for
            logistic; ``sat_half``/``sat_slope`` for hill; ``sat_half`` for
            michaelis_menten and tanh; empty for none).

    Returns:
        The saturated tensor.
    """
    if kind == SaturationType.LOGISTIC:
        sat_lam = params["sat_lam"]
        exponent = pt.clip(-sat_lam * x, -20, 0)
        return 1 - pt.exp(exponent)
    if kind == SaturationType.HILL:
        # x^s / (x^s + k^s). Clamp x away from 0: d/dx x^s is unbounded at
        # x = 0 for s < 1, which would hand NUTS infinite gradients on
        # zero-spend weeks (maximum's gradient is exactly 0 below the bound).
        sat_half = params["sat_half"]
        sat_slope = params["sat_slope"]
        x_safe = pt.maximum(x, 1e-9)
        x_pow = x_safe**sat_slope
        return x_pow / (x_pow + sat_half**sat_slope)
    if kind == SaturationType.MICHAELIS_MENTEN:
        sat_half = params["sat_half"]
        return x / (x + sat_half)
    if kind == SaturationType.TANH:
        sat_half = params["sat_half"]
        return pt.tanh(x / sat_half)
    # SaturationType.NONE: identity (no saturation RVs).
    return x


class BayesianMMM:
    """
    Bayesian Marketing Mix Model - Robust Implementation with Prediction Support.

    This implementation prioritizes numerical stability:
    - All data is standardized before modeling
    - Adstock: by default pre-computed at fixed alphas and blended via a learned
      mix (fast). Set ``ModelConfig.use_parametric_adstock=True`` to instead
      estimate a continuous in-graph kernel per channel, honoring each
      ``MediaChannelConfig.adstock`` (geometric, delayed, or Weibull).
    - Saturation: logistic (``1 - exp(-lam * x)``) by default -- the most
      stable choice -- honoring each ``MediaChannelConfig.saturation`` type per
      channel (logistic, hill, michaelis_menten, tanh, or none)
    - Priors are carefully scaled for standardized data
    - Flexible trend modeling with GP, spline, and piecewise options
    - Support for prediction and counterfactual analysis
    - Save/load functionality for model persistence

    Parameters
    ----------
    panel : PanelDataset
        Panel data from MFFLoader.
    model_config : ModelConfig
        Model configuration.
    trend_config : TrendConfig, optional
        Trend specification.
    adstock_alphas : list[float], optional
        Fixed adstock decay values to pre-compute.
    """

    _VERSION = "1.0.0"

    def __init__(
        self,
        panel: PanelDataset,
        model_config: ModelConfig,
        trend_config: TrendConfig | None = None,
        adstock_alphas: list[float] | None = None,
        experiments: "Sequence[ExperimentMeasurement] | None" = None,
    ):
        self.panel = panel
        self.model_config = model_config
        self.trend_config = trend_config or TrendConfig()
        self.adstock_alphas = adstock_alphas or [0.0, 0.3, 0.5, 0.7, 0.9]

        self.mff_config = panel.config
        self.hierarchical_config = model_config.hierarchical
        self.seasonality_config = model_config.seasonality
        self.use_parametric_adstock = getattr(
            model_config, "use_parametric_adstock", False
        )

        # Experimental results folded in as in-graph likelihood terms (None ->
        # the model graph is byte-identical to the un-calibrated model).
        self.experiments: list["ExperimentMeasurement"] = (
            list(experiments) if experiments else []
        )

        self._model: pm.Model | None = None
        self._trace: az.InferenceData | None = None

        # Store scaling parameters for prediction
        self._scaling_params: dict[str, Any] = {}

        self._prepare_data()

    def add_experiment_calibration(
        self, experiments: "Sequence[ExperimentMeasurement]"
    ) -> "BayesianMMM":
        """Register experiment likelihood terms and invalidate the built graph.

        The experiments are folded into the model as likelihood terms the next
        time the graph is built (so call this before :meth:`fit`). Returns
        ``self`` for chaining.
        """
        self.experiments = list(experiments)
        self._model = None  # force a rebuild that includes the new likelihoods
        return self

    def _prepare_data(self):
        """Prepare and standardize all data."""
        # === Raw data ===
        self.y_raw = self.panel.y.values.astype(np.float64)
        self.X_media_raw = self.panel.X_media.values.astype(np.float64)

        if self.panel.X_controls is not None and self.panel.X_controls.shape[1] > 0:
            self.X_controls_raw = self.panel.X_controls.values.astype(np.float64)
        else:
            self.X_controls_raw = None

        # === Dimensions ===
        self.n_obs = len(self.y_raw)
        self.n_channels = self.X_media_raw.shape[1]
        self.n_controls = (
            self.X_controls_raw.shape[1] if self.X_controls_raw is not None else 0
        )

        self.channel_names = list(self.panel.coords.channels)
        self.control_names = (
            list(self.panel.coords.controls) if self.n_controls > 0 else []
        )

        # === Standardize target ===
        self.y_mean = float(self.y_raw.mean())
        self.y_std = float(self.y_raw.std()) + 1e-8
        self.y = (self.y_raw - self.y_mean) / self.y_std

        # Store scaling parameters
        self._scaling_params["y_mean"] = self.y_mean
        self._scaling_params["y_std"] = self.y_std

        # Per-channel max of the raw series, used to normalize media for the
        # parametric (in-graph) adstock path.
        self._media_raw_max = {
            ch: float(self.X_media_raw[:, c].max())
            for c, ch in enumerate(self.channel_names)
        }

        # === Geo/product/time structure ===
        # Resolved before any adstock computation: panel observations are a
        # stacked (period x geography x product) vector, and carryover must run
        # along each cross-section cell's own time axis — never across cells.
        self.has_geo = self.panel.coords.has_geo
        self.has_product = self.panel.coords.has_product
        self.n_geos = self.panel.coords.n_geos
        self.n_products = self.panel.coords.n_products

        if self.has_geo:
            self.geo_names = list(self.panel.coords.geographies)
            self.geo_idx = self._get_group_indices("geography")
        else:
            self.geo_idx = np.zeros(self.n_obs, dtype=np.int32)

        if self.has_product:
            self.product_names = list(self.panel.coords.products)
            self.product_idx = self._get_group_indices("product")
        else:
            self.product_idx = np.zeros(self.n_obs, dtype=np.int32)

        self.n_periods = self.panel.coords.n_periods
        self.time_idx = self._get_time_index()
        self.t_scaled = np.linspace(0, 1, self.n_periods)

        # Flat cross-section cell index (geo x product) for per-cell adstock.
        self.n_cells = self.n_geos * self.n_products
        self.cell_idx = (
            self.geo_idx.astype(np.int64) * self.n_products
            + self.product_idx.astype(np.int64)
        ).astype(np.int32)

        # === Pre-compute adstocked media at fixed alphas ===
        self._media_max = {}
        self.X_media_adstocked = {}

        for alpha in self.adstock_alphas:
            adstocked = self._geometric_adstock_per_cell(self.X_media_raw, alpha)
            for c in range(self.n_channels):
                key = self.channel_names[c]
                current_max = adstocked[:, c].max()
                if key not in self._media_max:
                    self._media_max[key] = current_max
                else:
                    self._media_max[key] = max(self._media_max[key], current_max)

        # Normalize using consistent max values
        for alpha in self.adstock_alphas:
            adstocked = self._geometric_adstock_per_cell(self.X_media_raw, alpha)
            normalized = np.zeros_like(adstocked)
            for c, ch_name in enumerate(self.channel_names):
                normalized[:, c] = adstocked[:, c] / (self._media_max[ch_name] + 1e-8)
            self.X_media_adstocked[alpha] = normalized

        self._scaling_params["media_max"] = self._media_max.copy()

        # === Standardize controls ===
        if self.X_controls_raw is not None:
            self.control_mean = self.X_controls_raw.mean(axis=0)
            self.control_std = self.X_controls_raw.std(axis=0) + 1e-8
            self.X_controls = (
                self.X_controls_raw - self.control_mean
            ) / self.control_std
            self._scaling_params["control_mean"] = self.control_mean.copy()
            self._scaling_params["control_std"] = self.control_std.copy()
        else:
            self.X_controls = None

        # === Causal roles of controls (bad-control prevention) ===
        # Read each control's causal role from its config and refuse to condition
        # on mediators/colliders (doing so biases a total-effect estimate). This
        # is the enforcement point for both manually-typed roles (P1-2) and roles
        # the DAG builder infers from the identified adjustment set (P1-1).
        self._control_causal_roles = self._resolve_control_causal_roles()

        # === Seasonality features ===
        self._prepare_seasonality()

        # === Trend features ===
        self._prepare_trend()

        # === Media hierarchy ===
        self.media_groups = self.mff_config.get_hierarchical_media_groups()
        self.has_media_hierarchy = len(self.media_groups) > 0

    def _resolve_control_causal_roles(self) -> list[CausalControlRole | None]:
        """Resolve the causal role of each control and refuse bad controls.

        Returns a list aligned with ``self.control_names`` giving each control's
        :class:`CausalControlRole` (or ``None`` when unmarked). Raises
        ``ValueError`` if any control is a mediator or collider, because
        conditioning on a post-treatment / collider variable induces bias for a
        total-effect estimate (the exact "bad control" error the framework
        documents). This is intentionally a hard failure: a silently-conditioned
        mediator produces a confidently wrong number.
        """
        roles: list[CausalControlRole | None] = []
        blocked: list[str] = []
        for name in self.control_names:
            cfg = self.mff_config.get_control_config(name)
            role = getattr(cfg, "causal_role", None) if cfg is not None else None
            roles.append(role)
            if role in _BLOCKED_CONTROL_ROLES:
                reason = getattr(cfg, "causal_role_reason", None)
                detail = f" ({reason})" if reason else ""
                blocked.append(f"'{name}' [{role.value}]{detail}")

        if blocked:
            raise ValueError(
                "Refusing to condition on post-treatment / collider variables "
                "used as controls: "
                + "; ".join(blocked)
                + ". Conditioning on a mediator blocks part of the media effect, "
                "and conditioning on a collider opens a spurious path -- either "
                "biases a total-effect estimate. Remove these from `controls`. "
                "If a variable is genuinely a common cause of media and the KPI, "
                "mark it `causal_role=CausalControlRole.CONFOUNDER` instead."
            )
        return roles

    def _control_prior_sigmas(self) -> np.ndarray:
        """Per-control coefficient-prior standard deviations, keyed by role.

        Confounders receive a wide, un-shrunk prior; every other control keeps
        the model's historical regularizing width. Returned as a vector so the
        single ``beta_controls`` random variable (relied on by reporting and
        validation) keeps its name and ``(n_controls,)`` shape.
        """
        sigmas = np.full(self.n_controls, _PRECISION_CONTROL_PRIOR_SIGMA)
        for i, role in enumerate(self._control_causal_roles):
            if role == CausalControlRole.CONFOUNDER:
                sigmas[i] = _CONFOUNDER_PRIOR_SIGMA
        return sigmas

    def _selection_active(self) -> bool:
        """True when a variable-selection prior should be wired on controls.

        Off by default and whenever there are fewer than two *selectable* (non-
        confounder) controls -- confounders are never shrunk, so selection needs a
        non-trivial selectable set to do anything.
        """
        method = getattr(self.model_config.control_selection, "method", "none")
        if self.n_controls == 0 or method == "none":
            return False
        n_select = sum(
            1 for r in self._control_causal_roles if r != CausalControlRole.CONFOUNDER
        )
        return n_select >= 2

    def _build_control_betas(self, sigma: "pt.TensorVariable | None"):
        """Control coefficients, honoring causal roles and optional selection.

        **Off (default):** a single ``Normal('beta_controls', sigma=role_widths)``
        -- *bit-identical* to the historical model.

        **On** (``ModelConfig.control_selection.method != 'none'``): confounders
        keep the wide, un-shrunk prior (shrinking a confounder re-opens the
        back-door), while the remaining precision/irrelevant controls get a
        horseshoe / spike-slab / LASSO prior so the model *selects* among them.
        ``beta_controls`` is preserved (name, shape, ``control`` dim) so reporting
        and validation are unaffected.
        """
        if not self._selection_active():
            return pm.Normal(
                "beta_controls",
                mu=0,
                sigma=self._control_prior_sigmas(),
                shape=self.n_controls,
            )

        conf_mask = np.array(
            [r == CausalControlRole.CONFOUNDER for r in self._control_causal_roles],
            dtype=bool,
        )
        beta = pt.zeros(self.n_controls)
        if conf_mask.any():
            conf_idx = np.where(conf_mask)[0]
            beta_conf = pm.Normal(
                "beta_controls_confounder",
                mu=0,
                sigma=_CONFOUNDER_PRIOR_SIGMA,
                shape=int(conf_mask.sum()),
            )
            beta = pt.set_subtensor(beta[conf_idx], beta_conf)
        sel_idx = np.where(~conf_mask)[0]
        beta = pt.set_subtensor(
            beta[sel_idx], self._selection_prior(int((~conf_mask).sum()), sigma)
        )
        return pm.Deterministic("beta_controls", beta, dims="control")

    def _selection_prior(self, n_select: int, sigma: "pt.TensorVariable | None"):
        """Coefficients for the selectable controls under the configured method.

        Reuses the tested ``mmm_extensions`` priors (lazy import: only when
        selection is actually enabled). The horseshoe global scale uses the real
        observation ``sigma`` (Piironen & Vehtari, 2017), so it shrinks correctly
        on the standardized scale.
        """
        cfg = self.model_config.control_selection
        from ..mmm_extensions.components.variable_selection import (
            create_bayesian_lasso_prior,
            create_regularized_horseshoe_prior,
            create_spike_slab_prior,
        )
        from ..mmm_extensions.config import (
            HorseshoeConfig,
            LassoConfig,
            SpikeSlabConfig,
        )

        name = "beta_controls_select"
        if cfg.method in ("horseshoe", "finnish_horseshoe"):
            hc = HorseshoeConfig(
                expected_nonzero=max(1, min(cfg.expected_nonzero, n_select - 1))
            )
            return create_regularized_horseshoe_prior(
                name, n_select, self.n_obs, sigma, hc
            ).beta
        if cfg.method == "spike_slab":
            return create_spike_slab_prior(name, n_select, SpikeSlabConfig()).beta
        if cfg.method == "lasso":
            return create_bayesian_lasso_prior(
                name, n_select, LassoConfig(regularization=cfg.regularization)
            ).beta
        raise ValueError(f"Unknown control_selection.method: {cfg.method!r}")

    def _get_group_indices(self, level_name: str) -> np.ndarray:
        """Get group indices for a hierarchical level."""
        cols = self.mff_config.columns
        col_name = getattr(cols, level_name)

        if isinstance(self.panel.index, pd.MultiIndex):
            values = self.panel.index.get_level_values(col_name)
            if level_name == "geography":
                categories = self.geo_names
            else:
                categories = self.product_names
            return pd.Categorical(values, categories=categories).codes.astype(np.int32)
        return np.zeros(self.n_obs, dtype=np.int32)

    def _get_time_index(self) -> np.ndarray:
        """Get time index for each observation."""
        cols = self.mff_config.columns

        if isinstance(self.panel.index, pd.MultiIndex):
            period_values = self.panel.index.get_level_values(cols.period)
            periods_unique = list(self.panel.coords.periods)
            return pd.Categorical(
                period_values, categories=periods_unique
            ).codes.astype(np.int32)
        return np.arange(self.n_obs, dtype=np.int32)

    def _prepare_seasonality(self):
        """Prepare Fourier features for seasonality.

        Periods are derived from the data frequency (MFFConfig.frequency), so
        yearly/monthly/weekly components mean the same thing on weekly and
        daily data. Components that the sampling frequency cannot represent
        (e.g. weekly seasonality in weekly data) are skipped with a warning —
        previously monthly/weekly were silently ignored for ALL data.
        """
        self.seasonality_features = {}
        t = np.arange(self.n_periods)

        freq = getattr(self.mff_config, "frequency", "W") or "W"
        periods_by_freq: dict[str, dict[str, float]] = {
            "W": {"yearly": 52.0, "monthly": 52.0 / 12.0},
            "D": {"yearly": 365.25, "monthly": 365.25 / 12.0, "weekly": 7.0},
            "M": {"yearly": 12.0},
        }
        component_periods = periods_by_freq.get(freq, periods_by_freq["W"])

        for component in ("yearly", "monthly", "weekly"):
            order = getattr(self.seasonality_config, component, None)
            if not order or order <= 0:
                continue
            period = component_periods.get(component)
            if period is None:
                warnings.warn(
                    f"{component} seasonality cannot be represented at data "
                    f"frequency '{freq}'; skipping this component.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            # Harmonics above the Nyquist limit (period/2 observations) alias
            max_order = max(1, int(period / 2))
            if order > max_order:
                warnings.warn(
                    f"{component} seasonality order {order} exceeds the "
                    f"max resolvable order {max_order} for period {period:.2f} "
                    f"at frequency '{freq}'; clamping to {max_order}.",
                    UserWarning,
                    stacklevel=2,
                )
                order = max_order
            features = create_fourier_features(t, period, order)
            if features.shape[1] > 0:
                self.seasonality_features[component] = features

    def _prepare_trend(self):
        """Prepare trend features based on configuration."""
        t_unique = np.linspace(0, 1, self.n_periods)

        self.trend_features = {}

        if self.trend_config.type == TrendType.SPLINE:
            self.trend_features["spline_basis"] = create_bspline_basis(
                t_unique,
                n_knots=self.trend_config.n_knots,
                degree=self.trend_config.spline_degree,
            )
            self.trend_features["n_spline_coef"] = self.trend_features[
                "spline_basis"
            ].shape[1]

        elif self.trend_config.type == TrendType.PIECEWISE:
            s, A = create_piecewise_trend_matrix(
                t_unique,
                n_changepoints=self.trend_config.n_changepoints,
                changepoint_range=self.trend_config.changepoint_range,
            )
            self.trend_features["changepoints"] = s
            self.trend_features["changepoint_matrix"] = A

        elif self.trend_config.type == TrendType.GP:
            self.trend_features["gp_config"] = {
                "lengthscale_mu": self.trend_config.gp_lengthscale_prior_mu,
                "lengthscale_sigma": self.trend_config.gp_lengthscale_prior_sigma,
                "amplitude_sigma": self.trend_config.gp_amplitude_prior_sigma,
                "n_basis": self.trend_config.gp_n_basis,
                "c": self.trend_config.gp_c,
            }

    def _get_time_mask(self, time_period: tuple[int, int] | None) -> NDArray[np.bool_]:
        """Create boolean mask for time period filtering."""
        if time_period is not None:
            start_idx, end_idx = time_period
            return (self.time_idx >= start_idx) & (self.time_idx <= end_idx)
        return np.ones(self.n_obs, dtype=bool)

    def _build_coords(self) -> dict:
        """Build PyMC coordinate dictionary."""
        coords = {
            "obs": np.arange(self.n_obs),
            "channel": self.channel_names,
        }

        if self.has_geo:
            coords["geo"] = self.geo_names
        if self.has_product:
            coords["product"] = self.product_names
        if self.n_controls > 0:
            coords["control"] = self.control_names

        for name, features in self.seasonality_features.items():
            n_features = features.shape[1]
            coords[f"{name}_fourier"] = [f"{name}_{i}" for i in range(n_features)]

        for parent, children in self.media_groups.items():
            coords[f"{parent}_platform"] = list(children)

        if self.trend_config.type == TrendType.SPLINE:
            n_coef = self.trend_features.get("n_spline_coef", 0)
            coords["spline_idx"] = list(range(n_coef))

        elif self.trend_config.type == TrendType.PIECEWISE:
            n_cp = len(self.trend_features.get("changepoints", []))
            coords["changepoint"] = list(range(n_cp))

        return coords

    def _build_trend_component(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build the trend component based on configuration."""
        t_scaled_tensor = pt.as_tensor_variable(self.t_scaled)

        if self.trend_config.type == TrendType.NONE:
            return pt.zeros(time_idx.shape[0])

        elif self.trend_config.type == TrendType.LINEAR:
            trend_slope = pm.Normal(
                "trend_slope",
                mu=self.trend_config.growth_prior_mu,
                sigma=self.trend_config.growth_prior_sigma,
            )
            return trend_slope * t_scaled_tensor[time_idx]

        elif self.trend_config.type == TrendType.PIECEWISE:
            return self._build_piecewise_trend(model, time_idx)

        elif self.trend_config.type == TrendType.SPLINE:
            return self._build_spline_trend(model, time_idx)

        elif self.trend_config.type == TrendType.GP:
            return self._build_gp_trend(model, time_idx)

        else:
            warnings.warn(f"Unknown trend type: {self.trend_config.type}, using linear")
            trend_slope = pm.Normal("trend_slope", mu=0, sigma=0.5)
            return trend_slope * t_scaled_tensor[time_idx]

    def _build_piecewise_trend(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build Prophet-style piecewise linear trend."""
        s = self.trend_features["changepoints"]
        A = self.trend_features["changepoint_matrix"]
        n_changepoints = len(s)

        k = pm.Normal(
            "trend_k",
            mu=self.trend_config.growth_prior_mu,
            sigma=self.trend_config.growth_prior_sigma,
        )

        delta = pm.Laplace(
            "trend_delta",
            mu=0,
            b=self.trend_config.changepoint_prior_scale,
            shape=n_changepoints,
            dims="changepoint",
        )

        m = pm.Normal("trend_m", mu=0, sigma=0.5)

        t_unique = np.linspace(0, 1, self.n_periods)
        t_unique_tensor = pt.as_tensor_variable(t_unique)
        A_tensor = pt.as_tensor_variable(A)
        s_tensor = pt.as_tensor_variable(s)

        # Prophet piecewise-linear trend: each ``delta_j`` adjusts the growth *rate*
        # (slope) from changepoint ``s_j`` onward, and ``gamma_j = -s_j * delta_j`` keeps
        # the trend continuous at the changepoint. The cumulative slope ``k + A . delta``
        # multiplies ``t``; ``m + A . gamma`` is the offset. (Previously ``A . delta`` was
        # added *without* the ``* t``, which produced piecewise-constant level shifts
        # rather than the intended slope changes -- the lone ``gamma`` continuity term,
        # meaningless under level shifts, is the tell that slope changes were intended.)
        gamma = -s_tensor * delta
        slope = k + pt.dot(A_tensor, delta)
        offset = m + pt.dot(A_tensor, gamma)
        trend_unique = slope * t_unique_tensor + offset

        return trend_unique[time_idx]

    def _build_spline_trend(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build B-spline trend."""
        basis = self.trend_features["spline_basis"]
        n_coef = basis.shape[1]

        spline_coef_raw = pm.Normal(
            "spline_coef_raw", mu=0, sigma=1, shape=n_coef, dims="spline_idx"
        )

        spline_scale = pm.HalfNormal(
            "spline_scale", sigma=self.trend_config.spline_prior_sigma
        )

        spline_coef = pm.Deterministic(
            "spline_coef", spline_scale * pt.cumsum(spline_coef_raw), dims="spline_idx"
        )

        basis_tensor = pt.as_tensor_variable(basis)
        trend_unique = pt.dot(basis_tensor, spline_coef)
        trend_unique = trend_unique - trend_unique.mean()

        return trend_unique[time_idx]

    def _build_gp_trend(self, model: pm.Model, time_idx) -> pt.TensorVariable:
        """Build Gaussian Process trend using HSGP approximation."""
        gp_config = self.trend_features["gp_config"]

        gp_lengthscale = pm.LogNormal(
            "gp_lengthscale",
            mu=np.log(gp_config["lengthscale_mu"]),
            sigma=gp_config["lengthscale_sigma"],
        )

        gp_amplitude = pm.HalfNormal("gp_amplitude", sigma=gp_config["amplitude_sigma"])

        try:
            import pymc.gp as gp_module

            cov_func = gp_amplitude**2 * gp_module.cov.Matern32(
                input_dim=1, ls=gp_lengthscale
            )

            gp = gp_module.HSGP(
                m=[gp_config["n_basis"]], c=gp_config["c"], cov_func=cov_func
            )

            t_unique = np.linspace(-1, 1, self.n_periods).reshape(-1, 1)
            trend_unique = gp.prior("trend_gp", X=t_unique)

            return trend_unique[time_idx]

        except (ImportError, AttributeError) as e:
            warnings.warn(
                f"HSGP not available ({e}), falling back to basis function GP"
            )
            return self._build_gp_trend_basis(
                model, gp_lengthscale, gp_amplitude, time_idx
            )

    def _build_gp_trend_basis(
        self,
        model: pm.Model,
        lengthscale: pt.TensorVariable,
        amplitude: pt.TensorVariable,
        time_idx,
    ) -> pt.TensorVariable:
        """Build GP trend using explicit basis function approximation."""
        gp_config = self.trend_features["gp_config"]
        n_basis = gp_config["n_basis"]

        t_unique = np.linspace(0, 1, self.n_periods)
        frequencies = np.arange(1, n_basis + 1)

        basis_sin = np.sin(2 * np.pi * np.outer(t_unique, frequencies))
        basis_cos = np.cos(2 * np.pi * np.outer(t_unique, frequencies))
        basis = np.hstack([basis_sin, basis_cos])

        basis_tensor = pt.as_tensor_variable(basis)

        omega = 2 * np.pi * frequencies
        omega_tensor = pt.as_tensor_variable(omega)

        gp_coef = pm.Normal("gp_coef", mu=0, sigma=1, shape=2 * n_basis)

        spectral_weights_sin = pt.exp(-0.5 * (omega_tensor * lengthscale) ** 2)
        spectral_weights_cos = pt.exp(-0.5 * (omega_tensor * lengthscale) ** 2)
        spectral_weights = pt.concatenate([spectral_weights_sin, spectral_weights_cos])

        scaled_coef = amplitude * gp_coef * pt.sqrt(spectral_weights / n_basis)
        trend_unique = pt.dot(basis_tensor, scaled_coef)
        trend_unique = trend_unique - trend_unique.mean()

        return trend_unique[time_idx]

    def _geometric_adstock_per_cell(self, X: np.ndarray, alpha: float) -> np.ndarray:
        """Geometric adstock applied within each geo/product cell separately.

        For national data (one cell) this is exactly
        :func:`geometric_adstock_2d`. For panels it convolves each
        cross-section cell's own time series, so carryover never bleeds from
        one geography/product into the rows of another.
        """
        if self.n_cells <= 1:
            return geometric_adstock_2d(X, alpha)
        out = np.zeros_like(X, dtype=np.float64)
        for k in range(self.n_cells):
            rows = np.flatnonzero(self.cell_idx == k)
            rows = rows[np.argsort(self.time_idx[rows], kind="stable")]
            out[rows] = geometric_adstock_2d(X[rows], alpha)
        return out

    def _prepare_media_data_for_model(
        self, X_media_raw: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare media data for model (compute adstock at low/high alpha)."""
        if X_media_raw is None:
            X_media_raw = self.X_media_raw

        alpha_low = self.adstock_alphas[0]
        alpha_high = self.adstock_alphas[-1]

        adstock_low = self._geometric_adstock_per_cell(X_media_raw, alpha_low)
        adstock_high = self._geometric_adstock_per_cell(X_media_raw, alpha_high)

        for c, ch_name in enumerate(self.channel_names):
            max_val = self._media_max[ch_name] + 1e-8
            adstock_low[:, c] = adstock_low[:, c] / max_val
            adstock_high[:, c] = adstock_high[:, c] / max_val

        return adstock_low, adstock_high

    def _prepare_raw_media_for_model(
        self, X_media_raw: np.ndarray | None = None
    ) -> np.ndarray:
        """Normalize raw media per channel for the parametric adstock path.

        Adstock is applied in-graph for this path, so the model is fed the
        normalized *raw* series (scaled to roughly [0, 1] by the per-channel
        training max) rather than a pre-adstocked series.
        """
        if X_media_raw is None:
            X_media_raw = self.X_media_raw

        normalized = np.zeros_like(X_media_raw, dtype=np.float64)
        for c, ch_name in enumerate(self.channel_names):
            max_val = self._media_raw_max[ch_name] + 1e-8
            normalized[:, c] = X_media_raw[:, c] / max_val
        return normalized

    def _get_adstock_config(self, channel_name: str) -> AdstockConfig:
        """Resolve the AdstockConfig for a channel, defaulting to geometric."""
        media_cfg = self.mff_config.get_media_config(channel_name)
        if media_cfg is not None and media_cfg.adstock is not None:
            return media_cfg.adstock
        return AdstockConfig.geometric()

    def _get_saturation_config(self, channel_name: str) -> SaturationConfig:
        """Resolve the SaturationConfig for a channel, defaulting to logistic."""
        media_cfg = self.mff_config.get_media_config(channel_name)
        if media_cfg is not None and media_cfg.saturation is not None:
            return media_cfg.saturation
        return SaturationConfig.logistic()

    def _build_channel_saturation(
        self, channel_name: str
    ) -> tuple[SaturationType, dict[str, Any]]:
        """Create the saturation RVs for one channel per its SaturationConfig.

        Must be called inside the ``pm.Model`` context. Returns the channel's
        saturation kind plus a params dict consumed by
        :func:`_apply_saturation_pt`. The logistic default reproduces the
        historical graph exactly (``sat_lam_<ch> ~ Exponential(lam=0.5)``), so
        default configs stay bit-identical to models built before saturation
        types were honored.
        """
        cfg = self._get_saturation_config(channel_name)
        kind = cfg.type
        params: dict[str, Any] = {}

        if kind == SaturationType.LOGISTIC:
            params["sat_lam"] = _sample_from_prior_config(
                f"sat_lam_{channel_name}",
                cfg.lam_prior,
                lambda: pm.Exponential(f"sat_lam_{channel_name}", lam=0.5),
            )
        elif kind == SaturationType.HILL:
            # Half-saturation point of the adstocked normalized (~[0, 1])
            # spend; Beta(2, 2) centers it well inside the data's support.
            params["sat_half"] = _sample_from_prior_config(
                f"sat_half_{channel_name}",
                cfg.kappa_prior,
                lambda: pm.Beta(f"sat_half_{channel_name}", alpha=2, beta=2),
            )
            params["sat_slope"] = _sample_from_prior_config(
                f"sat_slope_{channel_name}",
                cfg.slope_prior,
                lambda: pm.HalfNormal(f"sat_slope_{channel_name}", sigma=1.5),
            )
        elif kind in (SaturationType.MICHAELIS_MENTEN, SaturationType.TANH):
            params["sat_half"] = _sample_from_prior_config(
                f"sat_half_{channel_name}",
                cfg.kappa_prior,
                lambda: pm.Beta(f"sat_half_{channel_name}", alpha=2, beta=2),
            )
        # SaturationType.NONE: no RVs.
        return kind, params

    def _apply_adstock_kernel_pt(
        self, x: "pt.TensorVariable", kind: str, l_max: int, normalize: bool, **params
    ) -> "pt.TensorVariable":
        """Apply an in-graph adstock kernel, per-cell for panel data.

        National data (one geo/product cell) keeps the historical 1-D
        convolution — graphs stay bit-identical to pre-panel-fix models. Panel
        data routes through the panel-aware convolution so each cell carries
        over only its own spend history.
        """
        if self.n_cells > 1:
            return parametric_adstock_panel_pt(
                x,
                kind,
                l_max,
                time_idx=self.time_idx,
                cell_idx=self.cell_idx,
                n_periods=self.n_periods,
                n_cells=self.n_cells,
                normalize=normalize,
                **params,
            )
        return parametric_adstock_pt(x, kind, l_max, normalize=normalize, **params)

    def _channel_adstock_apply(
        self, channel_name: str
    ) -> "Callable[[pt.TensorVariable], pt.TensorVariable]":
        """Return a closure applying this channel's in-graph adstock kernel.

        The kernel's random variables (decay/delay/shape/scale) are created once
        here; the returned closure re-applies that *same* kernel to any input
        series. This lets the model adstock both the observed spend and a
        perturbed (e.g. experiment-scaled) spend with shared parameters, which is
        what the marginal-ROAS experiment likelihood needs.
        """
        cfg = self._get_adstock_config(channel_name)
        kind = _ADSTOCK_KIND.get(cfg.type, "geometric")
        l_max = cfg.l_max
        normalize = cfg.normalize

        if kind == "none":
            return lambda x: x

        if kind == "geometric":
            alpha = _sample_from_prior_config(
                f"adstock_alpha_{channel_name}",
                cfg.alpha_prior,
                lambda: pm.Beta(f"adstock_alpha_{channel_name}", alpha=2, beta=2),
            )
            return lambda x: self._apply_adstock_kernel_pt(
                x, "geometric", l_max, normalize, alpha=alpha
            )

        if kind == "delayed":
            alpha = _sample_from_prior_config(
                f"adstock_alpha_{channel_name}",
                cfg.alpha_prior,
                lambda: pm.Beta(f"adstock_alpha_{channel_name}", alpha=2, beta=2),
            )
            theta = _sample_from_prior_config(
                f"adstock_theta_{channel_name}",
                cfg.theta_prior,
                lambda: pm.HalfNormal(f"adstock_theta_{channel_name}", sigma=2.0),
            )
            return lambda x: self._apply_adstock_kernel_pt(
                x, "delayed", l_max, normalize, alpha=alpha, theta=theta
            )

        # weibull
        shape = _sample_from_prior_config(
            f"adstock_shape_{channel_name}",
            cfg.shape_prior,
            lambda: pm.Gamma(f"adstock_shape_{channel_name}", alpha=2.0, beta=1.0),
        )
        # Fallback scale prior follows AdstockConfig.weibull()'s l_max scaling
        # rule (mean max(2, (l_max - 9) / 2)): a fixed mean-2 prior puts no mass
        # where long-window kernels live and causes divergence storms.
        scale_mean = max(2.0, (l_max - 9) / 2)
        scale = _sample_from_prior_config(
            f"adstock_scale_{channel_name}",
            cfg.scale_prior,
            lambda: pm.Gamma(
                f"adstock_scale_{channel_name}", alpha=2.0, beta=2.0 / scale_mean
            ),
        )
        return lambda x: self._apply_adstock_kernel_pt(
            x, "weibull", l_max, normalize, shape=shape, scale=scale
        )

    def _build_channel_adstock(
        self, channel_name: str, x_raw: "pt.TensorVariable"
    ) -> "pt.TensorVariable":
        """Build the in-graph parametric adstock for one channel.

        Reads the channel's :class:`AdstockConfig` (type, l_max, normalize, and
        priors) and estimates a continuous adstock kernel, supporting geometric,
        delayed, and Weibull carryover shapes.
        """
        return self._channel_adstock_apply(channel_name)(x_raw)

    def compute_adstock_curves(
        self, l_max_default: int = 8
    ) -> dict[str, np.ndarray] | None:
        """Posterior-mean adstock kernel (lag weights) per channel.

        Returns the *actual* estimated carryover shape for reporting, so a
        delayed or Weibull channel is rendered with its true (possibly humped)
        kernel rather than being forced to a geometric curve. Requires a fitted
        model; returns None before fitting.
        """
        if self._trace is None or not hasattr(self._trace, "posterior"):
            return None

        posterior = self._trace.posterior
        curves: dict[str, np.ndarray] = {}

        for ch in self.channel_names:
            if self.use_parametric_adstock:
                cfg = self._get_adstock_config(ch)
                kind = _ADSTOCK_KIND.get(cfg.type, "geometric")
                l_max = cfg.l_max
                if kind == "none":
                    curves[ch] = adstock_weights("none", l_max)
                elif kind in ("geometric", "delayed"):
                    name = f"adstock_alpha_{ch}"
                    if name not in posterior:
                        continue
                    alpha = float(posterior[name].values.mean())
                    theta = 0.0
                    if kind == "delayed" and f"adstock_theta_{ch}" in posterior:
                        theta = float(posterior[f"adstock_theta_{ch}"].values.mean())
                    curves[ch] = adstock_weights(
                        kind, l_max, alpha=alpha, theta=theta, normalize=cfg.normalize
                    )
                else:  # weibull
                    if (
                        f"adstock_shape_{ch}" not in posterior
                        or f"adstock_scale_{ch}" not in posterior
                    ):
                        continue
                    shape = float(posterior[f"adstock_shape_{ch}"].values.mean())
                    scale = float(posterior[f"adstock_scale_{ch}"].values.mean())
                    curves[ch] = adstock_weights(
                        "weibull",
                        l_max,
                        shape=shape,
                        scale=scale,
                        normalize=cfg.normalize,
                    )
            else:
                # Legacy two-point mix: reconstruct the effective kernel as the
                # learned convex blend of the low/high fixed-alpha geometrics.
                name = f"adstock_{ch}"
                if name not in posterior:
                    continue
                mix = float(posterior[name].values.mean())
                lags = np.arange(l_max_default)
                a_low = self.adstock_alphas[0]
                a_high = self.adstock_alphas[-1]
                w = (1 - mix) * (a_low**lags) + mix * (a_high**lags)
                total = w.sum()
                curves[ch] = w / total if total > 0 else w

        return curves or None

    def _build_model(self) -> pm.Model:
        """Build the PyMC model with Data for prediction support."""
        coords = self._build_coords()

        if self.use_parametric_adstock:
            X_media_raw_norm = self._prepare_raw_media_for_model()
        else:
            X_adstock_low, X_adstock_high = self._prepare_media_data_for_model()

        with pm.Model(coords=coords) as model:
            # MUTABLE DATA
            if self.use_parametric_adstock:
                X_media_raw_data = pm.Data(
                    "X_media_raw", X_media_raw_norm, dims=("obs", "channel")
                )
            else:
                X_media_low_data = pm.Data(
                    "X_media_low", X_adstock_low, dims=("obs", "channel")
                )
                X_media_high_data = pm.Data(
                    "X_media_high", X_adstock_high, dims=("obs", "channel")
                )

            if self.X_controls is not None:
                X_controls_data = pm.Data(
                    "X_controls", self.X_controls, dims=("obs", "control")
                )

            time_idx_data = pm.Data("time_idx", self.time_idx)
            geo_idx_data = pm.Data("geo_idx", self.geo_idx)
            product_idx_data = pm.Data("product_idx", self.product_idx)

            if self.use_parametric_adstock:
                n_obs_data = X_media_raw_data.shape[0]
            else:
                n_obs_data = X_media_low_data.shape[0]

            # INTERCEPT (getattr defaults keep configs pickled before these
            # fields existed loadable)
            intercept = pm.Normal(
                "intercept",
                mu=getattr(self.model_config, "intercept_prior_mu", 0.0),
                sigma=getattr(self.model_config, "intercept_prior_sigma", 0.5),
            )
            pm.Deterministic(
                "intercept_component", intercept + pt.zeros(n_obs_data), dims="obs"
            )

            # TREND
            trend = self._build_trend_component(model, time_idx_data)
            pm.Deterministic("trend_component", trend)

            # SEASONALITY
            n_periods = self.n_periods
            seasonality_at_periods = pt.zeros(n_periods)

            for name, features in self.seasonality_features.items():
                n_features = features.shape[1]
                season_coef = pm.Normal(
                    f"season_{name}",
                    mu=0,
                    sigma=self.seasonality_config.prior_sigma_for(name),
                    shape=n_features,
                    dims=f"{name}_fourier",
                )
                features_tensor = pt.as_tensor_variable(features)
                season_effect = pt.dot(features_tensor, season_coef)
                seasonality_at_periods = seasonality_at_periods + season_effect

            seasonality = seasonality_at_periods[time_idx_data]
            pm.Deterministic("seasonality_component", seasonality)
            pm.Deterministic("seasonality_by_period", seasonality_at_periods)

            # GEO EFFECTS
            if self.has_geo and self.hierarchical_config.pool_across_geo:
                geo_sigma = pm.HalfNormal("geo_sigma", sigma=0.3)
                geo_offset = pm.Normal("geo_offset", mu=0, sigma=1, shape=self.n_geos)
                geo_effect = geo_sigma * geo_offset
                geo_contribution = geo_effect[geo_idx_data]
            else:
                geo_contribution = pt.zeros(n_obs_data)
            pm.Deterministic("geo_component", geo_contribution, dims="obs")

            # PRODUCT EFFECTS
            if self.has_product and self.hierarchical_config.pool_across_product:
                product_sigma = pm.HalfNormal("product_sigma", sigma=0.3)
                product_offset = pm.Normal(
                    "product_offset", mu=0, sigma=1, shape=self.n_products
                )
                product_effect = product_sigma * product_offset
                product_contribution = product_effect[product_idx_data]
            else:
                product_contribution = pt.zeros(n_obs_data)
            pm.Deterministic("product_component", product_contribution, dims="obs")

            # MEDIA EFFECTS
            channel_contribs = []
            # Per-channel in-graph handles, captured so an experiment likelihood
            # can re-express each channel's estimand (contribution / ROAS /
            # mROAS) from the *same* beta, saturation, and adstock parameters.
            channel_handles: dict[str, dict[str, Any]] = {}

            for c, channel_name in enumerate(self.channel_names):
                if self.use_parametric_adstock:
                    # Estimate a continuous adstock kernel in-graph (geometric,
                    # delayed, or Weibull) per the channel's AdstockConfig. Keep
                    # the apply-closure so a perturbed spend can be re-adstocked
                    # with the *same* kernel RVs (marginal-ROAS likelihood).
                    adstock_apply = self._channel_adstock_apply(channel_name)
                    x_input = X_media_raw_data[:, c]
                    x_adstocked = adstock_apply(x_input)
                else:
                    # Legacy: blend two fixed-alpha geometric adstocks.
                    adstock_apply = None
                    x_input = None
                    x_low = X_media_low_data[:, c]
                    x_high = X_media_high_data[:, c]
                    adstock_mix = pm.Beta(f"adstock_{channel_name}", alpha=2, beta=2)
                    x_adstocked = (1 - adstock_mix) * x_low + adstock_mix * x_high

                # Saturation: per the channel's SaturationConfig (logistic by
                # default, matching historical behavior bit-for-bit). The same
                # helper is reused for perturbed-spend re-evaluations so the
                # marginal/counterfactual estimands cannot drift.
                sat_kind, sat_params = self._build_channel_saturation(channel_name)
                x_saturated = _apply_saturation_pt(x_adstocked, sat_kind, sat_params)

                # Coefficient (effect size). When an experiment-calibrated
                # ``roi_prior`` is configured for this channel (e.g. derived from
                # a geo-lift test by ``mmm_framework.calibration``), honor it --
                # this is how randomized incrementality evidence enters the
                # likelihood. Otherwise use a Gamma prior placing more mass > 1.0
                # (encouraging ROI > 1x).
                media_cfg = self.mff_config.get_media_config(channel_name)
                roi_prior = getattr(media_cfg, "roi_prior", None)
                beta = _sample_from_prior_config(
                    f"beta_{channel_name}",
                    roi_prior,
                    lambda: pm.Gamma(f"beta_{channel_name}", mu=1.5, sigma=1.0),
                )
                channel_contrib = beta * x_saturated
                channel_contribs.append(channel_contrib)

                channel_handles[channel_name] = {
                    "index": c,
                    "beta": beta,
                    "sat_kind": sat_kind,
                    "sat_params": sat_params,
                    # Back-compat alias (None unless logistic).
                    "sat_lam": sat_params.get("sat_lam"),
                    "channel_contrib": channel_contrib,  # standardized, per-obs
                    "adstock_apply": adstock_apply,  # parametric path only
                    "x_input": x_input,  # normalized raw series (parametric)
                    "x_low": x_low if not self.use_parametric_adstock else None,
                    "x_high": x_high if not self.use_parametric_adstock else None,
                    "adstock_mix": (
                        adstock_mix if not self.use_parametric_adstock else None
                    ),
                }

            media_matrix = pt.stack(channel_contribs, axis=1)
            media_contribution = media_matrix.sum(axis=1)

            pm.Deterministic(
                "channel_contributions", media_matrix, dims=("obs", "channel")
            )
            pm.Deterministic("media_total", media_contribution)

            # EXPERIMENT LIKELIHOODS (incrementality / lift / ROAS calibration)
            # Fold any registered experimental results into the joint posterior
            # as likelihood terms on the model-implied estimand. No-op (and graph
            # byte-identical) when no experiments are registered.
            if self.experiments:
                self._add_experiment_likelihoods(channel_handles)

            # CONTROL EFFECTS
            # ``beta_controls`` (kept for reporting/validation which read it by
            # name): a confounder gets a wide, un-shrunk prior so it is not biased
            # toward zero, while precision controls keep the historical
            # Normal(0, 0.5). When ``control_selection`` is enabled (off by
            # default), the non-confounder controls instead get a horseshoe /
            # spike-slab / LASSO prior so the model selects among them -- the
            # horseshoe needs the observation ``sigma`` for its global scale, so
            # create it here on the selection path (the off path is unchanged).
            sigma = (
                pm.HalfNormal("sigma", sigma=0.5) if self._selection_active() else None
            )
            if self.n_controls > 0:
                beta_controls = self._build_control_betas(sigma)
                control_contribution = pt.dot(X_controls_data, beta_controls)
                # Per-control, per-obs contribution (column c = X[:, c] * beta_c).
                pm.Deterministic(
                    "control_contributions",
                    X_controls_data * beta_controls,
                    dims=("obs", "control"),
                )
            else:
                control_contribution = pt.zeros(n_obs_data)
            pm.Deterministic("controls_total", control_contribution, dims="obs")

            # COMBINE AND LIKELIHOOD
            mu = (
                intercept
                + trend
                + seasonality
                + geo_contribution
                + product_contribution
                + media_contribution
                + control_contribution
            )

            # Off path: create sigma here exactly as before (bit-identical). On
            # the selection path it was already created above for the horseshoe.
            if sigma is None:
                sigma = pm.HalfNormal("sigma", sigma=0.5)
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=self.y, dims="obs")
            pm.Deterministic(
                "y_obs_scaled", y_obs * self.y_std + self.y_mean, dims="obs"
            )

        return model

    # =====================================================================
    # Experiment (incrementality / lift / ROAS) calibration likelihoods
    # =====================================================================

    def _period_to_indices(
        self, test_period: tuple[Any, Any]
    ) -> tuple[int, int] | None:
        """Resolve an experiment window to ``(start_idx, end_idx)`` period codes.

        Accepts dates (parsed against the panel's period axis) or integer period
        indices. Returns the first and last period indices that fall inside the
        date window, or ``None`` when *no* period does (window entirely outside
        the panel, or reversed). The indices index into ``coords.periods`` -- the
        same axis ``self.time_idx`` is built from -- so a date window resolves to
        exactly the observations whose period lies in ``[start, end]``.

        (This is a corrected re-implementation of the validator's date parser,
        which uses ``start_idx == 0`` as a "not found" sentinel and so mis-handles
        a window starting at period 0 and silently maps out-of-range windows onto
        the whole panel.)
        """
        start, end = test_period
        if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)):
            return int(start), int(end)

        try:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
        except Exception:
            return int(start), int(end)

        periods = pd.DatetimeIndex(pd.to_datetime(list(self.panel.coords.periods)))
        in_window = np.asarray((periods >= start_date) & (periods <= end_date))
        matched = np.flatnonzero(in_window)
        if matched.size == 0:
            return None
        return int(matched[0]), int(matched[-1])

    def _experiment_obs_mask(
        self, exp: "ExperimentMeasurement"
    ) -> NDArray[np.bool_] | None:
        """Boolean obs mask for an experiment window (+ optional geo restriction).

        Returns ``None`` (with a warning) when the experiment cannot be located:
        an unparseable window, geos that the model does not have, or a window
        that selects no observations.
        """
        try:
            indices = self._period_to_indices(exp.test_period)
        except (ValueError, TypeError) as exc:
            warnings.warn(
                f"Experiment on {exp.channel!r} skipped: cannot parse period "
                f"{exp.test_period!r} ({exc}).",
                stacklevel=3,
            )
            return None

        if indices is None:
            warnings.warn(
                f"Experiment on {exp.channel!r} skipped: window "
                f"{exp.test_period!r} falls outside the panel's period range.",
                stacklevel=3,
            )
            return None

        start_idx, end_idx = indices
        mask = (self.time_idx >= start_idx) & (self.time_idx <= end_idx)

        if exp.holdout_regions:
            if not self.has_geo:
                # A geo-restricted measurement cannot be represented on a model
                # with no geo dimension (it would be fit against the national
                # contribution, biasing it). Skip rather than silently mis-scale.
                warnings.warn(
                    f"Experiment on {exp.channel!r} skipped: holdout_regions "
                    f"{exp.holdout_regions} require a geo model but this model has "
                    "no geo dimension.",
                    stacklevel=3,
                )
                return None
            else:
                name_to_idx = {g: j for j, g in enumerate(self.geo_names)}
                unknown = [g for g in exp.holdout_regions if g not in name_to_idx]
                if unknown:
                    warnings.warn(
                        f"Experiment on {exp.channel!r} skipped: unknown "
                        f"holdout_regions {unknown}.",
                        stacklevel=3,
                    )
                    return None
                hold_idx = [name_to_idx[g] for g in exp.holdout_regions]
                mask = mask & np.isin(self.geo_idx, hold_idx)

        if not mask.any():
            warnings.warn(
                f"Experiment on {exp.channel!r} skipped: window "
                f"{exp.test_period!r} selects no observations.",
                stacklevel=3,
            )
            return None
        return mask

    def _perturbed_contribution_sum(
        self,
        handle: dict[str, Any],
        mask: NDArray[np.bool_],
        mask_idx: np.ndarray,
        lift: float,
    ) -> "pt.TensorVariable":
        """Standardized contribution sum over the window with spend scaled by lift.

        Re-evaluates the channel's adstock+saturation at spend scaled by
        ``(1 + lift)`` *inside the window only*, reusing the channel's existing
        ``beta``, saturation and adstock RVs so the marginal effect is a genuine
        function of the fitted parameters. Adstock is linear, so scaling the
        normalized input is exact. The sum is taken over the same window mask the
        observed contribution uses (carryover landing after the window is not
        counted -- matching ``compute_marginal_contributions``).
        """
        ch_idx = handle["index"]
        beta = handle["beta"]

        if self.use_parametric_adstock:
            pert_mult = np.ones(self.n_obs, dtype=np.float64)
            pert_mult[mask] = 1.0 + lift
            x_pert = handle["x_input"] * pt.as_tensor_variable(pert_mult)
            a_pert = handle["adstock_apply"](x_pert)
        else:
            # Legacy two-alpha blend: re-adstock the perturbed raw series in numpy
            # (exact, adstock is linear) and re-blend with the same learned mix RV,
            # normalizing by the same per-channel max the base model used.
            x_media_pert = self.X_media_raw.copy()
            x_media_pert[mask, ch_idx] *= 1.0 + lift
            max_val = self._media_max[self.channel_names[ch_idx]] + 1e-8
            alpha_low = self.adstock_alphas[0]
            alpha_high = self.adstock_alphas[-1]
            pert_low = (
                geometric_adstock_2d(x_media_pert, alpha_low)[:, ch_idx] / max_val
            )
            pert_high = (
                geometric_adstock_2d(x_media_pert, alpha_high)[:, ch_idx] / max_val
            )
            mix = handle["adstock_mix"]
            a_pert = (1 - mix) * pt.as_tensor_variable(pert_low) + mix * (
                pt.as_tensor_variable(pert_high)
            )

        sat_pert = _apply_saturation_pt(
            a_pert, handle["sat_kind"], handle["sat_params"]
        )
        contrib_pert = beta * sat_pert
        return contrib_pert[mask_idx].sum()

    def _add_experiment_likelihoods(self, channel_handles: dict[str, dict]) -> None:
        """Fold registered experiments into the graph as likelihood terms.

        Called inside the ``pm.Model`` context of :meth:`_build_model`. Each
        experiment's model-implied estimand (contribution / ROAS / marginal ROAS)
        is built from the channel's in-graph parameters and compared to the
        measured value via :func:`mmm_framework.calibration.likelihood.attach_experiment_likelihood`.
        Experiments that cannot be located or scaled are skipped with a warning
        rather than aborting the build.
        """
        from ..calibration.likelihood import (
            ExperimentEstimand,
            attach_experiment_likelihood,
        )

        used_names: set[str] = set()
        for i, exp in enumerate(self.experiments):
            if exp.channel not in channel_handles:
                warnings.warn(
                    f"Experiment skipped: unknown channel {exp.channel!r}.",
                    stacklevel=2,
                )
                continue

            mask = self._experiment_obs_mask(exp)
            if mask is None:
                continue
            mask_idx = np.where(mask)[0]
            handle = channel_handles[exp.channel]
            ch_idx = handle["index"]

            contrib_std = handle["channel_contrib"][mask_idx].sum()
            contribution = contrib_std * self.y_std

            if exp.spend is not None:
                spend_window = float(exp.spend)
            else:
                spend_window = float(self.X_media_raw[mask_idx, ch_idx].sum())

            if exp.estimand is ExperimentEstimand.CONTRIBUTION:
                estimand = contribution
            elif exp.estimand is ExperimentEstimand.ROAS:
                if spend_window <= 0:
                    warnings.warn(
                        f"ROAS experiment on {exp.channel!r} skipped: window "
                        "spend is zero (cannot form ROAS denominator).",
                        stacklevel=2,
                    )
                    continue
                estimand = contribution / spend_window
            else:  # MROAS
                if spend_window <= 0:
                    warnings.warn(
                        f"mROAS experiment on {exp.channel!r} skipped: window "
                        "spend is zero (cannot form marginal-spend denominator).",
                        stacklevel=2,
                    )
                    continue
                lift = exp.spend_lift_pct / 100.0
                contrib_pert_std = self._perturbed_contribution_sum(
                    handle, mask, mask_idx, lift
                )
                delta_contribution = (contrib_pert_std - contrib_std) * self.y_std
                spend_delta = lift * spend_window
                estimand = delta_contribution / spend_delta

            # Reserve both the likelihood node name and the companion
            # ``{name}_model_estimand`` Deterministic so two experiments with
            # colliding explicit names can't clobber each other's nodes.
            base_name = exp.default_node_name(i)
            name = base_name
            bump = 2
            while name in used_names or f"{name}_model_estimand" in used_names:
                name = f"{base_name}_{bump}"
                bump += 1
            used_names.add(name)
            used_names.add(f"{name}_model_estimand")
            attach_experiment_likelihood(name, estimand, exp)

    @property
    def model(self) -> pm.Model:
        """Get or build the PyMC model."""
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def get_prior(
        self,
        samples: int = 500,
        random_seed: int | None = None,
    ) -> az.InferenceData:
        """Sample from the prior distribution of the model."""
        with self.model:
            prior_trace = pm.sample_prior_predictive(
                samples=samples, random_seed=random_seed
            )
        return prior_trace

    def fit(
        self,
        draws: int | None = None,
        tune: int | None = None,
        chains: int | None = None,
        target_accept: float | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> MMMResults:
        """
        Fit the model using MCMC.

        Args:
            draws: Number of posterior draws per chain. Default from config.
            tune: Number of tuning samples. Default from config.
            chains: Number of MCMC chains. Default from config.
            target_accept: Target acceptance rate for NUTS. Default 0.9.
            random_seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to pm.sample().

        Returns:
            Fitted model results with diagnostics.
        """
        draws = draws or self.model_config.n_draws
        tune = tune or self.model_config.n_tune
        chains = chains or self.model_config.n_chains
        target_accept = target_accept or 0.9
        random_seed = random_seed or self.model_config.optim_seed

        nuts_sampler = "numpyro" if self.model_config.use_numpyro else "pymc"

        prior = self.get_prior(samples=1000, random_seed=random_seed)

        with self.model:
            trace: az.InferenceData = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                init="adapt_diag",
                **kwargs,
            )
        trace.extend(prior)

        self._trace = trace

        try:
            div_count = int(trace.sample_stats.diverging.sum().values)
        except Exception:
            div_count = 0

        diagnostics = {
            "divergences": div_count,
            "rhat_max": float(az.rhat(trace).max().to_array().max()),
            "ess_bulk_min": float(az.ess(trace, method="bulk").min().to_array().min()),
        }

        results = MMMResults(
            trace=trace,
            model=self.model,
            panel=self.panel,
            diagnostics=diagnostics,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )

        return results

    def predict(
        self,
        X_media: np.ndarray | None = None,
        X_controls: np.ndarray | None = None,
        return_original_scale: bool = True,
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
    ) -> PredictionResults:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        X_media : np.ndarray, optional
            New media data for counterfactual. If None, uses training data.
        X_controls : np.ndarray, optional
            New control data. If None, uses training data.
        return_original_scale : bool
            If True, returns predictions in original scale.
        hdi_prob : float
            HDI probability for uncertainty intervals.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        PredictionResults
            Prediction results with samples and uncertainty bounds.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        with self.model:
            if self.use_parametric_adstock:
                pm.set_data({"X_media_raw": self._prepare_raw_media_for_model(X_media)})
            else:
                X_adstock_low, X_adstock_high = self._prepare_media_data_for_model(
                    X_media
                )
                pm.set_data(
                    {
                        "X_media_low": X_adstock_low,
                        "X_media_high": X_adstock_high,
                    }
                )

            if X_controls is not None and self.n_controls > 0:
                X_controls_std = (X_controls - self.control_mean) / self.control_std
                pm.set_data({"X_controls": X_controls_std})

            pp = pm.sample_posterior_predictive(
                self._trace,
                var_names=["y_obs"],
                random_seed=random_seed,
            )

        y_samples = pp.posterior_predictive["y_obs"].values
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])

        if return_original_scale:
            y_samples = y_samples * self.y_std + self.y_mean

        y_mean = y_samples.mean(axis=0)
        y_std = y_samples.std(axis=0)
        y_hdi_low, y_hdi_high = compute_hdi_bounds(y_samples, hdi_prob=hdi_prob, axis=0)

        return PredictionResults(
            posterior_predictive=pp,
            y_pred_mean=y_mean,
            y_pred_std=y_std,
            y_pred_hdi_low=y_hdi_low,
            y_pred_hdi_high=y_hdi_high,
            y_pred_samples=y_samples,
        )

    def compute_component_decomposition(self) -> ComponentDecomposition:
        """
        Decompose predictions into component contributions.

        Returns
        -------
        ComponentDecomposition
            Full component breakdown in original scale.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        posterior = self._trace.posterior

        def get_mean(var_name: str) -> np.ndarray:
            if var_name in posterior:
                return posterior[var_name].mean(dim=["chain", "draw"]).values
            return np.zeros(self.n_obs)

        intercept_scaled = get_mean("intercept_component")
        trend_scaled = get_mean("trend_component")
        seasonality_scaled = get_mean("seasonality_component")
        media_total_scaled = get_mean("media_total")
        controls_total_scaled = get_mean("controls_total")
        geo_scaled = get_mean("geo_component") if self.has_geo else None
        product_scaled = get_mean("product_component") if self.has_product else None

        channel_contributions_scaled = get_mean("channel_contributions")

        if self.n_controls > 0 and "control_contributions" in posterior:
            control_contributions_scaled = get_mean("control_contributions")
        else:
            control_contributions_scaled = None

        # Convert to original scale
        intercept = intercept_scaled * self.y_std + self.y_mean
        trend = trend_scaled * self.y_std
        seasonality = seasonality_scaled * self.y_std
        media_total = media_total_scaled * self.y_std
        controls_total = controls_total_scaled * self.y_std

        media_by_channel = pd.DataFrame(
            channel_contributions_scaled * self.y_std,
            index=self.panel.index,
            columns=self.channel_names,
        )

        if control_contributions_scaled is not None:
            controls_by_var = pd.DataFrame(
                control_contributions_scaled * self.y_std,
                index=self.panel.index,
                columns=self.control_names,
            )
        else:
            controls_by_var = None

        geo_effects = geo_scaled * self.y_std if geo_scaled is not None else None
        product_effects = (
            product_scaled * self.y_std if product_scaled is not None else None
        )

        total_intercept = float(intercept.sum())
        total_trend = float(trend.sum())
        total_seasonality = float(seasonality.sum())
        total_media = float(media_total.sum())
        total_controls = float(controls_total.sum())
        total_geo = float(geo_effects.sum()) if geo_effects is not None else None
        total_product = (
            float(product_effects.sum()) if product_effects is not None else None
        )

        return ComponentDecomposition(
            intercept=intercept,
            trend=trend,
            seasonality=seasonality,
            media_total=media_total,
            media_by_channel=media_by_channel,
            controls_total=controls_total,
            controls_by_var=controls_by_var,
            geo_effects=geo_effects,
            product_effects=product_effects,
            total_intercept=total_intercept,
            total_trend=total_trend,
            total_seasonality=total_seasonality,
            total_media=total_media,
            total_controls=total_controls,
            total_geo=total_geo,
            total_product=total_product,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )

    def compute_counterfactual_contributions(
        self,
        time_period: tuple[int, int] | None = None,
        channels: list[str] | None = None,
        compute_uncertainty: bool = True,
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
    ) -> ContributionResults:
        """
        Compute channel contributions using counterfactual analysis.

        Parameters
        ----------
        time_period : tuple[int, int], optional
            Time period (start_idx, end_idx) for contribution calculation.
        channels : list[str], optional
            List of channel names to compute contributions for.
        compute_uncertainty : bool
            If True, computes HDI for contributions.
        hdi_prob : float
            Probability mass for HDI calculation.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ContributionResults
            Container with per-observation and total contributions.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        channels = channels or self.channel_names

        invalid_channels = [c for c in channels if c not in self.channel_names]
        if invalid_channels:
            raise ValueError(f"Unknown channels: {invalid_channels}")

        time_mask = self._get_time_mask(time_period)

        baseline_pred = self.predict(
            return_original_scale=True,
            hdi_prob=hdi_prob,
            random_seed=random_seed,
        )

        counterfactual_preds = {}

        for channel in channels:
            X_media_counterfactual = self.X_media_raw.copy()
            ch_idx = self.channel_names.index(channel)
            X_media_counterfactual[:, ch_idx] = 0.0

            cf_pred = self.predict(
                X_media=X_media_counterfactual,
                return_original_scale=True,
                hdi_prob=hdi_prob,
                random_seed=random_seed,
            )

            counterfactual_preds[channel] = cf_pred

        contribution_data = {}
        contribution_samples = {}

        for channel in channels:
            cf_pred = counterfactual_preds[channel]
            contrib = baseline_pred.y_pred_mean - cf_pred.y_pred_mean
            contribution_data[channel] = contrib

            if compute_uncertainty:
                contrib_samples = baseline_pred.y_pred_samples - cf_pred.y_pred_samples
                contribution_samples[channel] = contrib_samples

        channel_contributions = pd.DataFrame(
            contribution_data,
            index=self.panel.index,
        )

        if time_period is not None:
            contrib_masked = channel_contributions.iloc[time_mask]
        else:
            contrib_masked = channel_contributions

        total_contributions = contrib_masked.sum()

        total_effect = total_contributions.sum()
        contribution_pct = (
            (total_contributions / total_effect * 100)
            if total_effect != 0
            else total_contributions * 0
        )

        contribution_hdi_low = None
        contribution_hdi_high = None

        if compute_uncertainty:
            hdi_low_values = {}
            hdi_high_values = {}

            for channel in channels:
                samples = contribution_samples[channel]
                if time_period is not None:
                    samples = samples[:, time_mask]

                total_samples = samples.sum(axis=1)

                low, high = compute_hdi_bounds(total_samples, hdi_prob=hdi_prob, axis=0)
                hdi_low_values[channel] = low
                hdi_high_values[channel] = high

            contribution_hdi_low = pd.Series(hdi_low_values)
            contribution_hdi_high = pd.Series(hdi_high_values)

        cf_pred_arrays = {
            ch: pred.y_pred_mean for ch, pred in counterfactual_preds.items()
        }

        return ContributionResults(
            channel_contributions=channel_contributions,
            total_contributions=total_contributions,
            contribution_pct=contribution_pct,
            baseline_prediction=baseline_pred.y_pred_mean,
            counterfactual_predictions=cf_pred_arrays,
            time_period=time_period,
            contribution_hdi_low=contribution_hdi_low,
            contribution_hdi_high=contribution_hdi_high,
        )

    def compute_marginal_contributions(
        self,
        spend_increase_pct: float = 10.0,
        time_period: tuple[int, int] | None = None,
        channels: list[str] | None = None,
        compute_uncertainty: bool = True,
        hdi_prob: float = 0.94,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """Compute marginal contributions for a given spend increase.

        When ``compute_uncertainty`` is True the marginal contribution and
        marginal ROAS are propagated through the full posterior (saturation and
        coefficient uncertainty included) and the returned frame gains HDI
        columns. This addresses critique.md §3.9: the headline efficiency number
        should never be reported as a bare point estimate.

        Uncertainty requires the baseline and increased posterior-predictive
        draws to be *paired* so the sampled observation noise cancels in their
        per-draw difference; a single shared seed is used to guarantee that.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        channels = channels or self.channel_names
        multiplier = 1.0 + spend_increase_pct / 100.0

        time_mask = self._get_time_mask(time_period)

        # Pair the baseline/increased draws. With ``random_seed=None`` each
        # predict() call would re-seed independently and the observation noise
        # would not cancel, inflating the HDI -- so synthesize one shared seed.
        pair_seed = random_seed
        if compute_uncertainty and pair_seed is None:
            pair_seed = int(np.random.default_rng().integers(0, 2**31 - 1))

        baseline_pred = self.predict(random_seed=pair_seed)
        baseline_total = baseline_pred.y_pred_mean[time_mask].sum()
        if compute_uncertainty:
            baseline_samples = baseline_pred.y_pred_samples[:, time_mask].sum(axis=1)

        results = []

        for channel in channels:
            ch_idx = self.channel_names.index(channel)

            X_media_increased = self.X_media_raw.copy()
            X_media_increased[:, ch_idx] *= multiplier

            increased_pred = self.predict(
                X_media=X_media_increased,
                random_seed=pair_seed,
            )
            increased_total = increased_pred.y_pred_mean[time_mask].sum()

            marginal_contrib = increased_total - baseline_total

            current_spend = self.X_media_raw[time_mask, ch_idx].sum()
            spend_increase = current_spend * (multiplier - 1)

            marginal_roas = (
                marginal_contrib / spend_increase if spend_increase > 0 else 0
            )

            row = {
                "Channel": channel,
                "Current Spend": current_spend,
                f"Spend Increase ({spend_increase_pct}%)": spend_increase,
                "Marginal Contribution": marginal_contrib,
                "Marginal ROAS": marginal_roas,
            }

            if compute_uncertainty:
                increased_samples = increased_pred.y_pred_samples[:, time_mask].sum(
                    axis=1
                )
                contrib_samples = increased_samples - baseline_samples
                contrib_low, contrib_high = _hdi_finite(contrib_samples, hdi_prob)
                # Guard the per-draw division: a channel with no spend in the
                # period has spend_increase == 0, which would make every
                # marginal_roas sample inf/nan and poison the HDI.
                if spend_increase > 0:
                    roas_samples = contrib_samples / spend_increase
                    roas_low, roas_high = _hdi_finite(roas_samples, hdi_prob)
                else:
                    roas_low = roas_high = 0.0
                row.update(
                    {
                        "Marginal Contribution HDI Low": contrib_low,
                        "Marginal Contribution HDI High": contrib_high,
                        "Marginal ROAS HDI Low": roas_low,
                        "Marginal ROAS HDI High": roas_high,
                    }
                )

            results.append(row)

        return pd.DataFrame(results)

    def what_if_scenario(
        self,
        spend_changes: dict[str, float],
        time_period: tuple[int, int] | None = None,
        random_seed: int | None = None,
    ) -> dict:
        """Run a what-if scenario with custom spend changes."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        time_mask = self._get_time_mask(time_period)

        baseline_pred = self.predict(random_seed=random_seed)
        baseline_total = baseline_pred.y_pred_mean[time_mask].sum()

        X_media_scenario = self.X_media_raw.copy()

        spend_summary = {}
        for channel, multiplier in spend_changes.items():
            if channel not in self.channel_names:
                raise ValueError(f"Unknown channel: {channel}")

            ch_idx = self.channel_names.index(channel)
            original_spend = X_media_scenario[time_mask, ch_idx].sum()
            X_media_scenario[:, ch_idx] *= multiplier
            new_spend = X_media_scenario[time_mask, ch_idx].sum()

            spend_summary[channel] = {
                "original": original_spend,
                "scenario": new_spend,
                "change": new_spend - original_spend,
                "change_pct": (multiplier - 1) * 100,
            }

        scenario_pred = self.predict(
            X_media=X_media_scenario,
            random_seed=random_seed,
        )
        scenario_total = scenario_pred.y_pred_mean[time_mask].sum()

        outcome_change = scenario_total - baseline_total
        outcome_change_pct = (
            (outcome_change / baseline_total * 100) if baseline_total != 0 else 0
        )

        return {
            "baseline_outcome": baseline_total,
            "scenario_outcome": scenario_total,
            "outcome_change": outcome_change,
            "outcome_change_pct": outcome_change_pct,
            "spend_changes": spend_summary,
            "baseline_prediction": baseline_pred.y_pred_mean,
            "scenario_prediction": scenario_pred.y_pred_mean,
        }

    def sample_channel_contributions(
        self,
        X_media: np.ndarray | None = None,
        max_draws: int | None = None,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Posterior draws of per-channel contributions under a media scenario.

        Evaluates the ``channel_contributions`` deterministic with ``X_media``
        swapped in (training data when ``None``), returning ORIGINAL-scale
        contributions of shape ``(n_draws, n_obs, n_channels)``. Because the
        model is additive in channels, a scenario that changes every channel at
        once still yields each channel's own response — this is what makes
        budget-response curves cheap (one pass per spend level, not per
        channel × level).

        ``max_draws`` thins the trace evenly to at most that many draws per
        chain-flattened posterior — response-curve grids don't need the full
        posterior.
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")

        trace = self._trace
        if max_draws is not None:
            n_chains = trace.posterior.sizes["chain"]
            per_chain = max(1, int(np.ceil(max_draws / n_chains)))
            step = max(1, trace.posterior.sizes["draw"] // per_chain)
            trace = trace.sel(draw=slice(None, None, step))

        with self.model:
            if self.use_parametric_adstock:
                pm.set_data({"X_media_raw": self._prepare_raw_media_for_model(X_media)})
            else:
                X_adstock_low, X_adstock_high = self._prepare_media_data_for_model(
                    X_media
                )
                pm.set_data(
                    {"X_media_low": X_adstock_low, "X_media_high": X_adstock_high}
                )
            pp = pm.sample_posterior_predictive(
                trace,
                var_names=["channel_contributions"],
                random_seed=random_seed,
                progressbar=False,
            )

        contrib = pp.posterior_predictive["channel_contributions"].values
        contrib = contrib.reshape(-1, *contrib.shape[2:])  # (draws, obs, channel)
        return contrib * self.y_std  # contributions scale by y_std (no mean shift)

    def sample_prior_predictive(
        self, samples: int = 500, random_seed: int | None = None
    ) -> az.InferenceData:
        """Sample from prior predictive distribution."""
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
        parameter -- the honest way to tell whether a posterior reflects the *data* or
        merely re-states an informative/constrained prior. See
        :func:`mmm_framework.diagnostics.parameter_learning`.

        Parameters
        ----------
        var_names:
            Parameters to diagnose. ``None`` (default) uses the model's free random
            variables (``beta_*``, ``sat_lam_*``, ``adstock_*``, ``intercept``, ...).
        prior_samples:
            Number of prior draws used to estimate the prior moments/overlap.
        random_seed:
            Seed for the prior draw (reproducibility).
        **kwargs:
            Forwarded to :func:`~mmm_framework.diagnostics.parameter_learning`
            (e.g. ``bins``, ``c_strong``, ``c_weak``, ``ovl_dominated``).

        Returns
        -------
        pandas.DataFrame
            One row per parameter, sorted by ``contraction`` ascending (most
            prior-dominated first).
        """
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        from ..diagnostics import parameter_learning

        if var_names is None:
            var_names = [rv.name for rv in self.model.free_RVs]
        prior = self.sample_prior_predictive(
            samples=prior_samples, random_seed=random_seed
        )
        return parameter_learning(prior, self._trace, var_names=var_names, **kwargs)

    def summary(self, var_names: list[str] | None = None) -> pd.DataFrame:
        """Get posterior summary."""
        if self._trace is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return az.summary(self._trace, var_names=var_names)

    def save(
        self,
        path: str | Path,
        save_trace: bool = True,
        compress: bool = True,
    ) -> None:
        """Save the fitted model to disk."""
        from ..serialization import MMMSerializer

        MMMSerializer.save(self, path, save_trace=save_trace, compress=compress)

    @classmethod
    def load(
        cls,
        path: str | Path,
        panel: PanelDataset,
        rebuild_model: bool = True,
    ) -> BayesianMMM:
        """Load a saved model from disk."""
        from ..serialization import MMMSerializer

        return MMMSerializer.load(path, panel, rebuild_model=rebuild_model)

    def save_trace_only(self, path: str | Path) -> None:
        """Save only the fitted trace to a file."""
        if self._trace is None:
            raise ValueError("No trace to save. Fit the model first.")

        from ..serialization import MMMSerializer

        MMMSerializer.save_trace_only(self._trace, path)

    def load_trace_only(self, path: str | Path) -> None:
        """Load a trace from a file into the current model."""
        from ..serialization import MMMSerializer

        self._trace = MMMSerializer.load_trace_only(path)


__all__ = ["BayesianMMM"]
