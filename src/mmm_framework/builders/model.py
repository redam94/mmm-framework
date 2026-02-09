"""
Model-level configuration builders.

Provides builders for ModelConfig and related configuration objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import (
    AllocationMethod,
    ControlSelectionConfig,
    DimensionAlignmentConfig,
    HierarchicalConfig,
    InferenceMethod,
    ModelConfig,
    ModelSpecification,
    PriorConfig,
    SeasonalityConfig,
)
from .prior import PriorConfigBuilder

if TYPE_CHECKING:
    from typing import Self


class HierarchicalConfigBuilder:
    """
    Builder for HierarchicalConfig objects.

    Examples
    --------
    >>> hierarchical = (HierarchicalConfigBuilder()
    ...     .enabled()
    ...     .pool_across_geo()
    ...     .pool_across_product()
    ...     .with_non_centered_threshold(20)
    ...     .build())
    """

    def __init__(self) -> None:
        self._enabled: bool = True
        self._pool_across_geo: bool = True
        self._pool_across_product: bool = True
        self._use_non_centered: bool = True
        self._non_centered_threshold: int = 20
        self._mu_prior: PriorConfig | None = None
        self._sigma_prior: PriorConfig | None = None

    def enabled(self, enable: bool = True) -> Self:
        """Enable hierarchical modeling."""
        self._enabled = enable
        return self

    def disabled(self) -> Self:
        """Disable hierarchical modeling."""
        self._enabled = False
        return self

    def pool_across_geo(self, pool: bool = True) -> Self:
        """Enable partial pooling across geographies."""
        self._pool_across_geo = pool
        return self

    def pool_across_product(self, pool: bool = True) -> Self:
        """Enable partial pooling across products."""
        self._pool_across_product = pool
        return self

    def no_geo_pooling(self) -> Self:
        """Disable geo pooling (independent effects)."""
        self._pool_across_geo = False
        return self

    def no_product_pooling(self) -> Self:
        """Disable product pooling."""
        self._pool_across_product = False
        return self

    def use_non_centered(self, use: bool = True) -> Self:
        """Use non-centered parameterization."""
        self._use_non_centered = use
        return self

    def use_centered(self) -> Self:
        """Use centered parameterization."""
        self._use_non_centered = False
        return self

    def with_non_centered_threshold(self, threshold: int) -> Self:
        """Set minimum observations for centered parameterization."""
        self._non_centered_threshold = threshold
        return self

    def with_mu_prior(self, prior: PriorConfig) -> Self:
        """Set prior for group mean."""
        self._mu_prior = prior
        return self

    def with_sigma_prior(self, prior: PriorConfig) -> Self:
        """Set prior for group standard deviation."""
        self._sigma_prior = prior
        return self

    def build(self) -> HierarchicalConfig:
        """Build the HierarchicalConfig object."""
        mu_prior = self._mu_prior or PriorConfigBuilder().normal(0, 1).build()
        sigma_prior = self._sigma_prior or PriorConfigBuilder().half_normal(0.5).build()

        return HierarchicalConfig(
            enabled=self._enabled,
            pool_across_geo=self._pool_across_geo,
            pool_across_product=self._pool_across_product,
            use_non_centered=self._use_non_centered,
            non_centered_threshold=self._non_centered_threshold,
            mu_prior=mu_prior,
            sigma_prior=sigma_prior,
        )


class SeasonalityConfigBuilder:
    """
    Builder for SeasonalityConfig objects.

    Examples
    --------
    >>> seasonality = (SeasonalityConfigBuilder()
    ...     .with_yearly(order=2)
    ...     .with_weekly(order=3)
    ...     .build())
    """

    def __init__(self) -> None:
        self._yearly: int | None = 2
        self._monthly: int | None = None
        self._weekly: int | None = None

    def with_yearly(self, order: int = 2) -> Self:
        """Add yearly seasonality with given Fourier order."""
        self._yearly = order
        return self

    def with_monthly(self, order: int = 2) -> Self:
        """Add monthly seasonality."""
        self._monthly = order
        return self

    def with_weekly(self, order: int = 3) -> Self:
        """Add weekly seasonality."""
        self._weekly = order
        return self

    def no_yearly(self) -> Self:
        """Disable yearly seasonality."""
        self._yearly = None
        return self

    def no_seasonality(self) -> Self:
        """Disable all seasonality."""
        self._yearly = None
        self._monthly = None
        self._weekly = None
        return self

    def build(self) -> SeasonalityConfig:
        """Build the SeasonalityConfig object."""
        return SeasonalityConfig(
            yearly=self._yearly,
            monthly=self._monthly,
            weekly=self._weekly,
        )


class ControlSelectionConfigBuilder:
    """
    Builder for ControlSelectionConfig objects.

    Examples
    --------
    >>> selection = (ControlSelectionConfigBuilder()
    ...     .horseshoe(expected_nonzero=3)
    ...     .build())
    """

    def __init__(self) -> None:
        self._method: str = "none"
        self._expected_nonzero: int = 3
        self._regularization: float = 1.0

    def none(self) -> Self:
        """No variable selection (use all controls)."""
        self._method = "none"
        return self

    def horseshoe(self, expected_nonzero: int = 3) -> Self:
        """Use horseshoe prior for sparse selection."""
        self._method = "horseshoe"
        self._expected_nonzero = expected_nonzero
        return self

    def spike_slab(self) -> Self:
        """Use spike-and-slab prior."""
        self._method = "spike_slab"
        return self

    def lasso(self, regularization: float = 1.0) -> Self:
        """Use LASSO-like regularization."""
        self._method = "lasso"
        self._regularization = regularization
        return self

    def with_expected_nonzero(self, n: int) -> Self:
        """Set expected number of nonzero controls."""
        self._expected_nonzero = n
        return self

    def with_regularization(self, strength: float) -> Self:
        """Set regularization strength."""
        self._regularization = strength
        return self

    def build(self) -> ControlSelectionConfig:
        """Build the ControlSelectionConfig object."""
        return ControlSelectionConfig(
            method=self._method,
            expected_nonzero=self._expected_nonzero,
            regularization=self._regularization,
        )


class ModelConfigBuilder:
    """
    Builder for ModelConfig objects.

    Examples
    --------
    >>> model = (ModelConfigBuilder()
    ...     .additive()
    ...     .bayesian_numpyro()
    ...     .with_chains(4)
    ...     .with_draws(2000)
    ...     .with_hierarchical(HierarchicalConfigBuilder().enabled().build())
    ...     .build())
    """

    def __init__(self) -> None:
        self._specification: ModelSpecification = ModelSpecification.ADDITIVE
        self._inference_method: InferenceMethod = InferenceMethod.BAYESIAN_NUMPYRO
        self._n_chains: int = 4
        self._n_draws: int = 1000
        self._n_tune: int = 1000
        self._target_accept: float = 0.9
        self._hierarchical: HierarchicalConfig | None = None
        self._seasonality: SeasonalityConfig | None = None
        self._control_selection: ControlSelectionConfig | None = None
        self._ridge_alpha: float = 1.0
        self._bootstrap_samples: int = 1000
        self._optim_maxiter: int = 500
        self._optim_seed: int | None = 42

    # Model specification
    def additive(self) -> Self:
        """Use additive model specification."""
        self._specification = ModelSpecification.ADDITIVE
        return self

    def multiplicative(self) -> Self:
        """Use multiplicative model specification."""
        self._specification = ModelSpecification.MULTIPLICATIVE
        return self

    # Inference method
    def bayesian_pymc(self) -> Self:
        """Use PyMC for Bayesian inference (CPU)."""
        self._inference_method = InferenceMethod.BAYESIAN_PYMC
        return self

    def bayesian_numpyro(self) -> Self:
        """Use NumPyro for Bayesian inference (JAX, faster)."""
        self._inference_method = InferenceMethod.BAYESIAN_NUMPYRO
        return self

    def frequentist_ridge(self) -> Self:
        """Use Ridge regression (fast, frequentist)."""
        self._inference_method = InferenceMethod.FREQUENTIST_RIDGE
        return self

    def frequentist_cvxpy(self) -> Self:
        """Use CVXPY for constrained optimization."""
        self._inference_method = InferenceMethod.FREQUENTIST_CVXPY
        return self

    # MCMC settings
    def with_chains(self, n: int) -> Self:
        """Set number of MCMC chains."""
        self._n_chains = n
        return self

    def with_draws(self, n: int) -> Self:
        """Set number of posterior draws per chain."""
        self._n_draws = n
        return self

    def with_tune(self, n: int) -> Self:
        """Set number of tuning samples."""
        self._n_tune = n
        return self

    def with_target_accept(self, rate: float) -> Self:
        """Set target acceptance rate for NUTS."""
        if not 0 < rate < 1:
            raise ValueError(f"Target accept must be between 0 and 1, got {rate}")
        self._target_accept = rate
        return self

    # Component configs
    def with_hierarchical(self, config: HierarchicalConfig) -> Self:
        """Set hierarchical configuration."""
        self._hierarchical = config
        return self

    def with_hierarchical_builder(self, builder: HierarchicalConfigBuilder) -> Self:
        """Set hierarchical from builder."""
        self._hierarchical = builder.build()
        return self

    def with_seasonality(self, config: SeasonalityConfig) -> Self:
        """Set seasonality configuration."""
        self._seasonality = config
        return self

    def with_seasonality_builder(self, builder: SeasonalityConfigBuilder) -> Self:
        """Set seasonality from builder."""
        self._seasonality = builder.build()
        return self

    def with_control_selection(self, config: ControlSelectionConfig) -> Self:
        """Set control selection configuration."""
        self._control_selection = config
        return self

    def with_control_selection_builder(
        self, builder: ControlSelectionConfigBuilder
    ) -> Self:
        """Set control selection from builder."""
        self._control_selection = builder.build()
        return self

    # Frequentist settings
    def with_ridge_alpha(self, alpha: float) -> Self:
        """Set Ridge regularization strength."""
        self._ridge_alpha = alpha
        return self

    def with_bootstrap_samples(self, n: int) -> Self:
        """Set number of bootstrap samples for uncertainty."""
        self._bootstrap_samples = n
        return self

    # Optimization settings
    def with_optim_maxiter(self, n: int) -> Self:
        """Set maximum optimization iterations."""
        self._optim_maxiter = n
        return self

    def with_optim_seed(self, seed: int | None) -> Self:
        """Set optimization random seed."""
        self._optim_seed = seed
        return self

    def build(self) -> ModelConfig:
        """Build the ModelConfig object."""
        return ModelConfig(
            specification=self._specification,
            inference_method=self._inference_method,
            n_chains=self._n_chains,
            n_draws=self._n_draws,
            n_tune=self._n_tune,
            target_accept=self._target_accept,
            hierarchical=self._hierarchical or HierarchicalConfigBuilder().build(),
            seasonality=self._seasonality or SeasonalityConfigBuilder().build(),
            control_selection=self._control_selection
            or ControlSelectionConfigBuilder().build(),
            ridge_alpha=self._ridge_alpha,
            bootstrap_samples=self._bootstrap_samples,
            optim_maxiter=self._optim_maxiter,
            optim_seed=self._optim_seed,
        )


class TrendConfigBuilder:
    """
    Builder for TrendConfig objects with support for various trend types.

    Supports:
    - None (intercept only)
    - Linear (simple linear trend)
    - Piecewise (Prophet-style changepoint detection)
    - Spline (B-spline flexible trend)
    - Gaussian Process (HSGP approximation for smooth trends)

    Examples
    --------
    >>> # Simple linear trend
    >>> trend = TrendConfigBuilder().linear().build()

    >>> # Piecewise linear with changepoints
    >>> trend = (TrendConfigBuilder()
    ...     .piecewise()
    ...     .with_n_changepoints(10)
    ...     .with_changepoint_range(0.8)
    ...     .with_changepoint_prior_scale(0.05)
    ...     .build())

    >>> # B-spline trend
    >>> trend = (TrendConfigBuilder()
    ...     .spline()
    ...     .with_n_knots(15)
    ...     .with_spline_degree(3)
    ...     .with_spline_prior_sigma(1.0)
    ...     .build())

    >>> # Gaussian Process trend
    >>> trend = (TrendConfigBuilder()
    ...     .gaussian_process()
    ...     .with_gp_lengthscale(mu=0.3, sigma=0.2)
    ...     .with_gp_amplitude(sigma=0.5)
    ...     .with_gp_n_basis(25)
    ...     .build())
    """

    def __init__(self) -> None:
        from ..model import TrendType

        self._TrendType = TrendType

        # Type
        self._type = TrendType.LINEAR

        # Piecewise parameters
        self._n_changepoints: int = 10
        self._changepoint_range: float = 0.8
        self._changepoint_prior_scale: float = 0.05

        # Spline parameters
        self._n_knots: int = 10
        self._spline_degree: int = 3
        self._spline_prior_sigma: float = 1.0

        # GP parameters
        self._gp_lengthscale_prior_mu: float = 0.3
        self._gp_lengthscale_prior_sigma: float = 0.2
        self._gp_amplitude_prior_sigma: float = 0.5
        self._gp_n_basis: int = 20
        self._gp_c: float = 1.5

        # Linear parameters
        self._growth_prior_mu: float = 0.0
        self._growth_prior_sigma: float = 0.1

    # =========================================================================
    # Trend Type Selection
    # =========================================================================

    def none(self) -> Self:
        """No trend (intercept only)."""
        self._type = self._TrendType.NONE
        return self

    def linear(self) -> Self:
        """Simple linear trend."""
        self._type = self._TrendType.LINEAR
        return self

    def piecewise(self) -> Self:
        """Prophet-style piecewise linear trend with changepoints."""
        self._type = self._TrendType.PIECEWISE
        return self

    def spline(self) -> Self:
        """B-spline flexible trend."""
        self._type = self._TrendType.SPLINE
        return self

    def gaussian_process(self) -> Self:
        """Gaussian Process trend using HSGP approximation."""
        self._type = self._TrendType.GP
        return self

    # Alias for GP
    def gp(self) -> Self:
        """Alias for gaussian_process()."""
        return self.gaussian_process()

    # =========================================================================
    # Piecewise Trend Parameters
    # =========================================================================

    def with_n_changepoints(self, n: int) -> Self:
        """Set number of changepoints for piecewise trend."""
        if n < 0:
            raise ValueError(f"n_changepoints must be non-negative, got {n}")
        self._n_changepoints = n
        return self

    def with_changepoint_range(self, range_pct: float) -> Self:
        """Set range (0-1) for placing changepoints."""
        if not 0 < range_pct <= 1:
            raise ValueError(f"Changepoint range must be in (0, 1], got {range_pct}")
        self._changepoint_range = range_pct
        return self

    def with_changepoint_prior_scale(self, scale: float) -> Self:
        """Set prior scale for changepoint magnitudes."""
        if scale <= 0:
            raise ValueError(f"Changepoint prior scale must be positive, got {scale}")
        self._changepoint_prior_scale = scale
        return self

    # =========================================================================
    # Spline Trend Parameters
    # =========================================================================

    def with_n_knots(self, n: int) -> Self:
        """Set number of knots for spline trend."""
        if n < 1:
            raise ValueError(f"n_knots must be at least 1, got {n}")
        self._n_knots = n
        return self

    def with_spline_degree(self, degree: int) -> Self:
        """Set B-spline degree (default 3 = cubic)."""
        if degree < 1:
            raise ValueError(f"Spline degree must be at least 1, got {degree}")
        self._spline_degree = degree
        return self

    def with_spline_prior_sigma(self, sigma: float) -> Self:
        """Set prior sigma for spline coefficients."""
        if sigma <= 0:
            raise ValueError(f"Spline prior sigma must be positive, got {sigma}")
        self._spline_prior_sigma = sigma
        return self

    # =========================================================================
    # Gaussian Process Parameters
    # =========================================================================

    def with_gp_lengthscale(self, mu: float = 0.3, sigma: float = 0.2) -> Self:
        """Set prior for GP lengthscale."""
        if mu <= 0:
            raise ValueError(f"GP lengthscale mu must be positive, got {mu}")
        if sigma <= 0:
            raise ValueError(f"GP lengthscale sigma must be positive, got {sigma}")
        self._gp_lengthscale_prior_mu = mu
        self._gp_lengthscale_prior_sigma = sigma
        return self

    def with_gp_amplitude(self, sigma: float = 0.5) -> Self:
        """Set prior sigma for GP amplitude."""
        if sigma <= 0:
            raise ValueError(f"GP amplitude sigma must be positive, got {sigma}")
        self._gp_amplitude_prior_sigma = sigma
        return self

    def with_gp_n_basis(self, n: int) -> Self:
        """Set number of basis functions for HSGP approximation."""
        if n < 5:
            raise ValueError(f"GP n_basis should be at least 5, got {n}")
        self._gp_n_basis = n
        return self

    def with_gp_boundary_factor(self, c: float) -> Self:
        """Set boundary factor for HSGP."""
        if c < 1.0:
            raise ValueError(f"GP boundary factor should be >= 1.0, got {c}")
        self._gp_c = c
        return self

    # =========================================================================
    # Linear Trend Parameters
    # =========================================================================

    def with_growth_prior(self, mu: float = 0.0, sigma: float = 0.1) -> Self:
        """Set prior for linear growth rate."""
        self._growth_prior_mu = mu
        self._growth_prior_sigma = sigma
        return self

    # =========================================================================
    # Preset Configurations
    # =========================================================================

    def smooth(self) -> Self:
        """Preset: Very smooth trend (good for long-term patterns)."""
        self._type = self._TrendType.GP
        self._gp_lengthscale_prior_mu = 0.5
        self._gp_lengthscale_prior_sigma = 0.2
        self._gp_amplitude_prior_sigma = 0.3
        self._gp_n_basis = 15
        return self

    def flexible(self) -> Self:
        """Preset: Flexible trend (good for capturing shifts)."""
        self._type = self._TrendType.SPLINE
        self._n_knots = 15
        self._spline_prior_sigma = 1.5
        return self

    def changepoint_detection(self) -> Self:
        """Preset: Good for detecting structural breaks."""
        self._type = self._TrendType.PIECEWISE
        self._n_changepoints = 15
        self._changepoint_range = 0.9
        self._changepoint_prior_scale = 0.1
        return self

    # =========================================================================
    # Build
    # =========================================================================

    def build(self):
        """Build the TrendConfig object."""
        from ..model import TrendConfig

        return TrendConfig(
            type=self._type,
            n_changepoints=self._n_changepoints,
            changepoint_range=self._changepoint_range,
            changepoint_prior_scale=self._changepoint_prior_scale,
            n_knots=self._n_knots,
            spline_degree=self._spline_degree,
            spline_prior_sigma=self._spline_prior_sigma,
            gp_lengthscale_prior_mu=self._gp_lengthscale_prior_mu,
            gp_lengthscale_prior_sigma=self._gp_lengthscale_prior_sigma,
            gp_amplitude_prior_sigma=self._gp_amplitude_prior_sigma,
            gp_n_basis=self._gp_n_basis,
            gp_c=self._gp_c,
            growth_prior_mu=self._growth_prior_mu,
            growth_prior_sigma=self._growth_prior_sigma,
        )


class DimensionAlignmentConfigBuilder:
    """
    Builder for DimensionAlignmentConfig objects.

    Examples
    --------
    >>> alignment = (DimensionAlignmentConfigBuilder()
    ...     .geo_by_population()
    ...     .product_by_sales()
    ...     .build())
    """

    def __init__(self) -> None:
        self._geo_allocation: AllocationMethod = AllocationMethod.POPULATION
        self._product_allocation: AllocationMethod = AllocationMethod.SALES
        self._geo_weight_variable: str | None = None
        self._product_weight_variable: str | None = None
        self._prefer_disaggregation: bool = True

    def geo_equal(self) -> Self:
        """Allocate to geos equally."""
        self._geo_allocation = AllocationMethod.EQUAL
        return self

    def geo_by_population(self) -> Self:
        """Allocate to geos by population."""
        self._geo_allocation = AllocationMethod.POPULATION
        return self

    def geo_by_sales(self) -> Self:
        """Allocate to geos by historical sales."""
        self._geo_allocation = AllocationMethod.SALES
        return self

    def geo_by_custom(self, weight_variable: str) -> Self:
        """Allocate to geos using custom weight variable from MFF."""
        self._geo_allocation = AllocationMethod.CUSTOM
        self._geo_weight_variable = weight_variable
        return self

    def product_equal(self) -> Self:
        """Allocate to products equally."""
        self._product_allocation = AllocationMethod.EQUAL
        return self

    def product_by_sales(self) -> Self:
        """Allocate to products by historical sales."""
        self._product_allocation = AllocationMethod.SALES
        return self

    def product_by_custom(self, weight_variable: str) -> Self:
        """Allocate to products using custom weight variable."""
        self._product_allocation = AllocationMethod.CUSTOM
        self._product_weight_variable = weight_variable
        return self

    def prefer_disaggregation(self, prefer: bool = True) -> Self:
        """Prefer disaggregating national data vs aggregating detailed data."""
        self._prefer_disaggregation = prefer
        return self

    def prefer_aggregation(self) -> Self:
        """Prefer aggregating detailed data vs disaggregating."""
        self._prefer_disaggregation = False
        return self

    def build(self) -> DimensionAlignmentConfig:
        """Build the DimensionAlignmentConfig object."""
        return DimensionAlignmentConfig(
            geo_allocation=self._geo_allocation,
            product_allocation=self._product_allocation,
            geo_weight_variable=self._geo_weight_variable,
            product_weight_variable=self._product_weight_variable,
            prefer_disaggregation=self._prefer_disaggregation,
        )


__all__ = [
    "HierarchicalConfigBuilder",
    "SeasonalityConfigBuilder",
    "ControlSelectionConfigBuilder",
    "ModelConfigBuilder",
    "TrendConfigBuilder",
    "DimensionAlignmentConfigBuilder",
]
