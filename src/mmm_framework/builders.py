"""
Builder classes for MMM configuration objects.

Provides fluent API for constructing complex configuration objects step-by-step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    # Enums
    AdstockType,
    AllocationMethod,
    DimensionType,
    InferenceMethod,
    ModelSpecification,
    PriorType,
    SaturationType,
    VariableRole,
    # Config classes
    AdstockConfig,
    ControlSelectionConfig,
    ControlVariableConfig,
    DimensionAlignmentConfig,
    HierarchicalConfig,
    KPIConfig,
    MediaChannelConfig,
    MFFColumnConfig,
    MFFConfig,
    ModelConfig,
    PriorConfig,
    SaturationConfig,
    SeasonalityConfig,
)

if TYPE_CHECKING:
    from typing import Self


# =============================================================================
# Prior Config Builder
# =============================================================================

class PriorConfigBuilder:
    """
    Builder for PriorConfig objects.
    
    Examples
    --------
    >>> prior = (PriorConfigBuilder()
    ...     .half_normal(sigma=2.0)
    ...     .with_dims("channel")
    ...     .build())
    
    >>> prior = (PriorConfigBuilder()
    ...     .gamma(alpha=2, beta=1)
    ...     .build())
    """
    
    def __init__(self) -> None:
        self._distribution: PriorType | None = None
        self._params: dict[str, float] = {}
        self._dims: str | list[str] | None = None
    
    def half_normal(self, sigma: float = 1.0) -> Self:
        """Set HalfNormal distribution."""
        self._distribution = PriorType.HALF_NORMAL
        self._params = {"sigma": sigma}
        return self
    
    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> Self:
        """Set Normal distribution."""
        self._distribution = PriorType.NORMAL
        self._params = {"mu": mu, "sigma": sigma}
        return self
    
    def log_normal(self, mu: float = 0.0, sigma: float = 1.0) -> Self:
        """Set LogNormal distribution."""
        self._distribution = PriorType.LOG_NORMAL
        self._params = {"mu": mu, "sigma": sigma}
        return self
    
    def gamma(self, alpha: float = 2.0, beta: float = 1.0) -> Self:
        """Set Gamma distribution."""
        self._distribution = PriorType.GAMMA
        self._params = {"alpha": alpha, "beta": beta}
        return self
    
    def beta(self, alpha: float = 2.0, beta: float = 2.0) -> Self:
        """Set Beta distribution."""
        self._distribution = PriorType.BETA
        self._params = {"alpha": alpha, "beta": beta}
        return self
    
    def truncated_normal(
        self, 
        mu: float = 0.0, 
        sigma: float = 1.0, 
        lower: float = 0.0,
        upper: float | None = None,
    ) -> Self:
        """Set TruncatedNormal distribution."""
        self._distribution = PriorType.TRUNCATED_NORMAL
        self._params = {"mu": mu, "sigma": sigma, "lower": lower}
        if upper is not None:
            self._params["upper"] = upper
        return self
    
    def half_student_t(self, nu: float = 3.0, sigma: float = 1.0) -> Self:
        """Set HalfStudentT distribution."""
        self._distribution = PriorType.HALF_STUDENT_T
        self._params = {"nu": nu, "sigma": sigma}
        return self
    
    def with_dims(self, dims: str | list[str]) -> Self:
        """Set dimension(s) for the prior."""
        self._dims = dims
        return self
    
    def with_params(self, **params: float) -> Self:
        """Set additional parameters."""
        self._params.update(params)
        return self
    
    def build(self) -> PriorConfig:
        """Build the PriorConfig object."""
        if self._distribution is None:
            raise ValueError("Distribution not set. Call a distribution method first.")
        
        return PriorConfig(
            distribution=self._distribution,
            params=self._params,
            dims=self._dims,
        )


# =============================================================================
# Adstock Config Builder
# =============================================================================

class AdstockConfigBuilder:
    """
    Builder for AdstockConfig objects.
    
    Examples
    --------
    >>> adstock = (AdstockConfigBuilder()
    ...     .geometric()
    ...     .with_max_lag(8)
    ...     .with_alpha_prior(PriorConfigBuilder().beta(1, 3).build())
    ...     .build())
    """
    
    def __init__(self) -> None:
        self._type: AdstockType = AdstockType.GEOMETRIC
        self._l_max: int = 8
        self._normalize: bool = True
        self._alpha_prior: PriorConfig | None = None
        self._theta_prior: PriorConfig | None = None
    
    def geometric(self) -> Self:
        """Use geometric adstock transformation."""
        self._type = AdstockType.GEOMETRIC
        return self
    
    def weibull(self) -> Self:
        """Use Weibull adstock transformation."""
        self._type = AdstockType.WEIBULL
        return self
    
    def delayed(self) -> Self:
        """Use delayed adstock transformation."""
        self._type = AdstockType.DELAYED
        return self
    
    def none(self) -> Self:
        """Disable adstock transformation."""
        self._type = AdstockType.NONE
        self._l_max = 1
        return self
    
    def with_max_lag(self, l_max: int) -> Self:
        """Set maximum lag weeks (1-52)."""
        if not 1 <= l_max <= 52:
            raise ValueError(f"l_max must be between 1 and 52, got {l_max}")
        self._l_max = l_max
        return self
    
    def with_normalize(self, normalize: bool = True) -> Self:
        """Set whether to normalize adstock weights."""
        self._normalize = normalize
        return self
    
    def with_alpha_prior(self, prior: PriorConfig) -> Self:
        """Set prior for decay rate (geometric adstock)."""
        self._alpha_prior = prior
        return self
    
    def with_theta_prior(self, prior: PriorConfig) -> Self:
        """Set prior for peak delay (Weibull adstock)."""
        self._theta_prior = prior
        return self
    
    def build(self) -> AdstockConfig:
        """Build the AdstockConfig object."""
        return AdstockConfig(
            type=self._type,
            l_max=self._l_max,
            normalize=self._normalize,
            alpha_prior=self._alpha_prior,
            theta_prior=self._theta_prior,
        )


# =============================================================================
# Saturation Config Builder
# =============================================================================

class SaturationConfigBuilder:
    """
    Builder for SaturationConfig objects.
    
    Examples
    --------
    >>> saturation = (SaturationConfigBuilder()
    ...     .hill()
    ...     .with_kappa_prior(PriorConfigBuilder().beta(2, 2).build())
    ...     .with_slope_prior(PriorConfigBuilder().half_normal(1.5).build())
    ...     .with_kappa_bounds(0.1, 0.9)
    ...     .build())
    """
    
    def __init__(self) -> None:
        self._type: SaturationType = SaturationType.HILL
        self._kappa_prior: PriorConfig | None = None
        self._slope_prior: PriorConfig | None = None
        self._beta_prior: PriorConfig | None = None
        self._kappa_bounds: tuple[float, float] = (0.1, 0.9)
    
    def hill(self) -> Self:
        """Use Hill saturation function."""
        self._type = SaturationType.HILL
        return self
    
    def logistic(self) -> Self:
        """Use logistic saturation function."""
        self._type = SaturationType.LOGISTIC
        return self
    
    def michaelis_menten(self) -> Self:
        """Use Michaelis-Menten saturation function."""
        self._type = SaturationType.MICHAELIS_MENTEN
        return self
    
    def tanh(self) -> Self:
        """Use tanh saturation function."""
        self._type = SaturationType.TANH
        return self
    
    def none(self) -> Self:
        """Disable saturation transformation."""
        self._type = SaturationType.NONE
        return self
    
    def with_kappa_prior(self, prior: PriorConfig) -> Self:
        """Set prior for half-saturation point (EC50)."""
        self._kappa_prior = prior
        return self
    
    def with_slope_prior(self, prior: PriorConfig) -> Self:
        """Set prior for curve steepness."""
        self._slope_prior = prior
        return self
    
    def with_beta_prior(self, prior: PriorConfig) -> Self:
        """Set prior for maximum effect scaling."""
        self._beta_prior = prior
        return self
    
    def with_kappa_bounds(self, lower: float, upper: float) -> Self:
        """Set percentile bounds for kappa prior (data-driven)."""
        if not 0 <= lower < upper <= 1:
            raise ValueError(f"Bounds must be 0 <= lower < upper <= 1, got ({lower}, {upper})")
        self._kappa_bounds = (lower, upper)
        return self
    
    def build(self) -> SaturationConfig:
        """Build the SaturationConfig object."""
        return SaturationConfig(
            type=self._type,
            kappa_prior=self._kappa_prior,
            slope_prior=self._slope_prior,
            beta_prior=self._beta_prior,
            kappa_bounds_percentiles=self._kappa_bounds,
        )


# =============================================================================
# Media Channel Config Builder
# =============================================================================

class MediaChannelConfigBuilder:
    """
    Builder for MediaChannelConfig objects.
    
    Examples
    --------
    >>> tv_config = (MediaChannelConfigBuilder("TV")
    ...     .with_dimensions(DimensionType.PERIOD)
    ...     .with_adstock(AdstockConfigBuilder().geometric().with_max_lag(8).build())
    ...     .with_saturation(SaturationConfigBuilder().hill().build())
    ...     .with_positive_prior(sigma=2.0)
    ...     .build())
    
    >>> social_meta = (MediaChannelConfigBuilder("Meta")
    ...     .with_dimensions(DimensionType.PERIOD)
    ...     .with_parent_channel("Social")
    ...     .with_split_dimensions(DimensionType.OUTLET)
    ...     .build())
    """
    
    def __init__(self, name: str) -> None:
        self._name = name
        self._display_name: str | None = None
        self._unit: str | None = None
        self._dimensions: list[DimensionType] = [DimensionType.PERIOD]
        self._adstock: AdstockConfig | None = None
        self._saturation: SaturationConfig | None = None
        self._coefficient_prior: PriorConfig | None = None
        self._parent_channel: str | None = None
        self._split_dimensions: list[DimensionType] = []
    
    def with_display_name(self, name: str) -> Self:
        """Set human-readable display name."""
        self._display_name = name
        return self
    
    def with_unit(self, unit: str) -> Self:
        """Set unit of measurement (e.g., 'USD', 'GRPs')."""
        self._unit = unit
        return self
    
    def with_dimensions(self, *dims: DimensionType) -> Self:
        """Set dimensions this variable is defined over."""
        self._dimensions = list(dims)
        if DimensionType.PERIOD not in self._dimensions:
            self._dimensions.insert(0, DimensionType.PERIOD)
        return self
    
    def national(self) -> Self:
        """Set as national-level (Period only)."""
        self._dimensions = [DimensionType.PERIOD]
        return self
    
    def by_geo(self) -> Self:
        """Set as geo-level (Period + Geography)."""
        self._dimensions = [DimensionType.PERIOD, DimensionType.GEOGRAPHY]
        return self
    
    def by_product(self) -> Self:
        """Set as product-level (Period + Product)."""
        self._dimensions = [DimensionType.PERIOD, DimensionType.PRODUCT]
        return self
    
    def by_geo_and_product(self) -> Self:
        """Set as geo+product level."""
        self._dimensions = [DimensionType.PERIOD, DimensionType.GEOGRAPHY, DimensionType.PRODUCT]
        return self
    
    def with_adstock(self, config: AdstockConfig) -> Self:
        """Set adstock configuration."""
        self._adstock = config
        return self
    
    def with_adstock_builder(self, builder: AdstockConfigBuilder) -> Self:
        """Set adstock from builder."""
        self._adstock = builder.build()
        return self
    
    def with_geometric_adstock(self, l_max: int = 8) -> Self:
        """Convenience: set geometric adstock with max lag."""
        self._adstock = AdstockConfigBuilder().geometric().with_max_lag(l_max).build()
        return self
    
    def with_saturation(self, config: SaturationConfig) -> Self:
        """Set saturation configuration."""
        self._saturation = config
        return self
    
    def with_saturation_builder(self, builder: SaturationConfigBuilder) -> Self:
        """Set saturation from builder."""
        self._saturation = builder.build()
        return self
    
    def with_hill_saturation(self) -> Self:
        """Convenience: set Hill saturation with defaults."""
        self._saturation = SaturationConfigBuilder().hill().build()
        return self
    
    def with_coefficient_prior(self, prior: PriorConfig) -> Self:
        """Set coefficient prior."""
        self._coefficient_prior = prior
        return self
    
    def with_positive_prior(self, sigma: float = 2.0) -> Self:
        """Convenience: set HalfNormal prior for positive effects."""
        self._coefficient_prior = PriorConfigBuilder().half_normal(sigma).build()
        return self
    
    def with_parent_channel(self, parent: str) -> Self:
        """Set parent channel for hierarchical grouping."""
        self._parent_channel = parent
        return self
    
    def with_split_dimensions(self, *dims: DimensionType) -> Self:
        """Set additional split dimensions (e.g., Outlet for platforms)."""
        self._split_dimensions = list(dims)
        return self
    
    def build(self) -> MediaChannelConfig:
        """Build the MediaChannelConfig object."""
        # Set defaults if not specified
        adstock = self._adstock or AdstockConfigBuilder().geometric().with_max_lag(8).build()
        saturation = self._saturation or SaturationConfigBuilder().hill().build()
        coefficient_prior = self._coefficient_prior or PriorConfigBuilder().half_normal(2.0).build()
        
        return MediaChannelConfig(
            name=self._name,
            display_name=self._display_name,
            unit=self._unit,
            dimensions=self._dimensions,
            adstock=adstock,
            saturation=saturation,
            coefficient_prior=coefficient_prior,
            parent_channel=self._parent_channel,
            split_dimensions=self._split_dimensions,
        )


# =============================================================================
# Control Variable Config Builder
# =============================================================================

class ControlVariableConfigBuilder:
    """
    Builder for ControlVariableConfig objects.
    
    Examples
    --------
    >>> price = (ControlVariableConfigBuilder("Price")
    ...     .with_dimensions(DimensionType.PERIOD)
    ...     .allow_negative()
    ...     .with_normal_prior(mu=0, sigma=1)
    ...     .build())
    
    >>> distribution = (ControlVariableConfigBuilder("Distribution")
    ...     .national()
    ...     .with_shrinkage()
    ...     .build())
    """
    
    def __init__(self, name: str) -> None:
        self._name = name
        self._display_name: str | None = None
        self._unit: str | None = None
        self._dimensions: list[DimensionType] = [DimensionType.PERIOD]
        self._allow_negative: bool = True
        self._coefficient_prior: PriorConfig | None = None
        self._use_shrinkage: bool = False
    
    def with_display_name(self, name: str) -> Self:
        """Set human-readable display name."""
        self._display_name = name
        return self
    
    def with_unit(self, unit: str) -> Self:
        """Set unit of measurement."""
        self._unit = unit
        return self
    
    def with_dimensions(self, *dims: DimensionType) -> Self:
        """Set dimensions this variable is defined over."""
        self._dimensions = list(dims)
        if DimensionType.PERIOD not in self._dimensions:
            self._dimensions.insert(0, DimensionType.PERIOD)
        return self
    
    def national(self) -> Self:
        """Set as national-level (Period only)."""
        self._dimensions = [DimensionType.PERIOD]
        return self
    
    def by_geo(self) -> Self:
        """Set as geo-level."""
        self._dimensions = [DimensionType.PERIOD, DimensionType.GEOGRAPHY]
        return self
    
    def by_product(self) -> Self:
        """Set as product-level."""
        self._dimensions = [DimensionType.PERIOD, DimensionType.PRODUCT]
        return self
    
    def allow_negative(self, allow: bool = True) -> Self:
        """Allow negative coefficient (e.g., for price)."""
        self._allow_negative = allow
        return self
    
    def positive_only(self) -> Self:
        """Constrain coefficient to be positive."""
        self._allow_negative = False
        return self
    
    def with_coefficient_prior(self, prior: PriorConfig) -> Self:
        """Set coefficient prior."""
        self._coefficient_prior = prior
        return self
    
    def with_normal_prior(self, mu: float = 0.0, sigma: float = 1.0) -> Self:
        """Convenience: set Normal prior."""
        self._coefficient_prior = PriorConfigBuilder().normal(mu, sigma).build()
        return self
    
    def with_shrinkage(self, use: bool = True) -> Self:
        """Enable shrinkage prior for variable selection."""
        self._use_shrinkage = use
        return self
    
    def build(self) -> ControlVariableConfig:
        """Build the ControlVariableConfig object."""
        coefficient_prior = self._coefficient_prior
        if coefficient_prior is None:
            if self._allow_negative:
                coefficient_prior = PriorConfigBuilder().normal(0, 1).build()
            else:
                coefficient_prior = PriorConfigBuilder().half_normal(1).build()
        
        return ControlVariableConfig(
            name=self._name,
            display_name=self._display_name,
            unit=self._unit,
            dimensions=self._dimensions,
            allow_negative=self._allow_negative,
            coefficient_prior=coefficient_prior,
            use_shrinkage=self._use_shrinkage,
        )


# =============================================================================
# KPI Config Builder
# =============================================================================

class KPIConfigBuilder:
    """
    Builder for KPIConfig objects.
    
    Examples
    --------
    >>> kpi = (KPIConfigBuilder("Sales")
    ...     .with_dimensions(DimensionType.PERIOD, DimensionType.GEOGRAPHY)
    ...     .multiplicative()
    ...     .build())
    """
    
    def __init__(self, name: str) -> None:
        self._name = name
        self._display_name: str | None = None
        self._unit: str | None = None
        self._dimensions: list[DimensionType] = [DimensionType.PERIOD]
        self._log_transform: bool = False
        self._floor_value: float = 1e-6
    
    def with_display_name(self, name: str) -> Self:
        """Set human-readable display name."""
        self._display_name = name
        return self
    
    def with_unit(self, unit: str) -> Self:
        """Set unit of measurement."""
        self._unit = unit
        return self
    
    def with_dimensions(self, *dims: DimensionType) -> Self:
        """Set dimensions for the KPI."""
        self._dimensions = list(dims)
        if DimensionType.PERIOD not in self._dimensions:
            self._dimensions.insert(0, DimensionType.PERIOD)
        return self
    
    def national(self) -> Self:
        """Set as national-level (Period only)."""
        self._dimensions = [DimensionType.PERIOD]
        return self
    
    def by_geo(self) -> Self:
        """Set as geo-level."""
        self._dimensions = [DimensionType.PERIOD, DimensionType.GEOGRAPHY]
        return self
    
    def by_product(self) -> Self:
        """Set as product-level."""
        self._dimensions = [DimensionType.PERIOD, DimensionType.PRODUCT]
        return self
    
    def by_geo_and_product(self) -> Self:
        """Set as geo+product level."""
        self._dimensions = [DimensionType.PERIOD, DimensionType.GEOGRAPHY, DimensionType.PRODUCT]
        return self
    
    def additive(self) -> Self:
        """Use additive model specification (no log transform)."""
        self._log_transform = False
        return self
    
    def multiplicative(self) -> Self:
        """Use multiplicative model specification (log transform)."""
        self._log_transform = True
        return self
    
    def with_floor_value(self, floor: float) -> Self:
        """Set minimum value for log transform safety."""
        self._floor_value = floor
        return self
    
    def build(self) -> KPIConfig:
        """Build the KPIConfig object."""
        return KPIConfig(
            name=self._name,
            display_name=self._display_name,
            unit=self._unit,
            dimensions=self._dimensions,
            log_transform=self._log_transform,
            floor_value=self._floor_value,
        )


# =============================================================================
# Dimension Alignment Config Builder
# =============================================================================

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


# =============================================================================
# MFF Column Config Builder
# =============================================================================

class MFFColumnConfigBuilder:
    """
    Builder for MFFColumnConfig objects.
    
    Examples
    --------
    >>> columns = (MFFColumnConfigBuilder()
    ...     .with_period_column("Week")
    ...     .with_geography_column("Region")
    ...     .build())
    """
    
    def __init__(self) -> None:
        self._period: str = "Period"
        self._geography: str = "Geography"
        self._product: str = "Product"
        self._campaign: str = "Campaign"
        self._outlet: str = "Outlet"
        self._creative: str = "Creative"
        self._variable_name: str = "VariableName"
        self._variable_value: str = "VariableValue"
    
    def with_period_column(self, name: str) -> Self:
        """Set period/date column name."""
        self._period = name
        return self
    
    def with_geography_column(self, name: str) -> Self:
        """Set geography column name."""
        self._geography = name
        return self
    
    def with_product_column(self, name: str) -> Self:
        """Set product column name."""
        self._product = name
        return self
    
    def with_campaign_column(self, name: str) -> Self:
        """Set campaign column name."""
        self._campaign = name
        return self
    
    def with_outlet_column(self, name: str) -> Self:
        """Set outlet column name."""
        self._outlet = name
        return self
    
    def with_creative_column(self, name: str) -> Self:
        """Set creative column name."""
        self._creative = name
        return self
    
    def with_variable_name_column(self, name: str) -> Self:
        """Set variable name column name."""
        self._variable_name = name
        return self
    
    def with_variable_value_column(self, name: str) -> Self:
        """Set variable value column name."""
        self._variable_value = name
        return self
    
    def build(self) -> MFFColumnConfig:
        """Build the MFFColumnConfig object."""
        return MFFColumnConfig(
            period=self._period,
            geography=self._geography,
            product=self._product,
            campaign=self._campaign,
            outlet=self._outlet,
            creative=self._creative,
            variable_name=self._variable_name,
            variable_value=self._variable_value,
        )


# =============================================================================
# Hierarchical Config Builder
# =============================================================================

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


# =============================================================================
# Seasonality Config Builder
# =============================================================================

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


# =============================================================================
# Control Selection Config Builder
# =============================================================================

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


# =============================================================================
# Model Config Builder
# =============================================================================

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
    
    def with_control_selection_builder(self, builder: ControlSelectionConfigBuilder) -> Self:
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
            control_selection=self._control_selection or ControlSelectionConfigBuilder().build(),
            ridge_alpha=self._ridge_alpha,
            bootstrap_samples=self._bootstrap_samples,
            optim_maxiter=self._optim_maxiter,
            optim_seed=self._optim_seed,
        )


# =============================================================================
# Trend Config Builder
# =============================================================================

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
        from .model import TrendType
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
        """
        Set number of changepoints for piecewise trend.
        
        Parameters
        ----------
        n : int
            Number of potential changepoints. These are placed uniformly
            in the first `changepoint_range` proportion of the data.
        """
        if n < 0:
            raise ValueError(f"n_changepoints must be non-negative, got {n}")
        self._n_changepoints = n
        return self
    
    def with_changepoint_range(self, range_pct: float) -> Self:
        """
        Set range (0-1) for placing changepoints.
        
        Parameters
        ----------
        range_pct : float
            Proportion of the time series to place changepoints in.
            Default 0.8 means changepoints only in first 80% of data.
            This prevents overfitting to recent data.
        """
        if not 0 < range_pct <= 1:
            raise ValueError(f"Changepoint range must be in (0, 1], got {range_pct}")
        self._changepoint_range = range_pct
        return self
    
    def with_changepoint_prior_scale(self, scale: float) -> Self:
        """
        Set prior scale for changepoint magnitudes.
        
        Parameters
        ----------
        scale : float
            Scale parameter for Laplace prior on changepoint magnitudes.
            Smaller values = more regularization (smoother trends).
            Typical values: 0.01 (very smooth) to 0.5 (flexible).
        """
        if scale <= 0:
            raise ValueError(f"Changepoint prior scale must be positive, got {scale}")
        self._changepoint_prior_scale = scale
        return self
    
    # =========================================================================
    # Spline Trend Parameters
    # =========================================================================
    
    def with_n_knots(self, n: int) -> Self:
        """
        Set number of knots for spline trend.
        
        Parameters
        ----------
        n : int
            Number of interior knots for B-spline basis.
            More knots = more flexible trend.
            Typical values: 5-20 depending on data length.
        """
        if n < 1:
            raise ValueError(f"n_knots must be at least 1, got {n}")
        self._n_knots = n
        return self
    
    def with_spline_degree(self, degree: int) -> Self:
        """
        Set B-spline degree (default 3 = cubic).
        
        Parameters
        ----------
        degree : int
            Polynomial degree of B-spline:
            - 1: Linear splines (piecewise linear)
            - 2: Quadratic splines
            - 3: Cubic splines (recommended, smooth)
        """
        if degree < 1:
            raise ValueError(f"Spline degree must be at least 1, got {degree}")
        self._spline_degree = degree
        return self
    
    def with_spline_prior_sigma(self, sigma: float) -> Self:
        """
        Set prior sigma for spline coefficients.
        
        Parameters
        ----------
        sigma : float
            Scale for HalfNormal prior on spline coefficient variance.
            Smaller = smoother trends. Larger = more flexible.
        """
        if sigma <= 0:
            raise ValueError(f"Spline prior sigma must be positive, got {sigma}")
        self._spline_prior_sigma = sigma
        return self
    
    # =========================================================================
    # Gaussian Process Parameters
    # =========================================================================
    
    def with_gp_lengthscale(
        self,
        mu: float = 0.3,
        sigma: float = 0.2
    ) -> Self:
        """
        Set prior for GP lengthscale.
        
        Parameters
        ----------
        mu : float
            Prior mean for lengthscale (LogNormal prior).
            Value is in proportion of total time range [0, 1].
            Larger = smoother trends.
            Typical values: 0.1-0.5
        sigma : float
            Prior sigma for lengthscale uncertainty.
        """
        if mu <= 0:
            raise ValueError(f"GP lengthscale mu must be positive, got {mu}")
        if sigma <= 0:
            raise ValueError(f"GP lengthscale sigma must be positive, got {sigma}")
        self._gp_lengthscale_prior_mu = mu
        self._gp_lengthscale_prior_sigma = sigma
        return self
    
    def with_gp_amplitude(self, sigma: float = 0.5) -> Self:
        """
        Set prior sigma for GP amplitude.
        
        Parameters
        ----------
        sigma : float
            HalfNormal sigma for GP output scale.
            Controls how much the trend can vary.
        """
        if sigma <= 0:
            raise ValueError(f"GP amplitude sigma must be positive, got {sigma}")
        self._gp_amplitude_prior_sigma = sigma
        return self
    
    def with_gp_n_basis(self, n: int) -> Self:
        """
        Set number of basis functions for HSGP approximation.
        
        Parameters
        ----------
        n : int
            Number of spectral basis functions.
            More = better approximation but slower.
            Typical values: 15-30 for most time series.
        """
        if n < 5:
            raise ValueError(f"GP n_basis should be at least 5, got {n}")
        self._gp_n_basis = n
        return self
    
    def with_gp_boundary_factor(self, c: float) -> Self:
        """
        Set boundary factor for HSGP.
        
        Parameters
        ----------
        c : float
            Boundary factor for Hilbert space GP approximation.
            Typical values: 1.2-2.0
            Larger = better accuracy at boundaries but more computation.
        """
        if c < 1.0:
            raise ValueError(f"GP boundary factor should be >= 1.0, got {c}")
        self._gp_c = c
        return self
    
    # =========================================================================
    # Linear Trend Parameters
    # =========================================================================
    
    def with_growth_prior(self, mu: float = 0.0, sigma: float = 0.1) -> Self:
        """
        Set prior for linear growth rate.
        
        Parameters
        ----------
        mu : float
            Prior mean for growth rate.
        sigma : float
            Prior sigma for growth rate uncertainty.
        """
        self._growth_prior_mu = mu
        self._growth_prior_sigma = sigma
        return self
    
    # =========================================================================
    # Preset Configurations
    # =========================================================================
    
    def smooth(self) -> Self:
        """
        Preset: Very smooth trend (good for long-term patterns).
        
        Uses GP with long lengthscale.
        """
        self._type = self._TrendType.GP
        self._gp_lengthscale_prior_mu = 0.5
        self._gp_lengthscale_prior_sigma = 0.2
        self._gp_amplitude_prior_sigma = 0.3
        self._gp_n_basis = 15
        return self
    
    def flexible(self) -> Self:
        """
        Preset: Flexible trend (good for capturing shifts).
        
        Uses spline with many knots.
        """
        self._type = self._TrendType.SPLINE
        self._n_knots = 15
        self._spline_prior_sigma = 1.5
        return self
    
    def changepoint_detection(self) -> Self:
        """
        Preset: Good for detecting structural breaks.
        
        Uses piecewise with moderate regularization.
        """
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
        from .model import TrendConfig
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


# =============================================================================
# MFF Config Builder (Main Builder)
# =============================================================================

class MFFConfigBuilder:
    """
    Builder for MFFConfig objects - the main configuration builder.
    
    Examples
    --------
    >>> config = (MFFConfigBuilder()
    ...     .with_kpi(KPIConfigBuilder("Sales").by_geo().build())
    ...     .add_media(MediaChannelConfigBuilder("TV").national().with_geometric_adstock(8).build())
    ...     .add_media(MediaChannelConfigBuilder("Digital").national().with_geometric_adstock(4).build())
    ...     .add_control(ControlVariableConfigBuilder("Price").allow_negative().build())
    ...     .with_alignment(DimensionAlignmentConfigBuilder().geo_by_sales().build())
    ...     .with_date_format("%Y-%m-%d")
    ...     .weekly()
    ...     .build())
    """
    
    def __init__(self) -> None:
        self._columns: MFFColumnConfig | None = None
        self._kpi: KPIConfig | None = None
        self._media_channels: list[MediaChannelConfig] = []
        self._controls: list[ControlVariableConfig] = []
        self._alignment: DimensionAlignmentConfig | None = None
        self._date_format: str = "%Y-%m-%d"
        self._frequency: str = "W"
        self._fill_missing_media: float = 0.0
        self._fill_missing_controls: float | None = None
    
    # Column configuration
    def with_columns(self, config: MFFColumnConfig) -> Self:
        """Set column name mappings."""
        self._columns = config
        return self
    
    def with_columns_builder(self, builder: MFFColumnConfigBuilder) -> Self:
        """Set columns from builder."""
        self._columns = builder.build()
        return self
    
    # KPI configuration
    def with_kpi(self, config: KPIConfig) -> Self:
        """Set KPI configuration."""
        self._kpi = config
        return self
    
    def with_kpi_builder(self, builder: KPIConfigBuilder) -> Self:
        """Set KPI from builder."""
        self._kpi = builder.build()
        return self
    
    def with_kpi_name(self, name: str) -> Self:
        """Convenience: set simple national KPI by name."""
        self._kpi = KPIConfigBuilder(name).national().build()
        return self
    
    # Media channels
    def add_media(self, config: MediaChannelConfig) -> Self:
        """Add a media channel configuration."""
        self._media_channels.append(config)
        return self
    
    def add_media_builder(self, builder: MediaChannelConfigBuilder) -> Self:
        """Add media channel from builder."""
        self._media_channels.append(builder.build())
        return self
    
    def add_media_channels(self, *configs: MediaChannelConfig) -> Self:
        """Add multiple media channel configurations."""
        self._media_channels.extend(configs)
        return self
    
    def add_national_media(self, name: str, adstock_lmax: int = 8) -> Self:
        """Convenience: add national media with defaults."""
        config = (MediaChannelConfigBuilder(name)
                  .national()
                  .with_geometric_adstock(adstock_lmax)
                  .with_hill_saturation()
                  .build())
        self._media_channels.append(config)
        return self
    
    def add_social_platforms(
        self,
        platforms: list[str],
        parent_name: str = "Social",
        adstock_lmax: int = 4,
    ) -> Self:
        """Convenience: add social platforms with hierarchy."""
        for platform in platforms:
            config = (MediaChannelConfigBuilder(platform)
                      .national()
                      .with_geometric_adstock(adstock_lmax)
                      .with_hill_saturation()
                      .with_parent_channel(parent_name)
                      .with_split_dimensions(DimensionType.OUTLET)
                      .build())
            self._media_channels.append(config)
        return self
    
    # Controls
    def add_control(self, config: ControlVariableConfig) -> Self:
        """Add a control variable configuration."""
        self._controls.append(config)
        return self
    
    def add_control_builder(self, builder: ControlVariableConfigBuilder) -> Self:
        """Add control from builder."""
        self._controls.append(builder.build())
        return self
    
    def add_controls(self, *configs: ControlVariableConfig) -> Self:
        """Add multiple control configurations."""
        self._controls.extend(configs)
        return self
    
    def add_price_control(self, name: str = "Price") -> Self:
        """Convenience: add price control (allows negative)."""
        config = (ControlVariableConfigBuilder(name)
                  .national()
                  .allow_negative()
                  .build())
        self._controls.append(config)
        return self
    
    def add_distribution_control(self, name: str = "Distribution") -> Self:
        """Convenience: add distribution/ACV control."""
        config = (ControlVariableConfigBuilder(name)
                  .national()
                  .positive_only()
                  .build())
        self._controls.append(config)
        return self
    
    # Alignment
    def with_alignment(self, config: DimensionAlignmentConfig) -> Self:
        """Set dimension alignment configuration."""
        self._alignment = config
        return self
    
    def with_alignment_builder(self, builder: DimensionAlignmentConfigBuilder) -> Self:
        """Set alignment from builder."""
        self._alignment = builder.build()
        return self
    
    # Date and frequency
    def with_date_format(self, fmt: str) -> Self:
        """Set date parsing format."""
        self._date_format = fmt
        return self
    
    def weekly(self) -> Self:
        """Set weekly frequency."""
        self._frequency = "W"
        return self
    
    def daily(self) -> Self:
        """Set daily frequency."""
        self._frequency = "D"
        return self
    
    def monthly(self) -> Self:
        """Set monthly frequency."""
        self._frequency = "M"
        return self
    
    # Missing value handling
    def with_fill_missing_media(self, value: float) -> Self:
        """Set fill value for missing media (default: 0)."""
        self._fill_missing_media = value
        return self
    
    def with_fill_missing_controls(self, value: float | None) -> Self:
        """Set fill value for missing controls (None = forward fill)."""
        self._fill_missing_controls = value
        return self
    
    def build(self) -> MFFConfig:
        """Build the MFFConfig object."""
        if self._kpi is None:
            raise ValueError("KPI configuration is required. Call with_kpi() first.")
        
        if not self._media_channels:
            raise ValueError("At least one media channel is required. Call add_media() first.")
        
        return MFFConfig(
            columns=self._columns or MFFColumnConfig(),
            kpi=self._kpi,
            media_channels=self._media_channels,
            controls=self._controls,
            alignment=self._alignment or DimensionAlignmentConfig(),
            date_format=self._date_format,
            frequency=self._frequency,
            fill_missing_media=self._fill_missing_media,
            fill_missing_controls=self._fill_missing_controls,
        )