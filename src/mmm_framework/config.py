"""
Configuration classes for flexible MMM framework.

Handles variable-dimension MFF data with configurable KPI, media, and control specifications.
Uses Pydantic for validation and type safety.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

# TypeVar for generic config lookup
T = TypeVar("T", bound="VariableConfig")


# =============================================================================
# Enums for dimension and variable types
# =============================================================================


class DimensionType(str, Enum):
    """Standard MFF dimensions."""

    PERIOD = "Period"
    GEOGRAPHY = "Geography"
    PRODUCT = "Product"
    CAMPAIGN = "Campaign"
    OUTLET = "Outlet"
    CREATIVE = "Creative"


class VariableRole(str, Enum):
    """Role of a variable in the model."""

    KPI = "kpi"
    MEDIA = "media"
    CONTROL = "control"
    AUXILIARY = "auxiliary"  # For allocation weights, etc.


class AdstockType(str, Enum):
    """Supported adstock transformations."""

    GEOMETRIC = "geometric"
    WEIBULL = "weibull"
    DELAYED = "delayed"
    NONE = "none"


class SaturationType(str, Enum):
    """Supported saturation transformations."""

    HILL = "hill"
    LOGISTIC = "logistic"
    MICHAELIS_MENTEN = "michaelis_menten"
    TANH = "tanh"
    NONE = "none"


class PriorType(str, Enum):
    """Supported prior distributions."""

    HALF_NORMAL = "HalfNormal"
    NORMAL = "Normal"
    LOG_NORMAL = "LogNormal"
    GAMMA = "Gamma"
    BETA = "Beta"
    TRUNCATED_NORMAL = "TruncatedNormal"
    HALF_STUDENT_T = "HalfStudentT"


# =============================================================================
# Prior configuration
# =============================================================================


class PriorConfig(BaseModel):
    """Configuration for a prior distribution."""

    distribution: PriorType
    params: dict[str, float] = Field(default_factory=dict)
    dims: str | list[str] | None = None

    model_config = {"extra": "forbid"}

    @classmethod
    def half_normal(
        cls, sigma: float = 1.0, dims: str | list[str] | None = None
    ) -> PriorConfig:
        return cls(
            distribution=PriorType.HALF_NORMAL, params={"sigma": sigma}, dims=dims
        )

    @classmethod
    def gamma(
        cls, alpha: float = 2.0, beta: float = 1.0, dims: str | list[str] | None = None
    ) -> PriorConfig:
        return cls(
            distribution=PriorType.GAMMA,
            params={"alpha": alpha, "beta": beta},
            dims=dims,
        )

    @classmethod
    def beta(
        cls, alpha: float = 2.0, beta: float = 2.0, dims: str | list[str] | None = None
    ) -> PriorConfig:
        return cls(
            distribution=PriorType.BETA,
            params={"alpha": alpha, "beta": beta},
            dims=dims,
        )


# =============================================================================
# Transformation configurations
# =============================================================================


class AdstockConfig(BaseModel):
    """Configuration for adstock transformation."""

    type: AdstockType = AdstockType.GEOMETRIC
    l_max: int = Field(default=8, ge=1, le=52, description="Maximum lag weeks")
    normalize: bool = True

    # Prior configs for parameters
    alpha_prior: PriorConfig | None = None  # Decay rate for geometric
    theta_prior: PriorConfig | None = None  # Peak delay for weibull

    model_config = {"extra": "forbid"}

    @classmethod
    def geometric(
        cls, l_max: int = 8, alpha_prior: PriorConfig | None = None
    ) -> AdstockConfig:
        return cls(
            type=AdstockType.GEOMETRIC,
            l_max=l_max,
            alpha_prior=alpha_prior or PriorConfig.beta(alpha=1, beta=3),
        )

    @classmethod
    def none(cls) -> AdstockConfig:
        return cls(type=AdstockType.NONE, l_max=1)


class SaturationConfig(BaseModel):
    """Configuration for saturation transformation."""

    type: SaturationType = SaturationType.HILL

    # Hill function priors
    kappa_prior: PriorConfig | None = None  # Half-saturation point (EC50)
    slope_prior: PriorConfig | None = None  # Curve steepness
    beta_prior: PriorConfig | None = None  # Maximum effect scaling

    # Data-driven bounds for kappa
    kappa_bounds_percentiles: tuple[float, float] = (0.1, 0.9)

    model_config = {"extra": "forbid"}

    @classmethod
    def hill(
        cls,
        kappa_prior: PriorConfig | None = None,
        slope_prior: PriorConfig | None = None,
        beta_prior: PriorConfig | None = None,
    ) -> SaturationConfig:
        return cls(
            type=SaturationType.HILL,
            kappa_prior=kappa_prior or PriorConfig.beta(alpha=2, beta=2),
            slope_prior=slope_prior or PriorConfig.half_normal(sigma=1.5),
            beta_prior=beta_prior or PriorConfig.half_normal(sigma=1.5),
        )

    @classmethod
    def none(cls) -> SaturationConfig:
        return cls(type=SaturationType.NONE)


# =============================================================================
# Variable configurations
# =============================================================================


class VariableConfig(BaseModel):
    """Configuration for a single variable in the MFF."""

    name: str = Field(
        ..., description="Variable name as it appears in VariableName column"
    )
    role: VariableRole
    dimensions: list[DimensionType] = Field(
        default_factory=lambda: [DimensionType.PERIOD],
        description="Dimensions this variable is defined over",
    )

    # Optional metadata
    display_name: str | None = None
    unit: str | None = None

    model_config = {"extra": "forbid"}

    @property
    def dim_names(self) -> list[str]:
        """Get dimension names as strings."""
        return [d.value for d in self.dimensions]

    @property
    def has_geo(self) -> bool:
        return DimensionType.GEOGRAPHY in self.dimensions

    @property
    def has_product(self) -> bool:
        return DimensionType.PRODUCT in self.dimensions


class MediaChannelConfig(VariableConfig):
    """Extended configuration for media channels."""

    role: VariableRole = VariableRole.MEDIA

    # Transformation configs
    adstock: AdstockConfig = Field(default_factory=AdstockConfig.geometric)
    saturation: SaturationConfig = Field(default_factory=SaturationConfig.hill)

    # Coefficient prior (enforces positivity by default)
    coefficient_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig.half_normal(sigma=2.0)
    )

    # Hierarchical grouping (e.g., "social" groups meta, snapchat, twitter)
    parent_channel: str | None = None

    # Split dimensions beyond base dimensions (e.g., Outlet for social platforms)
    split_dimensions: list[DimensionType] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @property
    def is_child_channel(self) -> bool:
        return self.parent_channel is not None

    @property
    def all_dimensions(self) -> list[DimensionType]:
        """All dimensions including splits."""
        return list(set(self.dimensions + self.split_dimensions))


class ControlVariableConfig(VariableConfig):
    """Configuration for control variables."""

    role: VariableRole = VariableRole.CONTROL

    # Allow negative effects for controls (e.g., price)
    allow_negative: bool = True

    # Prior configuration
    coefficient_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig(
            distribution=PriorType.NORMAL, params={"mu": 0, "sigma": 1}
        )
    )

    # For sparse selection (horseshoe-like behavior)
    use_shrinkage: bool = False

    model_config = {"extra": "forbid"}


class KPIConfig(VariableConfig):
    """Configuration for the target KPI variable."""

    role: VariableRole = VariableRole.KPI

    # Log transform for multiplicative model
    log_transform: bool = False

    # Minimum value (for log safety)
    floor_value: float = 1e-6

    model_config = {"extra": "forbid"}


# =============================================================================
# Dimension alignment configuration
# =============================================================================


class AllocationMethod(str, Enum):
    """Methods for allocating national data to sub-dimensions."""

    EQUAL = "equal"  # Equal split across all levels
    POPULATION = "population"  # By population weight
    SALES = "sales"  # By historical sales weight
    CUSTOM = "custom"  # User-provided weights


class DimensionAlignmentConfig(BaseModel):
    """Configuration for aligning variables across different dimensions."""

    # How to allocate national media to geo/product
    geo_allocation: AllocationMethod = AllocationMethod.POPULATION
    product_allocation: AllocationMethod = AllocationMethod.SALES

    # Custom weight variable names (from MFF)
    geo_weight_variable: str | None = None
    product_weight_variable: str | None = None

    # Whether to aggregate up or disaggregate down
    prefer_disaggregation: bool = True

    model_config = {"extra": "forbid"}


# =============================================================================
# Main MFF configuration
# =============================================================================


class MFFColumnConfig(BaseModel):
    """Column name mappings for MFF format."""

    period: str = "Period"
    geography: str = "Geography"
    product: str = "Product"
    campaign: str = "Campaign"
    outlet: str = "Outlet"
    creative: str = "Creative"
    variable_name: str = "VariableName"
    variable_value: str = "VariableValue"

    model_config = {"extra": "forbid"}

    @property
    def dimension_columns(self) -> list[str]:
        """All dimension column names."""
        return [
            self.period,
            self.geography,
            self.product,
            self.campaign,
            self.outlet,
            self.creative,
        ]

    @property
    def all_columns(self) -> list[str]:
        """All expected columns."""
        return self.dimension_columns + [self.variable_name, self.variable_value]


class MFFConfig(BaseModel):
    """Complete configuration for MFF data and model specification."""

    # Column mappings
    columns: MFFColumnConfig = Field(default_factory=MFFColumnConfig)

    # KPI configuration (required)
    kpi: KPIConfig

    # Media channel configurations
    media_channels: list[MediaChannelConfig] = Field(default_factory=list)

    # Control variable configurations
    controls: list[ControlVariableConfig] = Field(default_factory=list)

    # Dimension alignment settings
    alignment: DimensionAlignmentConfig = Field(
        default_factory=DimensionAlignmentConfig
    )

    # Date parsing
    date_format: str = "%Y-%m-%d"
    frequency: Literal["W", "D", "M"] = "W"  # Weekly, daily, monthly

    # Missing value handling
    fill_missing_media: float = 0.0
    fill_missing_controls: float | None = None  # None = forward fill

    model_config = {"extra": "forbid"}

    @property
    def all_variables(self) -> list[VariableConfig]:
        """All configured variables."""
        return [self.kpi] + self.media_channels + self.controls

    @property
    def variable_names(self) -> list[str]:
        """All variable names."""
        return [v.name for v in self.all_variables]

    @property
    def media_names(self) -> list[str]:
        """Media channel names."""
        return [m.name for m in self.media_channels]

    @property
    def control_names(self) -> list[str]:
        """Control variable names."""
        return [c.name for c in self.controls]

    @property
    def target_dimensions(self) -> list[DimensionType]:
        """Dimensions of the target KPI."""
        return self.kpi.dimensions

    def _get_config_by_name(self, configs: list[T], name: str) -> T | None:
        """Generic config lookup by name.

        Parameters
        ----------
        configs : list[T]
            List of config objects to search.
        name : str
            Name to search for.

        Returns
        -------
        T | None
            The config with matching name, or None if not found.
        """
        for config in configs:
            if config.name == name:
                return config
        return None

    def get_media_config(self, name: str) -> MediaChannelConfig | None:
        """Get media channel config by name."""
        return self._get_config_by_name(self.media_channels, name)

    def get_control_config(self, name: str) -> ControlVariableConfig | None:
        """Get control config by name."""
        return self._get_config_by_name(self.controls, name)

    def get_variable_config(self, name: str) -> VariableConfig | None:
        """Get any variable config by name (media, control, or KPI).

        Parameters
        ----------
        name : str
            Variable name to search for.

        Returns
        -------
        VariableConfig | None
            The config with matching name, or None if not found.
        """
        if self.kpi.name == name:
            return self.kpi
        return self._get_config_by_name(
            self.media_channels, name
        ) or self._get_config_by_name(self.controls, name)

    @model_validator(mode="after")
    def validate_dimensions(self) -> MFFConfig:
        """Ensure dimension compatibility across variables."""
        kpi_dims = set(self.kpi.dimensions)

        # Media can be at same or higher level than KPI
        for media in self.media_channels:
            media_dims = set(media.dimensions)
            # Media dims should be subset of or equal to KPI dims
            # (national media is subset of geo-level KPI)
            if not media_dims.issubset(kpi_dims) and not kpi_dims.issubset(media_dims):
                # Allow if they share Period
                if DimensionType.PERIOD not in media_dims.intersection(kpi_dims):
                    raise ValueError(
                        f"Media '{media.name}' dimensions {media.dim_names} "
                        f"incompatible with KPI dimensions {self.kpi.dim_names}"
                    )

        return self

    def get_hierarchical_media_groups(self) -> dict[str, list[str]]:
        """Get parent->children mapping for hierarchical media."""
        groups: dict[str, list[str]] = {}
        for media in self.media_channels:
            if media.parent_channel:
                if media.parent_channel not in groups:
                    groups[media.parent_channel] = []
                groups[media.parent_channel].append(media.name)
        return groups


# =============================================================================
# Model-level configuration
# =============================================================================


class InferenceMethod(str, Enum):
    """Available inference methods."""

    BAYESIAN_PYMC = "bayesian_pymc"
    BAYESIAN_NUMPYRO = "bayesian_numpyro"
    FREQUENTIST_RIDGE = "frequentist_ridge"
    FREQUENTIST_CVXPY = "frequentist_cvxpy"


class ModelSpecification(str, Enum):
    """Model functional form."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class HierarchicalConfig(BaseModel):
    """Configuration for hierarchical/panel model structure."""

    enabled: bool = True

    # Pooling dimensions
    pool_across_geo: bool = True
    pool_across_product: bool = True

    # Parameterization
    use_non_centered: bool = True  # Better for sparse groups
    non_centered_threshold: int = 20  # Min obs for centered

    # Hyperprior settings
    mu_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig(
            distribution=PriorType.NORMAL, params={"mu": 0, "sigma": 1}
        )
    )
    sigma_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig.half_normal(sigma=0.5)
    )

    model_config = {"extra": "forbid"}


class SeasonalityConfig(BaseModel):
    """Configuration for seasonality components."""

    yearly: int | None = 2  # Fourier order for yearly seasonality
    monthly: int | None = None
    weekly: int | None = None

    model_config = {"extra": "forbid"}


class ControlSelectionConfig(BaseModel):
    """Configuration for control variable selection."""

    method: Literal["none", "horseshoe", "spike_slab", "lasso"] = "none"

    # Horseshoe settings
    expected_nonzero: int = 3

    # Regularization strength (for lasso-like)
    regularization: float = 1.0

    model_config = {"extra": "forbid"}


class ModelConfig(BaseModel):
    """Complete model configuration."""

    # Functional form
    specification: ModelSpecification = ModelSpecification.ADDITIVE

    # Inference settings
    inference_method: InferenceMethod = InferenceMethod.BAYESIAN_NUMPYRO

    # MCMC settings (for Bayesian)
    n_chains: int = 4
    n_draws: int = 1000
    n_tune: int = 1000
    target_accept: float = 0.9

    # Hierarchical structure
    hierarchical: HierarchicalConfig = Field(default_factory=HierarchicalConfig)

    # Seasonality
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)

    # Control selection
    control_selection: ControlSelectionConfig = Field(
        default_factory=ControlSelectionConfig
    )

    # Frequentist settings
    ridge_alpha: float = 1.0
    bootstrap_samples: int = 1000

    # Optimization settings (for transformation search)
    optim_maxiter: int = 500
    optim_seed: int | None = 42

    model_config = {"extra": "forbid"}

    @property
    def is_bayesian(self) -> bool:
        return self.inference_method in [
            InferenceMethod.BAYESIAN_PYMC,
            InferenceMethod.BAYESIAN_NUMPYRO,
        ]

    @property
    def use_numpyro(self) -> bool:
        return self.inference_method == InferenceMethod.BAYESIAN_NUMPYRO


# =============================================================================
# Factory functions for common configurations
# =============================================================================


def create_national_media_config(
    name: str,
    adstock_lmax: int = 8,
    display_name: str | None = None,
) -> MediaChannelConfig:
    """Create config for national-level media channel."""
    return MediaChannelConfig(
        name=name,
        display_name=display_name or name,
        dimensions=[DimensionType.PERIOD],
        adstock=AdstockConfig.geometric(l_max=adstock_lmax),
        saturation=SaturationConfig.hill(),
    )


def create_geo_media_config(
    name: str,
    adstock_lmax: int = 8,
    display_name: str | None = None,
) -> MediaChannelConfig:
    """Create config for geo-level media channel."""
    return MediaChannelConfig(
        name=name,
        display_name=display_name or name,
        dimensions=[DimensionType.PERIOD, DimensionType.GEOGRAPHY],
        adstock=AdstockConfig.geometric(l_max=adstock_lmax),
        saturation=SaturationConfig.hill(),
    )


def create_social_platform_configs(
    platforms: list[str],
    parent_name: str = "social",
    adstock_lmax: int = 4,
) -> list[MediaChannelConfig]:
    """Create configs for social media platforms with hierarchical structure."""
    configs = []
    for platform in platforms:
        configs.append(
            MediaChannelConfig(
                name=platform,
                display_name=platform.title(),
                dimensions=[DimensionType.PERIOD],
                split_dimensions=[DimensionType.OUTLET],
                parent_channel=parent_name,
                adstock=AdstockConfig.geometric(l_max=adstock_lmax),
                saturation=SaturationConfig.hill(),
            )
        )
    return configs


def create_simple_mff_config(
    kpi_name: str,
    media_names: list[str],
    control_names: list[str] | None = None,
    kpi_dimensions: list[DimensionType] | None = None,
    multiplicative: bool = False,
) -> MFFConfig:
    """Create a simple MFF config with sensible defaults."""

    kpi_dims = kpi_dimensions or [DimensionType.PERIOD]

    return MFFConfig(
        kpi=KPIConfig(
            name=kpi_name,
            dimensions=kpi_dims,
            log_transform=multiplicative,
        ),
        media_channels=[create_national_media_config(name) for name in media_names],
        controls=[
            ControlVariableConfig(
                name=name,
                dimensions=kpi_dims,
            )
            for name in (control_names or [])
        ],
    )
