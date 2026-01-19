"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class JobStatus(str, Enum):
    """Status of a background job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataFormat(str, Enum):
    """Supported data formats."""

    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"
    JSON = "json"


class TrendType(str, Enum):
    """Trend model types."""

    NONE = "none"
    LINEAR = "linear"
    PIECEWISE = "piecewise"
    SPLINE = "spline"
    GP = "gaussian_process"


class AllocationMethod(str, Enum):
    """Media allocation methods."""

    EQUAL = "equal"
    POPULATION = "population"
    SALES = "sales"
    CUSTOM = "custom"


# =============================================================================
# Data Models
# =============================================================================


class DataUploadResponse(BaseModel):
    """Response after uploading data."""

    data_id: str
    filename: str
    rows: int
    columns: int
    variables: list[str]
    dimensions: dict[str, list[str]]
    created_at: datetime
    size_bytes: int


class DataInfo(BaseModel):
    """Information about uploaded data."""

    data_id: str
    filename: str
    rows: int
    columns: int
    variables: list[str]
    dimensions: dict[str, list[str]]
    created_at: datetime
    size_bytes: int
    preview: list[dict[str, Any]] | None = None


class DataListResponse(BaseModel):
    """List of uploaded datasets."""

    datasets: list[DataInfo]
    total: int


# =============================================================================
# Configuration Models
# =============================================================================


class PriorConfigSchema(BaseModel):
    """Prior distribution configuration."""

    distribution: Literal[
        "HalfNormal",
        "Normal",
        "LogNormal",
        "Gamma",
        "Beta",
        "TruncatedNormal",
        "HalfStudentT",
    ] = "HalfNormal"
    params: dict[str, float] = Field(default_factory=dict)
    dims: str | list[str] | None = None


class AdstockConfigSchema(BaseModel):
    """Adstock transformation configuration."""

    type: Literal["geometric", "weibull", "delayed", "none"] = "geometric"
    l_max: int = Field(default=8, ge=1, le=52)
    normalize: bool = True
    alpha_prior: PriorConfigSchema | None = None


class SaturationConfigSchema(BaseModel):
    """Saturation transformation configuration."""

    type: Literal["hill", "logistic", "michaelis_menten", "tanh", "none"] = "hill"
    kappa_prior: PriorConfigSchema | None = None
    slope_prior: PriorConfigSchema | None = None
    beta_prior: PriorConfigSchema | None = None
    kappa_bounds_percentiles: tuple[float, float] = (0.1, 0.9)


class MediaChannelSchema(BaseModel):
    """Media channel configuration."""

    name: str
    display_name: str | None = None
    dimensions: list[Literal["Period", "Geography", "Product"]] = ["Period"]
    adstock: AdstockConfigSchema = Field(default_factory=AdstockConfigSchema)
    saturation: SaturationConfigSchema = Field(default_factory=SaturationConfigSchema)
    coefficient_prior: PriorConfigSchema | None = None
    parent_channel: str | None = None


class ControlVariableSchema(BaseModel):
    """Control variable configuration."""

    name: str
    display_name: str | None = None
    dimensions: list[Literal["Period", "Geography", "Product"]] = ["Period"]
    allow_negative: bool = True
    coefficient_prior: PriorConfigSchema | None = None
    use_shrinkage: bool = False


class KPIConfigSchema(BaseModel):
    """KPI configuration."""

    name: str
    display_name: str | None = None
    dimensions: list[Literal["Period", "Geography", "Product"]] = ["Period"]
    log_transform: bool = False
    floor_value: float = 1e-6


class TrendConfigSchema(BaseModel):
    """Trend configuration."""

    type: TrendType = TrendType.LINEAR
    n_changepoints: int = Field(default=10, ge=0, le=50)
    changepoint_range: float = Field(default=0.8, gt=0, le=1)
    changepoint_prior_scale: float = Field(default=0.05, gt=0)
    n_knots: int = Field(default=10, ge=1, le=50)
    spline_degree: int = Field(default=3, ge=1, le=5)
    spline_prior_sigma: float = Field(default=1.0, gt=0)
    gp_lengthscale_prior_mu: float = Field(default=0.3, gt=0)
    gp_lengthscale_prior_sigma: float = Field(default=0.2, gt=0)
    gp_amplitude_prior_sigma: float = Field(default=0.5, gt=0)
    gp_n_basis: int = Field(default=20, ge=5, le=100)
    growth_prior_mu: float = 0.0
    growth_prior_sigma: float = Field(default=0.1, gt=0)


class SeasonalityConfigSchema(BaseModel):
    """Seasonality configuration."""

    yearly: int | None = Field(default=2, ge=0, le=10)
    monthly: int | None = None
    weekly: int | None = None


class HierarchicalConfigSchema(BaseModel):
    """Hierarchical model configuration."""

    enabled: bool = True
    pool_across_geo: bool = True
    pool_across_product: bool = True
    use_non_centered: bool = True
    non_centered_threshold: int = 20


class AlignmentConfigSchema(BaseModel):
    """Dimension alignment configuration."""

    geo_allocation: AllocationMethod = AllocationMethod.EQUAL
    product_allocation: AllocationMethod = AllocationMethod.SALES
    geo_weight_variable: str | None = None
    product_weight_variable: str | None = None


class MFFConfigSchema(BaseModel):
    """Complete MFF model configuration."""

    kpi: KPIConfigSchema | None = None
    media_channels: list[MediaChannelSchema]
    controls: list[ControlVariableSchema] = Field(default_factory=list)
    alignment: AlignmentConfigSchema = Field(default_factory=AlignmentConfigSchema)
    date_format: str = "%Y-%m-%d"
    frequency: Literal["W", "D", "M"] = "W"
    fill_missing_media: float = 0.0
    fill_missing_controls: float | None = None


class ModelConfigSchema(BaseModel):
    """Model fitting configuration."""

    inference_method: Literal["bayesian_pymc", "bayesian_numpyro"] = "bayesian_pymc"
    n_chains: int = Field(default=4, ge=1, le=16)
    n_draws: int = Field(default=1000, ge=100, le=10000)
    n_tune: int = Field(default=1000, ge=100, le=5000)
    target_accept: float = Field(default=0.9, gt=0.5, lt=1.0)
    trend: TrendConfigSchema = Field(default_factory=TrendConfigSchema)
    seasonality: SeasonalityConfigSchema = Field(
        default_factory=SeasonalityConfigSchema
    )
    hierarchical: HierarchicalConfigSchema = Field(
        default_factory=HierarchicalConfigSchema
    )
    random_seed: int | None = 42


class ConfigCreateRequest(BaseModel):
    """Request to create a new configuration."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    mff_config: MFFConfigSchema
    model_settings: ModelConfigSchema = Field(default_factory=ModelConfigSchema)


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""

    name: str | None = None
    description: str | None = None
    mff_config: MFFConfigSchema | None = None
    model_settings: ModelConfigSchema | None = None


class ConfigResponse(BaseModel):
    """Configuration response."""

    config_id: str
    name: str
    description: str | None
    mff_config: MFFConfigSchema
    model_settings: ModelConfigSchema
    created_at: datetime
    updated_at: datetime


class ConfigListResponse(BaseModel):
    """List of configurations."""

    configs: list[ConfigResponse]
    total: int


# =============================================================================
# Model/Job Models
# =============================================================================


class ModelFitRequest(BaseModel):
    """Request to start model fitting."""

    data_id: str
    config_id: str
    name: str | None = None
    description: str | None = None
    # Optional overrides
    n_chains: int | None = None
    n_draws: int | None = None
    n_tune: int | None = None
    random_seed: int | None = None


class ModelInfo(BaseModel):
    """Information about a model."""

    model_id: str
    name: str | None
    description: str | None
    data_id: str
    config_id: str
    status: JobStatus
    progress: float = 0.0
    progress_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    # Diagnostics (available after completion)
    diagnostics: dict[str, Any] | None = None


class ModelListResponse(BaseModel):
    """List of models."""

    models: list[ModelInfo]
    total: int
    skip: int = 0
    limit: int = 20


class ModelResultsResponse(BaseModel):
    """Model results after fitting."""

    model_id: str
    status: JobStatus
    diagnostics: dict[str, Any]
    parameter_summary: list[dict[str, Any]]
    channel_contributions: dict[str, float] | None = None
    component_decomposition: dict[str, Any] | None = None


class ContributionRequest(BaseModel):
    """Request for contribution analysis."""

    time_period: tuple[int, int] | None = None
    channels: list[str] | None = None
    compute_uncertainty: bool = True
    hdi_prob: float = Field(default=0.94, gt=0.5, lt=1.0)


class ContributionResponse(BaseModel):
    """Contribution analysis response."""

    model_id: str
    total_contributions: dict[str, float]
    contribution_pct: dict[str, float]
    contribution_hdi_low: dict[str, float] | None = None
    contribution_hdi_high: dict[str, float] | None = None
    time_period: tuple[int, int] | None = None


class ScenarioRequest(BaseModel):
    """What-if scenario request."""

    spend_changes: dict[str, float]
    time_period: tuple[int, int] | None = None


class ScenarioResponse(BaseModel):
    """Scenario analysis response."""

    model_id: str
    baseline_outcome: float
    scenario_outcome: float
    outcome_change: float
    outcome_change_pct: float
    spend_changes: dict[str, dict[str, float]]


class PredictionRequest(BaseModel):
    """Prediction request."""

    media_spend: dict[str, list[float]] | None = None
    n_periods: int | None = None
    return_samples: bool = False


class PredictionResponse(BaseModel):
    """Prediction response."""

    model_id: str
    y_pred_mean: list[float]
    y_pred_std: list[float]
    y_pred_hdi_low: list[float]
    y_pred_hdi_high: list[float]
    y_pred_samples: list[list[float]] | None = None

# ============================================================================
# Report Models
# ============================================================================
class ReportRequest(BaseModel):
    """Request for HTML report generation."""

    title: str | None = Field(default=None, description="Report title")
    client: str | None = Field(default=None, description="Client name")
    subtitle: str | None = Field(default=None, description="Report subtitle")
    analysis_period: str | None = Field(
        default=None, description="Analysis period description"
    )

    # Section toggles
    include_executive_summary: bool = True
    include_model_fit: bool = True
    include_channel_roi: bool = True
    include_decomposition: bool = True
    include_saturation: bool = True
    include_diagnostics: bool = True
    include_methodology: bool = True

    # Formatting options
    credible_interval: float = Field(default=0.8, gt=0.5, lt=1.0)
    currency_symbol: str = "$"
    currency_scale: float = 1.0

    # Color scheme
    color_scheme: Literal["default", "corporate", "minimal"] = "default"


class ReportResponse(BaseModel):
    """Report generation response."""

    model_id: str
    report_id: str
    status: Literal["generating", "completed", "failed"]
    message: str | None = None
    download_url: str | None = None
    created_at: datetime


class ReportStatusResponse(BaseModel):
    """Report status response."""

    report_id: str
    model_id: str
    status: str
    message: str | None = None
    filename: str | None = None


class ReportListResponse(BaseModel):
    """List of reports."""

    model_id: str
    reports: list[dict]

# =============================================================================
# Generic Response Models
# =============================================================================


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    """Generic error response."""

    success: bool = False
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"]
    version: str
    redis_connected: bool
    worker_active: bool


# =============================================================================
# Extended Model Schemas (NestedMMM, MultivariateMMM, CombinedMMM)
# =============================================================================


class ModelType(str, Enum):
    """Types of MMM models."""

    STANDARD = "standard"
    NESTED = "nested"
    MULTIVARIATE = "multivariate"
    COMBINED = "combined"


class MediatorType(str, Enum):
    """Types of mediators for nested models."""

    FULLY_OBSERVED = "fully_observed"
    PARTIALLY_OBSERVED = "partially_observed"
    AGGREGATED_SURVEY = "aggregated_survey"
    FULLY_LATENT = "fully_latent"


class CrossEffectType(str, Enum):
    """Types of cross-effects between outcomes."""

    CANNIBALIZATION = "cannibalization"
    HALO = "halo"
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class EffectConstraint(str, Enum):
    """Constraint on effect direction."""

    NONE = "none"
    POSITIVE = "positive"
    NEGATIVE = "negative"


# -----------------------------------------------------------------------------
# Mediator Configuration (for NestedMMM)
# -----------------------------------------------------------------------------


class EffectPriorSchema(BaseModel):
    """Prior configuration for effects."""

    constraint: EffectConstraint = EffectConstraint.NONE
    mean: float = 0.0
    sigma: float = 1.0


class MediatorSchema(BaseModel):
    """Configuration for a mediator variable in nested models."""

    name: str = Field(..., description="Mediator name (e.g., 'brand_awareness')")
    display_name: str | None = None
    type: MediatorType = MediatorType.PARTIALLY_OBSERVED
    observation_noise: float = Field(
        default=0.1, ge=0, le=1, description="Observation noise for partially observed mediators"
    )

    # Effect priors
    media_effect_prior: EffectPriorSchema = Field(
        default_factory=lambda: EffectPriorSchema(constraint=EffectConstraint.POSITIVE)
    )
    outcome_effect_prior: EffectPriorSchema = Field(
        default_factory=lambda: EffectPriorSchema(constraint=EffectConstraint.POSITIVE)
    )

    # Transformations
    adstock: AdstockConfigSchema = Field(default_factory=AdstockConfigSchema)
    saturation: SaturationConfigSchema = Field(default_factory=SaturationConfigSchema)

    # Whether media also has direct effect on outcome (bypassing mediator)
    allow_direct_effect: bool = True
    direct_effect_prior: EffectPriorSchema | None = None

    # Data column (for observed mediators)
    data_column: str | None = None


class MediatorChannelMappingSchema(BaseModel):
    """Maps media channels to a mediator."""

    mediator_name: str
    channel_names: list[str]
    share_adstock: bool = False
    share_saturation: bool = False


class NestedModelConfigSchema(BaseModel):
    """Configuration for NestedMMM (mediation model)."""

    mediators: list[MediatorSchema] = Field(
        default_factory=list, description="List of mediator configurations"
    )
    channel_mappings: list[MediatorChannelMappingSchema] = Field(
        default_factory=list, description="Mappings from channels to mediators"
    )


# -----------------------------------------------------------------------------
# Outcome Configuration (for MultivariateMMM)
# -----------------------------------------------------------------------------


class OutcomeSchema(BaseModel):
    """Configuration for an outcome variable in multivariate models."""

    name: str = Field(..., description="Outcome name (e.g., 'sales_product_a')")
    display_name: str | None = None
    data_column: str = Field(..., description="Column name in data")
    log_transform: bool = False

    # Priors
    intercept_prior_sigma: float = Field(default=1.0, gt=0)
    media_effect_constraint: EffectConstraint = EffectConstraint.POSITIVE


class CrossEffectSchema(BaseModel):
    """Configuration for cross-effects between outcomes."""

    source_outcome: str = Field(..., description="Source outcome name")
    target_outcome: str = Field(..., description="Target outcome name")
    effect_type: CrossEffectType = CrossEffectType.CANNIBALIZATION
    effect_prior: EffectPriorSchema = Field(default_factory=EffectPriorSchema)

    # Optional modulation
    promotion_column: str | None = Field(
        default=None, description="Column for promotion-modulated effect"
    )
    lagged: bool = False
    lag_periods: int = Field(default=1, ge=1, le=12)


class MultivariateModelConfigSchema(BaseModel):
    """Configuration for MultivariateMMM (multiple outcomes)."""

    outcomes: list[OutcomeSchema] = Field(
        default_factory=list, description="List of outcome configurations"
    )
    cross_effects: list[CrossEffectSchema] = Field(
        default_factory=list, description="Cross-effects between outcomes"
    )

    # Correlation structure
    lkj_eta: float = Field(
        default=2.0, gt=0, description="LKJ prior eta for outcome correlations"
    )
    share_media_adstock: bool = Field(
        default=True, description="Share adstock parameters across outcomes"
    )
    share_media_saturation: bool = Field(
        default=True, description="Share saturation parameters across outcomes"
    )


# -----------------------------------------------------------------------------
# Combined Model Configuration
# -----------------------------------------------------------------------------


class CombinedModelConfigSchema(BaseModel):
    """Configuration for CombinedMMM (nested + multivariate)."""

    # Nested (mediation) configuration
    nested_config: NestedModelConfigSchema = Field(
        default_factory=NestedModelConfigSchema
    )

    # Multivariate configuration
    multivariate_config: MultivariateModelConfigSchema = Field(
        default_factory=MultivariateModelConfigSchema
    )

    # Mappings from mediators to outcomes
    mediator_outcome_mappings: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps mediator names to outcome names they affect",
    )


# -----------------------------------------------------------------------------
# Extended Model Configuration Wrapper
# -----------------------------------------------------------------------------


class ExtendedMFFConfigSchema(BaseModel):
    """Extended MFF configuration that supports all model types."""

    # Base configuration (same as MFFConfigSchema)
    kpi: KPIConfigSchema | None = Field(
        default=None, description="Primary KPI (for standard/nested models)"
    )
    media_channels: list[MediaChannelSchema] = Field(default_factory=list)
    controls: list[ControlVariableSchema] = Field(default_factory=list)
    alignment: AlignmentConfigSchema = Field(default_factory=AlignmentConfigSchema)
    date_format: str = "%Y-%m-%d"
    frequency: Literal["W", "D", "M"] = "W"
    fill_missing_media: float = 0.0
    fill_missing_controls: float | None = None

    # Extended model configurations
    model_type: ModelType = ModelType.STANDARD
    nested_config: NestedModelConfigSchema | None = None
    multivariate_config: MultivariateModelConfigSchema | None = None
    combined_config: CombinedModelConfigSchema | None = None


class ExtendedConfigCreateRequest(BaseModel):
    """Request to create an extended model configuration."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    mff_config: ExtendedMFFConfigSchema
    model_settings: ModelConfigSchema = Field(default_factory=ModelConfigSchema)


class ExtendedConfigResponse(BaseModel):
    """Extended configuration response."""

    config_id: str
    name: str
    description: str | None
    model_type: ModelType
    mff_config: ExtendedMFFConfigSchema
    model_settings: ModelConfigSchema
    created_at: datetime
    updated_at: datetime


# -----------------------------------------------------------------------------
# Extended Model Fit Request
# -----------------------------------------------------------------------------


class ExtendedModelFitRequest(BaseModel):
    """Request to start extended model fitting."""

    data_id: str
    config_id: str
    name: str | None = None
    description: str | None = None

    # Mediator data (for nested/combined models with observed mediators)
    mediator_data_id: str | None = Field(
        default=None,
        description="Data ID containing mediator observations",
    )

    # Additional outcome data (for multivariate/combined models)
    outcome_data_columns: dict[str, str] | None = Field(
        default=None,
        description="Maps outcome names to column names in data",
    )

    # Promotion data (for cross-effects)
    promotion_columns: list[str] | None = Field(
        default=None,
        description="Column names for promotion variables",
    )

    # Optional overrides
    n_chains: int | None = None
    n_draws: int | None = None
    n_tune: int | None = None
    random_seed: int | None = None


# -----------------------------------------------------------------------------
# Extended Model Results
# -----------------------------------------------------------------------------


class MediationEffectSchema(BaseModel):
    """Mediation effect for a single channel."""

    channel: str
    direct_effect: float
    direct_effect_sd: float
    indirect_effects: dict[str, float]  # mediator_name -> effect
    indirect_effects_sd: dict[str, float]
    total_indirect: float
    total_effect: float
    proportion_mediated: float


class MediationResultsResponse(BaseModel):
    """Response containing mediation analysis results."""

    model_id: str
    mediator_names: list[str]
    channel_names: list[str]
    effects: list[MediationEffectSchema]


class CrossEffectResultSchema(BaseModel):
    """Cross-effect result between two outcomes."""

    source: str
    target: str
    effect_type: CrossEffectType
    mean: float
    sd: float
    hdi_low: float
    hdi_high: float


class MultivariateResultsResponse(BaseModel):
    """Response containing multivariate model results."""

    model_id: str
    outcome_names: list[str]
    channel_names: list[str]
    outcome_correlations: dict[str, dict[str, float]]  # outcome -> outcome -> correlation
    cross_effects: list[CrossEffectResultSchema]
    per_outcome_metrics: dict[str, dict[str, float]]  # outcome -> metric -> value


class ExtendedModelInfo(BaseModel):
    """Extended model information."""

    model_id: str
    name: str | None
    description: str | None
    model_type: ModelType
    data_id: str
    config_id: str
    status: JobStatus
    progress: float = 0.0
    progress_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    diagnostics: dict[str, Any] | None = None

    # Extended model specifics
    mediator_names: list[str] | None = None
    outcome_names: list[str] | None = None
