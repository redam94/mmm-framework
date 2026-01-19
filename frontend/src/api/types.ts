// API Types - Mirrors api/schemas.py from the backend

// ============================================================================
// Enums
// ============================================================================

export type JobStatus = 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export type DataFormat = 'csv' | 'parquet' | 'excel' | 'json';

export type TrendType = 'none' | 'linear' | 'piecewise' | 'spline' | 'gaussian_process';

export type AllocationMethod = 'equal' | 'population' | 'sales' | 'custom';

export type ModelType = 'standard' | 'nested' | 'multivariate' | 'combined';

export type MediatorType = 'fully_observed' | 'partially_observed' | 'aggregated_survey' | 'fully_latent';

export type CrossEffectType = 'cannibalization' | 'halo' | 'symmetric' | 'asymmetric';

export type EffectConstraint = 'none' | 'positive' | 'negative';

export type AdstockType = 'geometric' | 'weibull' | 'delayed' | 'none';

export type SaturationType = 'hill' | 'logistic' | 'michaelis_menten' | 'tanh' | 'none';

export type PriorType = 'half_normal' | 'normal' | 'log_normal' | 'gamma' | 'beta' | 'truncated_normal' | 'half_student_t';

export type DimensionType = 'Period' | 'Geography' | 'Product' | 'Campaign' | 'Outlet' | 'Creative';

export type InferenceMethod = 'bayesian_pymc' | 'bayesian_numpyro';

export type DataFrequency = 'W' | 'D' | 'M';

// ============================================================================
// Prior Configuration
// ============================================================================

export interface PriorConfig {
  type: PriorType;
  params: Record<string, number>;
  dims?: string[];
}

// ============================================================================
// Adstock & Saturation Configuration
// ============================================================================

export interface AdstockConfig {
  type: AdstockType;
  l_max: number;
  alpha_prior?: PriorConfig;
  theta_prior?: PriorConfig;
  normalize?: boolean;
}

export interface SaturationConfig {
  type: SaturationType;
  kappa_prior?: PriorConfig;
  slope_prior?: PriorConfig;
  beta_prior?: PriorConfig;
}

// ============================================================================
// Variable Configuration
// ============================================================================

export interface KPIConfig {
  name: string;
  display_name?: string;
  dimensions: DimensionType[];
  log_transform: boolean;
  floor_value: number;
  unit?: string;
}

export interface MediaChannelConfig {
  name: string;
  display_name?: string;
  dimensions: DimensionType[];
  adstock: AdstockConfig;
  saturation: SaturationConfig;
  coefficient_prior?: PriorConfig;
  parent_channel?: string;
  split_dimensions?: DimensionType[];
  unit?: string;
}

export interface ControlVariableConfig {
  name: string;
  display_name?: string;
  dimensions: DimensionType[];
  allow_negative: boolean;
  coefficient_prior?: PriorConfig;
  use_shrinkage: boolean;
  unit?: string;
}

// ============================================================================
// Dimension Alignment
// ============================================================================

export interface DimensionAlignmentConfig {
  geo_allocation: AllocationMethod;
  product_allocation: AllocationMethod;
  geo_weight_variable?: string;
  product_weight_variable?: string;
  prefer_disaggregation: boolean;
}

// ============================================================================
// Model Settings
// ============================================================================

export interface TrendConfig {
  type: TrendType;
  n_changepoints?: number;
  changepoint_range?: number;
  n_knots?: number;
}

export interface SeasonalityConfig {
  yearly_order: number;
  monthly_order: number;
  weekly_order: number;
}

export interface HierarchicalConfig {
  enabled: boolean;
  pool_across_geo: boolean;
  pool_across_product: boolean;
  non_centered: boolean;
  mu_prior?: PriorConfig;
  sigma_prior?: PriorConfig;
}

export interface ModelSettings {
  inference_method: InferenceMethod;
  n_chains: number;
  n_draws: number;
  n_tune: number;
  target_accept: number;
  trend?: TrendConfig;
  seasonality?: SeasonalityConfig;
  hierarchical?: HierarchicalConfig;
  random_seed?: number;
}

// ============================================================================
// MFF Configuration
// ============================================================================

export interface MFFColumnConfig {
  period: string;
  geography: string;
  product: string;
  campaign: string;
  outlet: string;
  creative: string;
  variable_name: string;
  variable_value: string;
}

export interface MFFConfig {
  columns: MFFColumnConfig;
  kpi: KPIConfig;
  media_channels: MediaChannelConfig[];
  controls: ControlVariableConfig[];
  alignment: DimensionAlignmentConfig;
  date_format: string;
  frequency: DataFrequency;
  fill_missing_media: number;
  fill_missing_controls?: number;
}

// ============================================================================
// API Request/Response Types
// ============================================================================

// Data Upload
export interface DataUploadResponse {
  data_id: string;
  filename: string;
  format: DataFormat;
  rows: number;
  columns: number;
  variables: string[];
  dimensions: Record<string, string[]>;
  created_at: string;
}

export interface DatasetInfo {
  data_id: string;
  filename: string;
  format: DataFormat;
  rows: number;
  columns: number;
  variables: string[];
  dimensions: Record<string, string[]>;
  created_at: string;
  preview?: Record<string, unknown>[];
}

export interface VariableSummary {
  name: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  missing: number;
}

export interface DatasetListResponse {
  datasets: DatasetInfo[];
  total: number;
}

// Configuration
export interface ConfigCreateRequest {
  name: string;
  description?: string;
  mff_config: MFFConfig;
  model_settings: ModelSettings;
}

export interface ConfigUpdateRequest {
  name?: string;
  description?: string;
  mff_config?: Partial<MFFConfig>;
  model_settings?: Partial<ModelSettings>;
}

export interface ConfigInfo {
  config_id: string;
  name: string;
  description?: string;
  mff_config: MFFConfig;
  model_settings: ModelSettings;
  created_at: string;
  updated_at: string;
}

export interface ConfigListResponse {
  configs: ConfigInfo[];
  total: number;
}

export interface ConfigValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

// Model Fitting
export interface ModelFitRequest {
  data_id: string;
  config_id: string;
  name?: string;
  project_id?: string;
  overrides?: {
    n_chains?: number;
    n_draws?: number;
    n_tune?: number;
    target_accept?: number;
    random_seed?: number;
  };
}

export interface ModelInfo {
  model_id: string;
  name: string;
  data_id: string;
  config_id: string;
  project_id?: string;
  status: JobStatus;
  progress: number;
  progress_message?: string;
  error_message?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface ModelStatusResponse {
  model_id: string;
  status: JobStatus;
  progress: number;
  progress_message?: string;
  error_message?: string;
  updated_at: string;
}

export interface ModelListResponse {
  models: ModelInfo[];
  total: number;
}

// Model Results
export interface DiagnosticsSummary {
  n_divergences: number;
  rhat_max: number;
  rhat_mean: number;
  ess_bulk_min: number;
  ess_tail_min: number;
  warnings: string[];
}

export interface ParameterSummary {
  name: string;
  mean: number;
  std: number;
  hdi_low: number;
  hdi_high: number;
  rhat: number;
  ess_bulk: number;
  ess_tail: number;
}

export interface ModelResults {
  model_id: string;
  diagnostics: DiagnosticsSummary;
  parameters: ParameterSummary[];
  channel_names: string[];
  control_names: string[];
  n_observations: number;
  n_chains: number;
  n_draws: number;
}

export interface ModelFitData {
  model_id: string;
  periods: string[];
  observed: number[];
  predicted_mean: number[];
  predicted_std?: number[];
  r2: number;
  rmse: number;
  mape: number;
  has_geo: boolean;
  has_product: boolean;
  geographies?: string[];
  products?: string[];
  by_geography?: Record<string, {
    observed: number[];
    predicted_mean: number[];
    r2: number;
    rmse: number;
    mape: number;
  }>;
}

export interface PosteriorSamples {
  [parameterName: string]: {
    samples: number[];
    mean: number;
    std: number;
    shape?: number[];
  };
}

export interface PriorPosteriorComparison {
  model_id: string;
  parameters: {
    name: string;
    prior_samples: number[];
    posterior_samples: number[];
    prior_mean: number;
    prior_std: number;
    posterior_mean: number;
    posterior_std: number;
    shrinkage: number;
  }[];
}

export interface ResponseCurveData {
  spend: number[];
  response: number[];
  response_hdi_low: number[];
  response_hdi_high: number[];
  current_spend: number;
  spend_max: number;
}

export interface ResponseCurvesResponse {
  model_id: string;
  channels: Record<string, ResponseCurveData>;
}

export interface DecompositionResponse {
  model_id: string;
  periods: string[];
  components: Record<string, number[]>;
  by_geography?: Record<string, Record<string, number[]>>;
  observed: number[];
  metadata: {
    geo_names?: string[];
    product_names?: string[];
    channel_names: string[];
    has_geo: boolean;
    has_product: boolean;
    has_trend: boolean;
    has_seasonality: boolean;
    trend_type?: string;
  };
}

// Contributions & Scenarios
export interface ContributionRequest {
  time_period?: [string, string];
  channels?: string[];
  compute_uncertainty?: boolean;
  hdi_prob?: number;
}

export interface ContributionResult {
  contribution_id: string;
  model_id: string;
  status: JobStatus;
  channels: string[];
  total_contributions: Record<string, number>;
  contribution_pct: Record<string, number>;
  hdi_low?: Record<string, number>;
  hdi_high?: Record<string, number>;
  time_period: [string, string];
}

export interface ScenarioRequest {
  spend_changes: Record<string, number>;
  time_period?: [string, string];
}

export interface ScenarioResult {
  scenario_id: string;
  model_id: string;
  status: JobStatus;
  spend_changes: Record<string, number>;
  baseline_outcome: number;
  scenario_outcome: number;
  change: number;
  pct_change: number;
  time_period: [string, string];
}

// Reports
export interface ReportRequest {
  title?: string;
  client?: string;
  subtitle?: string;
  sections?: {
    executive_summary?: boolean;
    model_fit?: boolean;
    channel_contributions?: boolean;
    response_curves?: boolean;
    diagnostics?: boolean;
  };
  format_options?: {
    include_data_tables?: boolean;
    chart_width?: number;
    color_scheme?: string;
  };
}

export interface ReportInfo {
  report_id: string;
  model_id: string;
  status: JobStatus;
  title?: string;
  created_at: string;
  completed_at?: string;
  download_url?: string;
}

// Extended Models (Nested, Multivariate, Combined)
export interface MediatorConfig {
  name: string;
  type: MediatorType;
  observation_noise?: number;
  media_effect_prior?: PriorConfig;
  outcome_effect_prior?: PriorConfig;
  adstock?: AdstockConfig;
  saturation?: SaturationConfig;
}

export interface OutcomeConfig {
  name: string;
  column: string;
  intercept_prior?: PriorConfig;
  media_effect_prior?: PriorConfig;
  has_trend: boolean;
  has_seasonality: boolean;
}

export interface CrossEffectConfig {
  source: string;
  target: string;
  effect_type: CrossEffectType;
  prior_sigma: number;
  promotion_column?: string;
  lag: number;
}

export interface ExtendedModelConfig {
  model_type: ModelType;
  mediators?: MediatorConfig[];
  channel_mediator_mapping?: Record<string, string[]>;
  outcomes?: OutcomeConfig[];
  cross_effects?: CrossEffectConfig[];
  lkj_eta?: number;
  share_adstock?: boolean;
  share_saturation?: boolean;
}

export interface MediationEffect {
  channel: string;
  direct_effect: number;
  direct_effect_sd: number;
  indirect_effects: Record<string, number>;
  indirect_effects_sd: Record<string, number>;
  total_indirect: number;
  total_effect: number;
  proportion_mediated: number;
}

export interface MediationResults {
  model_id: string;
  mediator_names: string[];
  channel_names: string[];
  effects: MediationEffect[];
}

export interface CrossEffectResult {
  source: string;
  target: string;
  effect_type: CrossEffectType;
  mean: number;
  sd: number;
  hdi_low: number;
  hdi_high: number;
}

export interface MultivariateResults {
  model_id: string;
  outcome_names: string[];
  channel_names: string[];
  outcome_correlations: Record<string, Record<string, number>>;
  cross_effects: CrossEffectResult[];
  per_outcome_metrics: Record<string, {
    r2: number;
    rmse: number;
    mape: number;
  }>;
}

// Health Check
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  redis_connected: boolean;
  worker_healthy: boolean;
  timestamp: string;
}

export interface HealthDetailedResponse extends HealthResponse {
  queue_stats: {
    pending: number;
    running: number;
    completed: number;
    failed: number;
  };
  storage_stats: {
    datasets: number;
    configs: number;
    models: number;
  };
  redis_info?: Record<string, unknown>;
}

// ============================================================================
// Utility Types
// ============================================================================

export interface PaginationParams {
  skip?: number;
  limit?: number;
}

export interface ApiError {
  status: number;
  message: string;
  details?: unknown;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}
