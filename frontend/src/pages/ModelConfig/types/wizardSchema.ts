import { z } from 'zod';

// ============================================================================
// Enum Schemas
// ============================================================================

export const dimensionTypeSchema = z.enum([
  'Period',
  'Geography',
  'Product',
  'Campaign',
  'Outlet',
  'Creative',
]);

export const priorTypeSchema = z.enum([
  'half_normal',
  'normal',
  'log_normal',
  'gamma',
  'beta',
  'truncated_normal',
  'half_student_t',
]);

export const adstockTypeSchema = z.enum(['geometric', 'weibull', 'delayed', 'none']);

export const saturationTypeSchema = z.enum([
  'hill',
  'logistic',
  'michaelis_menten',
  'tanh',
  'none',
]);

export const trendTypeSchema = z.enum([
  'none',
  'linear',
  'piecewise',
  'spline',
  'gaussian_process',
]);

export const allocationMethodSchema = z.enum(['equal', 'population', 'sales', 'custom']);

export const inferenceMethodSchema = z.enum(['bayesian_pymc', 'bayesian_numpyro']);

export const dataFrequencySchema = z.enum(['W', 'D', 'M']);

// ============================================================================
// Prior Configuration Schema
// ============================================================================

export const priorConfigSchema = z.object({
  type: priorTypeSchema,
  params: z.record(z.string(), z.number()),
  dims: z.array(z.string()).optional(),
});

// ============================================================================
// Adstock & Saturation Schemas
// ============================================================================

export const adstockConfigSchema = z.object({
  type: adstockTypeSchema,
  l_max: z.number().min(1).max(52).default(8),
  alpha_prior: priorConfigSchema.optional(),
  theta_prior: priorConfigSchema.optional(),
  normalize: z.boolean().default(true),
});

export const saturationConfigSchema = z.object({
  type: saturationTypeSchema,
  kappa_prior: priorConfigSchema.optional(),
  slope_prior: priorConfigSchema.optional(),
  beta_prior: priorConfigSchema.optional(),
});

// ============================================================================
// KPI Configuration Schema
// ============================================================================

export const kpiConfigSchema = z.object({
  name: z.string().min(1, 'KPI variable is required'),
  display_name: z.string().optional(),
  dimensions: z.array(dimensionTypeSchema).min(1, 'At least one dimension is required'),
  log_transform: z.boolean().default(false),
  floor_value: z.number().default(0),
  unit: z.string().optional(),
});

// ============================================================================
// Media Channel Configuration Schema
// ============================================================================

export const mediaChannelConfigSchema = z.object({
  name: z.string().min(1, 'Channel name is required'),
  display_name: z.string().optional(),
  dimensions: z.array(dimensionTypeSchema).default(['Period']),
  adstock: adstockConfigSchema,
  saturation: saturationConfigSchema,
  coefficient_prior: priorConfigSchema.optional(),
  parent_channel: z.string().optional(),
  split_dimensions: z.array(dimensionTypeSchema).optional(),
  unit: z.string().optional(),
});

// ============================================================================
// Control Variable Configuration Schema
// ============================================================================

export const controlVariableConfigSchema = z.object({
  name: z.string().min(1, 'Control variable name is required'),
  display_name: z.string().optional(),
  dimensions: z.array(dimensionTypeSchema).default(['Period']),
  allow_negative: z.boolean().default(false),
  coefficient_prior: priorConfigSchema.optional(),
  use_shrinkage: z.boolean().default(false),
  unit: z.string().optional(),
});

// ============================================================================
// MFF Column Configuration Schema
// ============================================================================

export const mffColumnConfigSchema = z.object({
  period: z.string().min(1, 'Period column is required'),
  geography: z.string().default('Geography'),
  product: z.string().default('Product'),
  campaign: z.string().default('Campaign'),
  outlet: z.string().default('Outlet'),
  creative: z.string().default('Creative'),
  variable_name: z.string().min(1, 'Variable name column is required'),
  variable_value: z.string().min(1, 'Variable value column is required'),
});

// ============================================================================
// Dimension Alignment Schema
// ============================================================================

export const dimensionAlignmentConfigSchema = z.object({
  geo_allocation: allocationMethodSchema.default('equal'),
  product_allocation: allocationMethodSchema.default('equal'),
  geo_weight_variable: z.string().optional(),
  product_weight_variable: z.string().optional(),
  prefer_disaggregation: z.boolean().default(true),
});

// ============================================================================
// Model Settings Schemas
// ============================================================================

export const trendConfigSchema = z.object({
  type: trendTypeSchema,
  n_changepoints: z.number().min(0).max(25).optional(),
  changepoint_range: z.number().min(0).max(1).optional(),
  n_knots: z.number().min(2).max(20).optional(),
});

export const seasonalityConfigSchema = z.object({
  yearly_order: z.number().min(0).max(10).default(2),
  monthly_order: z.number().min(0).max(6).default(0),
  weekly_order: z.number().min(0).max(4).default(0),
});

export const hierarchicalConfigSchema = z.object({
  enabled: z.boolean().default(false),
  pool_across_geo: z.boolean().default(true),
  pool_across_product: z.boolean().default(true),
  non_centered: z.boolean().default(true),
  mu_prior: priorConfigSchema.optional(),
  sigma_prior: priorConfigSchema.optional(),
});

export const modelSettingsSchema = z.object({
  inference_method: inferenceMethodSchema.default('bayesian_numpyro'),
  n_chains: z.number().min(1).max(8).default(4),
  n_draws: z.number().min(100).max(10000).default(2000),
  n_tune: z.number().min(100).max(5000).default(1000),
  target_accept: z.number().min(0.5).max(0.99).default(0.9),
  trend: trendConfigSchema.optional(),
  seasonality: seasonalityConfigSchema.optional(),
  hierarchical: hierarchicalConfigSchema.optional(),
  random_seed: z.number().optional(),
});

// ============================================================================
// Full Wizard Form Schema
// ============================================================================

export const wizardFormSchema = z.object({
  // Configuration metadata
  name: z.string().min(1, 'Configuration name is required'),
  description: z.string().optional(),

  // Step 1: KPI
  kpi: kpiConfigSchema,

  // Step 2: Media Channels (at least one required)
  media_channels: z.array(mediaChannelConfigSchema).min(1, 'At least one media channel is required'),

  // Step 3: Control Variables (optional)
  controls: z.array(controlVariableConfigSchema).default([]),

  // Step 4: Model Settings
  model_settings: modelSettingsSchema,

  // Step 5: MFF Columns
  columns: mffColumnConfigSchema,

  // Step 6: Dimension Alignment
  alignment: dimensionAlignmentConfigSchema,

  // Additional MFF config
  date_format: z.string().default('%Y-%m-%d'),
  frequency: dataFrequencySchema.default('W'),
  fill_missing_media: z.number().default(0),
  fill_missing_controls: z.number().optional(),
});

// ============================================================================
// Step-Specific Schemas for Validation
// ============================================================================

export const kpiStepSchema = z.object({
  name: z.string().min(1, 'Configuration name is required'),
  kpi: kpiConfigSchema,
});

export const mediaChannelsStepSchema = z.object({
  media_channels: z.array(mediaChannelConfigSchema).min(1, 'At least one media channel is required'),
});

export const controlsStepSchema = z.object({
  controls: z.array(controlVariableConfigSchema),
});

export const modelSettingsStepSchema = z.object({
  model_settings: modelSettingsSchema,
});

export const mffColumnsStepSchema = z.object({
  columns: mffColumnConfigSchema,
  date_format: z.string(),
  frequency: dataFrequencySchema,
  fill_missing_media: z.number(),
  fill_missing_controls: z.number().optional(),
});

export const alignmentStepSchema = z.object({
  alignment: dimensionAlignmentConfigSchema,
});

// ============================================================================
// Type Exports
// ============================================================================

export type WizardFormData = z.infer<typeof wizardFormSchema>;
export type KPIStepData = z.infer<typeof kpiStepSchema>;
export type MediaChannelsStepData = z.infer<typeof mediaChannelsStepSchema>;
export type ControlsStepData = z.infer<typeof controlsStepSchema>;
export type ModelSettingsStepData = z.infer<typeof modelSettingsStepSchema>;
export type MFFColumnsStepData = z.infer<typeof mffColumnsStepSchema>;
export type AlignmentStepData = z.infer<typeof alignmentStepSchema>;

// ============================================================================
// Default Values for New Entries
// ============================================================================

export const DEFAULT_ADSTOCK_CONFIG: z.infer<typeof adstockConfigSchema> = {
  type: 'geometric',
  l_max: 8,
  normalize: true,
  alpha_prior: {
    type: 'beta',
    params: { alpha: 2, beta: 2 },
  },
};

export const DEFAULT_SATURATION_CONFIG: z.infer<typeof saturationConfigSchema> = {
  type: 'hill',
  kappa_prior: {
    type: 'gamma',
    params: { alpha: 2, beta: 1 },
  },
  slope_prior: {
    type: 'gamma',
    params: { alpha: 2, beta: 1 },
  },
};

export const DEFAULT_MEDIA_CHANNEL: z.infer<typeof mediaChannelConfigSchema> = {
  name: '',
  dimensions: ['Period'],
  adstock: DEFAULT_ADSTOCK_CONFIG,
  saturation: DEFAULT_SATURATION_CONFIG,
  coefficient_prior: {
    type: 'half_normal',
    params: { sigma: 1 },
  },
};

export const DEFAULT_CONTROL_VARIABLE: z.infer<typeof controlVariableConfigSchema> = {
  name: '',
  dimensions: ['Period'],
  allow_negative: false,
  use_shrinkage: false,
  coefficient_prior: {
    type: 'normal',
    params: { mu: 0, sigma: 1 },
  },
};

export const DEFAULT_PRIOR_CONFIG: z.infer<typeof priorConfigSchema> = {
  type: 'half_normal',
  params: { sigma: 1 },
};
