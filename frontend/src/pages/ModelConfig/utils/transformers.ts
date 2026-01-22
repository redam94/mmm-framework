import type { ConfigCreateRequest, ConfigInfo, MFFConfig } from '../../../api/types';
import type { WizardDraftConfig } from '../../../stores/configWizardStore';

/**
 * Transform wizard draft config to API request format.
 */
export function transformToConfigRequest(draft: Partial<WizardDraftConfig>): ConfigCreateRequest {
  const mff_config: MFFConfig = {
    columns: draft.columns || {
      period: 'Period',
      geography: 'Geography',
      product: 'Product',
      campaign: 'Campaign',
      outlet: 'Outlet',
      creative: 'Creative',
      variable_name: 'VariableName',
      variable_value: 'VariableValue',
    },
    kpi: draft.kpi || {
      name: '',
      dimensions: ['Period'],
      log_transform: false,
      floor_value: 0,
    },
    media_channels: draft.media_channels || [],
    controls: draft.controls || [],
    alignment: draft.alignment || {
      geo_allocation: 'equal',
      product_allocation: 'equal',
      prefer_disaggregation: true,
    },
    date_format: draft.date_format || '%Y-%m-%d',
    frequency: draft.frequency || 'W',
    fill_missing_media: draft.fill_missing_media ?? 0,
    fill_missing_controls: draft.fill_missing_controls,
  };

  return {
    name: draft.name || 'Unnamed Configuration',
    description: draft.description,
    mff_config,
    model_settings: draft.model_settings || {
      inference_method: 'bayesian_numpyro',
      n_chains: 4,
      n_draws: 2000,
      n_tune: 1000,
      target_accept: 0.9,
    },
  };
}

/**
 * Transform API config to wizard draft format (for editing existing configs).
 */
export function transformFromConfig(config: ConfigInfo): Partial<WizardDraftConfig> {
  return {
    name: config.name,
    description: config.description,
    kpi: config.mff_config.kpi,
    media_channels: config.mff_config.media_channels,
    controls: config.mff_config.controls,
    columns: config.mff_config.columns,
    alignment: config.mff_config.alignment,
    date_format: config.mff_config.date_format,
    frequency: config.mff_config.frequency,
    fill_missing_media: config.mff_config.fill_missing_media,
    fill_missing_controls: config.mff_config.fill_missing_controls,
    model_settings: config.model_settings,
  };
}
