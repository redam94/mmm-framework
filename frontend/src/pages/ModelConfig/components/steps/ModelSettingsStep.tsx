import { useEffect, useState } from 'react';
import { Card, Title, Text, Select, SelectItem, NumberInput, Switch } from '@tremor/react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { useConfigWizardStore } from '../../../../stores/configWizardStore';
import { PriorConfigEditor } from '../shared';
import type { InferenceMethod, TrendType, ModelSettings } from '../../../../api/types';

const INFERENCE_METHODS: { value: InferenceMethod; label: string; description: string }[] = [
  {
    value: 'bayesian_numpyro',
    label: 'NumPyro (Recommended)',
    description: 'Fast sampling with JAX acceleration',
  },
  {
    value: 'bayesian_pymc',
    label: 'PyMC',
    description: 'Standard PyMC sampling',
  },
];

const TREND_TYPES: { value: TrendType; label: string; description: string }[] = [
  { value: 'none', label: 'None', description: 'No baseline trend' },
  { value: 'linear', label: 'Linear', description: 'Simple linear trend' },
  { value: 'piecewise', label: 'Piecewise Linear', description: 'Linear segments with changepoints' },
  { value: 'spline', label: 'Spline', description: 'Smooth curve with knots' },
  { value: 'gaussian_process', label: 'Gaussian Process', description: 'Flexible non-parametric trend' },
];

export function ModelSettingsStep() {
  const { draftConfig, updateDraft, setStepValidation } = useConfigWizardStore();
  const [trendExpanded, setTrendExpanded] = useState(false);
  const [seasonalityExpanded, setSeasonalityExpanded] = useState(false);
  const [hierarchicalExpanded, setHierarchicalExpanded] = useState(false);

  const settings = draftConfig.model_settings || {
    inference_method: 'bayesian_numpyro',
    n_chains: 4,
    n_draws: 2000,
    n_tune: 1000,
    target_accept: 0.9,
    trend: { type: 'none' },
    seasonality: { yearly_order: 2, monthly_order: 0, weekly_order: 0 },
    hierarchical: { enabled: false, pool_across_geo: true, pool_across_product: true, non_centered: true },
  };

  // This step has sensible defaults - always valid
  useEffect(() => {
    setStepValidation('model_settings', true, []);
  }, [setStepValidation]);

  const updateSettings = (partial: Partial<ModelSettings>) => {
    updateDraft({ model_settings: { ...settings, ...partial } });
  };

  return (
    <div className="space-y-6">
      {/* Inference Settings */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Inference Settings</Title>

        <div className="grid grid-cols-2 gap-4">
          {/* Inference Method */}
          <div className="col-span-2">
            <label className="text-sm font-medium text-gray-700">Inference Method</label>
            <Select
              value={settings.inference_method}
              onValueChange={(val) => updateSettings({ inference_method: val as InferenceMethod })}
              className="mt-1"
            >
              {INFERENCE_METHODS.map((method) => (
                <SelectItem key={method.value} value={method.value}>
                  {method.label}
                </SelectItem>
              ))}
            </Select>
            <Text className="text-xs text-gray-500 mt-1">
              {INFERENCE_METHODS.find((m) => m.value === settings.inference_method)?.description}
            </Text>
          </div>

          {/* MCMC Parameters */}
          <div>
            <label className="text-sm font-medium text-gray-700">Chains</label>
            <NumberInput
              value={settings.n_chains}
              onValueChange={(val) => updateSettings({ n_chains: val ?? 4 })}
              min={1}
              max={8}
              step={1}
              className="mt-1"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Draws per Chain</label>
            <NumberInput
              value={settings.n_draws}
              onValueChange={(val) => updateSettings({ n_draws: val ?? 2000 })}
              min={100}
              max={10000}
              step={100}
              className="mt-1"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Tuning Steps</label>
            <NumberInput
              value={settings.n_tune}
              onValueChange={(val) => updateSettings({ n_tune: val ?? 1000 })}
              min={100}
              max={5000}
              step={100}
              className="mt-1"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Target Accept</label>
            <NumberInput
              value={settings.target_accept}
              onValueChange={(val) => updateSettings({ target_accept: val ?? 0.9 })}
              min={0.5}
              max={0.99}
              step={0.01}
              className="mt-1"
            />
          </div>

          {/* Random Seed */}
          <div className="col-span-2">
            <label className="text-sm font-medium text-gray-700">Random Seed (optional)</label>
            <NumberInput
              value={settings.random_seed || undefined}
              onValueChange={(val) => updateSettings({ random_seed: val || undefined })}
              min={1}
              step={1}
              placeholder="Leave empty for random"
              className="mt-1"
            />
            <Text className="text-xs text-gray-500 mt-1">
              Set a seed for reproducible results
            </Text>
          </div>
        </div>
      </Card>

      {/* Trend Configuration */}
      <Card className="p-4">
        <button
          type="button"
          onClick={() => setTrendExpanded(!trendExpanded)}
          className="w-full flex items-center justify-between text-left"
        >
          <div>
            <Title className="text-sm">Trend Configuration</Title>
            <Text className="text-xs text-gray-500">
              Current: {TREND_TYPES.find((t) => t.value === settings.trend?.type)?.label || 'None'}
            </Text>
          </div>
          {trendExpanded ? (
            <ChevronUpIcon className="h-5 w-5 text-gray-500" />
          ) : (
            <ChevronDownIcon className="h-5 w-5 text-gray-500" />
          )}
        </button>

        {trendExpanded && (
          <div className="mt-4 pt-4 border-t space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-700">Trend Type</label>
              <Select
                value={settings.trend?.type || 'none'}
                onValueChange={(val) =>
                  updateSettings({ trend: { ...settings.trend, type: val as TrendType } })
                }
                className="mt-1"
              >
                {TREND_TYPES.map((trend) => (
                  <SelectItem key={trend.value} value={trend.value}>
                    {trend.label}
                  </SelectItem>
                ))}
              </Select>
              <Text className="text-xs text-gray-500 mt-1">
                {TREND_TYPES.find((t) => t.value === settings.trend?.type)?.description}
              </Text>
            </div>

            {/* Piecewise options */}
            {settings.trend?.type === 'piecewise' && (
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700">Number of Changepoints</label>
                  <NumberInput
                    value={settings.trend?.n_changepoints ?? 5}
                    onValueChange={(val) =>
                      updateSettings({ trend: { ...settings.trend!, n_changepoints: val ?? 5 } })
                    }
                    min={1}
                    max={25}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium text-gray-700">Changepoint Range</label>
                  <NumberInput
                    value={settings.trend?.changepoint_range ?? 0.8}
                    onValueChange={(val) =>
                      updateSettings({ trend: { ...settings.trend!, changepoint_range: val ?? 0.8 } })
                    }
                    min={0.1}
                    max={1.0}
                    step={0.1}
                    className="mt-1"
                  />
                </div>
              </div>
            )}

            {/* Spline options */}
            {settings.trend?.type === 'spline' && (
              <div>
                <label className="text-sm font-medium text-gray-700">Number of Knots</label>
                <NumberInput
                  value={settings.trend?.n_knots ?? 5}
                  onValueChange={(val) =>
                    updateSettings({ trend: { ...settings.trend!, n_knots: val ?? 5 } })
                  }
                  min={2}
                  max={20}
                  className="mt-1"
                />
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Seasonality Configuration */}
      <Card className="p-4">
        <button
          type="button"
          onClick={() => setSeasonalityExpanded(!seasonalityExpanded)}
          className="w-full flex items-center justify-between text-left"
        >
          <div>
            <Title className="text-sm">Seasonality Configuration</Title>
            <Text className="text-xs text-gray-500">
              Yearly: {settings.seasonality?.yearly_order ?? 2}, Monthly: {settings.seasonality?.monthly_order ?? 0}, Weekly: {settings.seasonality?.weekly_order ?? 0}
            </Text>
          </div>
          {seasonalityExpanded ? (
            <ChevronUpIcon className="h-5 w-5 text-gray-500" />
          ) : (
            <ChevronDownIcon className="h-5 w-5 text-gray-500" />
          )}
        </button>

        {seasonalityExpanded && (
          <div className="mt-4 pt-4 border-t">
            <Text className="text-sm text-gray-600 mb-4">
              Set the Fourier order for each seasonality. Higher orders capture more complex patterns but risk overfitting.
            </Text>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="text-sm font-medium text-gray-700">Yearly Order</label>
                <NumberInput
                  value={settings.seasonality?.yearly_order ?? 2}
                  onValueChange={(val) =>
                    updateSettings({
                      seasonality: { ...settings.seasonality!, yearly_order: val ?? 2 },
                    })
                  }
                  min={0}
                  max={10}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-gray-700">Monthly Order</label>
                <NumberInput
                  value={settings.seasonality?.monthly_order ?? 0}
                  onValueChange={(val) =>
                    updateSettings({
                      seasonality: { ...settings.seasonality!, monthly_order: val ?? 0 },
                    })
                  }
                  min={0}
                  max={6}
                  className="mt-1"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-gray-700">Weekly Order</label>
                <NumberInput
                  value={settings.seasonality?.weekly_order ?? 0}
                  onValueChange={(val) =>
                    updateSettings({
                      seasonality: { ...settings.seasonality!, weekly_order: val ?? 0 },
                    })
                  }
                  min={0}
                  max={4}
                  className="mt-1"
                />
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Hierarchical Configuration */}
      <Card className="p-4">
        <button
          type="button"
          onClick={() => setHierarchicalExpanded(!hierarchicalExpanded)}
          className="w-full flex items-center justify-between text-left"
        >
          <div>
            <Title className="text-sm">Hierarchical Modeling</Title>
            <Text className="text-xs text-gray-500">
              {settings.hierarchical?.enabled ? 'Enabled' : 'Disabled'}
            </Text>
          </div>
          {hierarchicalExpanded ? (
            <ChevronUpIcon className="h-5 w-5 text-gray-500" />
          ) : (
            <ChevronDownIcon className="h-5 w-5 text-gray-500" />
          )}
        </button>

        {hierarchicalExpanded && (
          <div className="mt-4 pt-4 border-t space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">Enable Hierarchical Modeling</label>
                <Text className="text-xs text-gray-500">
                  Pool information across geographies and/or products
                </Text>
              </div>
              <Switch
                checked={settings.hierarchical?.enabled ?? false}
                onChange={(checked) =>
                  updateSettings({
                    hierarchical: { ...settings.hierarchical!, enabled: checked },
                  })
                }
              />
            </div>

            {settings.hierarchical?.enabled && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.hierarchical?.pool_across_geo ?? true}
                      onChange={(e) =>
                        updateSettings({
                          hierarchical: { ...settings.hierarchical!, pool_across_geo: e.target.checked },
                        })
                      }
                      className="rounded border-gray-300 text-blue-600"
                    />
                    <span className="text-sm text-gray-700">Pool across geographies</span>
                  </label>

                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={settings.hierarchical?.pool_across_product ?? true}
                      onChange={(e) =>
                        updateSettings({
                          hierarchical: { ...settings.hierarchical!, pool_across_product: e.target.checked },
                        })
                      }
                      className="rounded border-gray-300 text-blue-600"
                    />
                    <span className="text-sm text-gray-700">Pool across products</span>
                  </label>
                </div>

                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.hierarchical?.non_centered ?? true}
                    onChange={(e) =>
                      updateSettings({
                        hierarchical: { ...settings.hierarchical!, non_centered: e.target.checked },
                      })
                    }
                    className="rounded border-gray-300 text-blue-600"
                  />
                  <span className="text-sm text-gray-700">Use non-centered parameterization</span>
                </label>
                <Text className="text-xs text-gray-500">
                  Non-centered parameterization often improves sampling efficiency
                </Text>

                {/* Hyperpriors */}
                {settings.hierarchical?.mu_prior && (
                  <PriorConfigEditor
                    label="Group Mean Prior (mu)"
                    value={settings.hierarchical.mu_prior}
                    onChange={(prior) =>
                      updateSettings({
                        hierarchical: { ...settings.hierarchical!, mu_prior: prior },
                      })
                    }
                    defaultCollapsed={true}
                  />
                )}
                {settings.hierarchical?.sigma_prior && (
                  <PriorConfigEditor
                    label="Group SD Prior (sigma)"
                    value={settings.hierarchical.sigma_prior}
                    onChange={(prior) =>
                      updateSettings({
                        hierarchical: { ...settings.hierarchical!, sigma_prior: prior },
                      })
                    }
                    defaultCollapsed={true}
                  />
                )}
              </>
            )}
          </div>
        )}
      </Card>

      {/* Help Section */}
      <Card className="p-4 bg-blue-50 border-blue-200">
        <Title className="text-sm text-blue-900">Recommended Settings</Title>
        <div className="mt-2 text-sm text-blue-800 space-y-2">
          <p>
            <strong>For exploration:</strong> 2 chains, 1000 draws, 500 tune
          </p>
          <p>
            <strong>For final analysis:</strong> 4 chains, 2000+ draws, 1000 tune
          </p>
          <p>
            <strong>Target accept:</strong> Increase to 0.95+ if you see divergences
          </p>
        </div>
      </Card>
    </div>
  );
}

export default ModelSettingsStep;
