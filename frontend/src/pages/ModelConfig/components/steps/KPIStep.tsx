import { useEffect } from 'react';
import { Card, Title, Text, TextInput, NumberInput, Switch } from '@tremor/react';
import { useConfigWizardStore } from '../../../../stores/configWizardStore';
import { useDatasetVariables } from '../../../../api/hooks';
import { useProjectStore } from '../../../../stores/projectStore';
import { VariableSelector, DimensionSelector } from '../shared';
import { LoadingSpinner } from '../../../../components/common/LoadingSpinner';
import type { DimensionType } from '../../../../api/types';

export function KPIStep() {
  const { draftConfig, updateDraft, setStepValidation } = useConfigWizardStore();
  const { selectedDataId } = useProjectStore();
  const { data: variablesData, isLoading: variablesLoading } = useDatasetVariables(selectedDataId || undefined);

  const variables = variablesData?.map((v) => v.name) || [];
  const kpi = draftConfig.kpi || { name: '', dimensions: ['Period'], log_transform: false, floor_value: 0 };

  // Validate step on changes
  useEffect(() => {
    const isValid = !!(draftConfig.name && draftConfig.name.trim() !== '' && kpi.name && kpi.name.trim() !== '');
    const errors: string[] = [];
    if (!draftConfig.name || draftConfig.name.trim() === '') {
      errors.push('Configuration name is required');
    }
    if (!kpi.name || kpi.name.trim() === '') {
      errors.push('KPI variable is required');
    }
    setStepValidation('kpi', isValid, errors);
  }, [draftConfig.name, kpi.name, setStepValidation]);

  if (variablesLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner />
        <Text className="ml-2">Loading variables...</Text>
      </div>
    );
  }

  if (!selectedDataId) {
    return (
      <Card className="p-6 text-center">
        <Title>No Dataset Selected</Title>
        <Text className="mt-2">
          Please select a dataset from the Data Upload page before creating a configuration.
        </Text>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Configuration Name */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Configuration Details</Title>
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Configuration Name <span className="text-red-500">*</span>
            </label>
            <TextInput
              value={draftConfig.name || ''}
              onChange={(e) => updateDraft({ name: e.target.value })}
              placeholder="e.g., Q4 2024 Media Mix Model"
              className="mt-1"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Description</label>
            <TextInput
              value={draftConfig.description || ''}
              onChange={(e) => updateDraft({ description: e.target.value })}
              placeholder="Optional description of this configuration"
              className="mt-1"
            />
          </div>
        </div>
      </Card>

      {/* KPI Configuration */}
      <Card className="p-4">
        <Title className="text-sm mb-4">KPI (Target Variable)</Title>
        <Text className="text-gray-600 mb-4">
          Select the outcome variable you want to model (e.g., sales, conversions, revenue).
        </Text>

        <div className="space-y-4">
          {/* Variable Selection */}
          <div className="grid grid-cols-2 gap-4">
            <VariableSelector
              label="KPI Variable"
              value={kpi.name}
              onChange={(name) => updateDraft({ kpi: { ...kpi, name } })}
              variables={variables}
              required
              placeholder="Select KPI variable..."
            />
            <div>
              <label className="text-sm font-medium text-gray-700">Display Name</label>
              <TextInput
                value={kpi.display_name || ''}
                onChange={(e) => updateDraft({ kpi: { ...kpi, display_name: e.target.value } })}
                placeholder="Optional display name"
                className="mt-1"
              />
            </div>
          </div>

          {/* Dimensions */}
          <DimensionSelector
            label="KPI Dimensions"
            value={kpi.dimensions as DimensionType[]}
            onChange={(dimensions) => updateDraft({ kpi: { ...kpi, dimensions } })}
            required
          />
          <Text className="text-xs text-gray-500">
            Select the dimensions your KPI varies by. Period is always required.
          </Text>

          {/* Transform Options */}
          <div className="grid grid-cols-2 gap-4 pt-4 border-t">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium text-gray-700">Log Transform</label>
                <Text className="text-xs text-gray-500">
                  Use multiplicative model (recommended for most cases)
                </Text>
              </div>
              <Switch
                checked={kpi.log_transform}
                onChange={(checked) => updateDraft({ kpi: { ...kpi, log_transform: checked } })}
              />
            </div>
            <div>
              <label className="text-sm font-medium text-gray-700">Floor Value</label>
              <NumberInput
                value={kpi.floor_value}
                onValueChange={(val) => updateDraft({ kpi: { ...kpi, floor_value: val ?? 0 } })}
                min={0}
                step={0.01}
                className="mt-1"
              />
              <Text className="text-xs text-gray-500 mt-1">
                Minimum value before log transform (prevents log(0))
              </Text>
            </div>
          </div>

          {/* Unit (optional) */}
          <div>
            <label className="text-sm font-medium text-gray-700">Unit (optional)</label>
            <TextInput
              value={kpi.unit || ''}
              onChange={(e) => updateDraft({ kpi: { ...kpi, unit: e.target.value } })}
              placeholder="e.g., USD, units, conversions"
              className="mt-1"
            />
          </div>
        </div>
      </Card>

      {/* Help Section */}
      <Card className="p-4 bg-blue-50 border-blue-200">
        <Title className="text-sm text-blue-900">About KPI Configuration</Title>
        <div className="mt-2 text-sm text-blue-800 space-y-2">
          <p>
            <strong>Log Transform:</strong> Enable this for multiplicative models where effects are
            percentage-based rather than additive. Most marketing mix models use log-transformed outcomes.
          </p>
          <p>
            <strong>Dimensions:</strong> If your KPI varies by geography or product, include those dimensions.
            The model will then estimate separate effects for each level.
          </p>
        </div>
      </Card>
    </div>
  );
}

export default KPIStep;
