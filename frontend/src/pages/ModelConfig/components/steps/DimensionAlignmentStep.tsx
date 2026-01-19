import { useEffect } from 'react';
import { Card, Title, Text, Select, SelectItem, Switch } from '@tremor/react';
import { useConfigWizardStore } from '../../../../stores/configWizardStore';
import { useDatasetVariables } from '../../../../api/hooks';
import { useProjectStore } from '../../../../stores/projectStore';
import { VariableSelector } from '../shared';
import { LoadingSpinner } from '../../../../components/common/LoadingSpinner';
import type { AllocationMethod } from '../../../../api/types';

const ALLOCATION_METHODS: { value: AllocationMethod; label: string; description: string }[] = [
  { value: 'equal', label: 'Equal', description: 'Allocate equally across all levels' },
  { value: 'population', label: 'By Population', description: 'Weight by population data' },
  { value: 'sales', label: 'By Sales', description: 'Weight by sales data' },
  { value: 'custom', label: 'Custom', description: 'Use a custom weight variable' },
];

export function DimensionAlignmentStep() {
  const { draftConfig, updateDraft, setStepValidation } = useConfigWizardStore();
  const { selectedDataId } = useProjectStore();
  const { data: variablesData, isLoading } = useDatasetVariables(selectedDataId || undefined);

  const variables = variablesData?.map((v) => v.name) || [];

  const alignment = draftConfig.alignment || {
    geo_allocation: 'equal',
    product_allocation: 'equal',
    prefer_disaggregation: true,
  };

  // Validate - custom allocation requires a weight variable
  useEffect(() => {
    const geoValid = alignment.geo_allocation !== 'custom' || !!alignment.geo_weight_variable;
    const productValid = alignment.product_allocation !== 'custom' || !!alignment.product_weight_variable;
    const isValid = geoValid && productValid;

    const errors: string[] = [];
    if (!geoValid) errors.push('Geography weight variable is required for custom allocation');
    if (!productValid) errors.push('Product weight variable is required for custom allocation');

    setStepValidation('alignment', isValid, errors);
  }, [alignment, setStepValidation]);

  const updateAlignment = (key: string, value: unknown) => {
    updateDraft({ alignment: { ...alignment, [key]: value } });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner />
        <Text className="ml-2">Loading variables...</Text>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Introduction */}
      <Card className="p-4">
        <Title className="text-sm mb-2">Dimension Alignment</Title>
        <Text className="text-gray-600">
          Configure how data should be allocated and aggregated across dimensions. This is
          important when your media and KPI data are at different granularities.
        </Text>
      </Card>

      {/* Geography Allocation */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Geography Allocation</Title>
        <Text className="text-gray-600 mb-4">
          How should national-level media spend be allocated to regional KPIs?
        </Text>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-gray-700">Allocation Method</label>
            <Select
              value={alignment.geo_allocation}
              onValueChange={(val) => updateAlignment('geo_allocation', val as AllocationMethod)}
              className="mt-1"
            >
              {ALLOCATION_METHODS.map((method) => (
                <SelectItem key={method.value} value={method.value}>
                  {method.label}
                </SelectItem>
              ))}
            </Select>
            <Text className="text-xs text-gray-500 mt-1">
              {ALLOCATION_METHODS.find((m) => m.value === alignment.geo_allocation)?.description}
            </Text>
          </div>

          {alignment.geo_allocation === 'custom' && (
            <VariableSelector
              label="Geography Weight Variable"
              value={alignment.geo_weight_variable || ''}
              onChange={(val) => updateAlignment('geo_weight_variable', val)}
              variables={variables}
              required
              placeholder="Select weight variable..."
            />
          )}
        </div>
      </Card>

      {/* Product Allocation */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Product Allocation</Title>
        <Text className="text-gray-600 mb-4">
          How should brand-level media spend be allocated to product-level KPIs?
        </Text>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-gray-700">Allocation Method</label>
            <Select
              value={alignment.product_allocation}
              onValueChange={(val) => updateAlignment('product_allocation', val as AllocationMethod)}
              className="mt-1"
            >
              {ALLOCATION_METHODS.map((method) => (
                <SelectItem key={method.value} value={method.value}>
                  {method.label}
                </SelectItem>
              ))}
            </Select>
            <Text className="text-xs text-gray-500 mt-1">
              {ALLOCATION_METHODS.find((m) => m.value === alignment.product_allocation)?.description}
            </Text>
          </div>

          {alignment.product_allocation === 'custom' && (
            <VariableSelector
              label="Product Weight Variable"
              value={alignment.product_weight_variable || ''}
              onChange={(val) => updateAlignment('product_weight_variable', val)}
              variables={variables}
              required
              placeholder="Select weight variable..."
            />
          )}
        </div>
      </Card>

      {/* Disaggregation Preference */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <Title className="text-sm">Prefer Disaggregation</Title>
            <Text className="text-gray-600 text-sm mt-1">
              When possible, model at the most granular level available rather than aggregating up.
            </Text>
          </div>
          <Switch
            checked={alignment.prefer_disaggregation ?? true}
            onChange={(checked) => updateAlignment('prefer_disaggregation', checked)}
          />
        </div>
      </Card>

      {/* Help Section */}
      <Card className="p-4 bg-blue-50 border-blue-200">
        <Title className="text-sm text-blue-900">About Dimension Alignment</Title>
        <div className="mt-2 text-sm text-blue-800 space-y-2">
          <p>
            <strong>When to use allocation:</strong> Your media data is at a higher level (e.g., national)
            than your KPI data (e.g., regional). The model needs to know how to distribute media
            effects.
          </p>
          <p>
            <strong>Equal:</strong> Simple equal split. Use when you don't have weights.
          </p>
          <p>
            <strong>By Sales/Population:</strong> Weight by sales or population proportions. More
            accurate if you have this data.
          </p>
          <p>
            <strong>Custom:</strong> Use any variable in your dataset as weights.
          </p>
        </div>
      </Card>
    </div>
  );
}

export default DimensionAlignmentStep;
