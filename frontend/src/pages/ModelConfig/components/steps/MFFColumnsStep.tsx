import { useEffect } from 'react';
import { Card, Title, Text, Select, SelectItem, NumberInput } from '@tremor/react';
import { useConfigWizardStore } from '../../../../stores/configWizardStore';
import { useDataset } from '../../../../api/hooks';
import { useProjectStore } from '../../../../stores/projectStore';
import { LoadingSpinner } from '../../../../components/common/LoadingSpinner';
import type { DataFrequency } from '../../../../api/types';

const DATE_FORMATS = [
  { value: '%Y-%m-%d', label: 'YYYY-MM-DD (2024-01-15)' },
  { value: '%m/%d/%Y', label: 'MM/DD/YYYY (01/15/2024)' },
  { value: '%d/%m/%Y', label: 'DD/MM/YYYY (15/01/2024)' },
  { value: '%Y/%m/%d', label: 'YYYY/MM/DD (2024/01/15)' },
  { value: '%m-%d-%Y', label: 'MM-DD-YYYY (01-15-2024)' },
];

const FREQUENCIES: { value: DataFrequency; label: string }[] = [
  { value: 'D', label: 'Daily' },
  { value: 'W', label: 'Weekly' },
  { value: 'M', label: 'Monthly' },
];

export function MFFColumnsStep() {
  const { draftConfig, updateDraft, setStepValidation } = useConfigWizardStore();
  const { selectedDataId } = useProjectStore();
  const { data: dataset, isLoading } = useDataset(selectedDataId || undefined);

  // Get column names from the dataset
  const columns = dataset?.variables || [];

  const mffColumns = draftConfig.columns || {
    period: 'Period',
    geography: 'Geography',
    product: 'Product',
    campaign: 'Campaign',
    outlet: 'Outlet',
    creative: 'Creative',
    variable_name: 'VariableName',
    variable_value: 'VariableValue',
  };

  // Validate - period, variable_name, variable_value are required
  useEffect(() => {
    const isValid =
      mffColumns.period.trim() !== '' &&
      mffColumns.variable_name.trim() !== '' &&
      mffColumns.variable_value.trim() !== '';

    const errors: string[] = [];
    if (!mffColumns.period.trim()) errors.push('Period column is required');
    if (!mffColumns.variable_name.trim()) errors.push('Variable name column is required');
    if (!mffColumns.variable_value.trim()) errors.push('Variable value column is required');

    setStepValidation('mff_columns', isValid, errors);
  }, [mffColumns, setStepValidation]);

  const updateColumn = (key: keyof typeof mffColumns, value: string) => {
    updateDraft({ columns: { ...mffColumns, [key]: value } });
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner />
        <Text className="ml-2">Loading dataset columns...</Text>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Required Columns */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Required Columns</Title>
        <Text className="text-gray-600 mb-4">
          Map your dataset columns to the MFF format. These columns are required.
        </Text>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-700">
              Period Column <span className="text-red-500">*</span>
            </label>
            <Select
              value={mffColumns.period}
              onValueChange={(val) => updateColumn('period', val)}
              placeholder="Select column..."
              className="mt-1"
            >
              {columns.map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </Select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">
              Variable Name Column <span className="text-red-500">*</span>
            </label>
            <Select
              value={mffColumns.variable_name}
              onValueChange={(val) => updateColumn('variable_name', val)}
              placeholder="Select column..."
              className="mt-1"
            >
              {columns.map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </Select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">
              Variable Value Column <span className="text-red-500">*</span>
            </label>
            <Select
              value={mffColumns.variable_value}
              onValueChange={(val) => updateColumn('variable_value', val)}
              placeholder="Select column..."
              className="mt-1"
            >
              {columns.map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </Select>
          </div>
        </div>
      </Card>

      {/* Optional Dimension Columns */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Dimension Columns (Optional)</Title>
        <Text className="text-gray-600 mb-4">
          Map optional dimension columns. Leave as default if your data doesn't have these dimensions.
        </Text>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-700">Geography Column</label>
            <Select
              value={mffColumns.geography}
              onValueChange={(val) => updateColumn('geography', val)}
              placeholder="Select column..."
              className="mt-1"
            >
              {columns.map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </Select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Product Column</label>
            <Select
              value={mffColumns.product}
              onValueChange={(val) => updateColumn('product', val)}
              placeholder="Select column..."
              className="mt-1"
            >
              {columns.map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </Select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Campaign Column</label>
            <Select
              value={mffColumns.campaign}
              onValueChange={(val) => updateColumn('campaign', val)}
              placeholder="Select column..."
              className="mt-1"
            >
              {columns.map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </Select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Outlet Column</label>
            <Select
              value={mffColumns.outlet}
              onValueChange={(val) => updateColumn('outlet', val)}
              placeholder="Select column..."
              className="mt-1"
            >
              {columns.map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </Select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Creative Column</label>
            <Select
              value={mffColumns.creative}
              onValueChange={(val) => updateColumn('creative', val)}
              placeholder="Select column..."
              className="mt-1"
            >
              {columns.map((col) => (
                <SelectItem key={col} value={col}>
                  {col}
                </SelectItem>
              ))}
            </Select>
          </div>
        </div>
      </Card>

      {/* Data Format Settings */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Data Format Settings</Title>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-700">Date Format</label>
            <Select
              value={draftConfig.date_format || '%Y-%m-%d'}
              onValueChange={(val) => updateDraft({ date_format: val })}
              className="mt-1"
            >
              {DATE_FORMATS.map((fmt) => (
                <SelectItem key={fmt.value} value={fmt.value}>
                  {fmt.label}
                </SelectItem>
              ))}
            </Select>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Data Frequency</label>
            <Select
              value={draftConfig.frequency || 'W'}
              onValueChange={(val) => updateDraft({ frequency: val as DataFrequency })}
              className="mt-1"
            >
              {FREQUENCIES.map((freq) => (
                <SelectItem key={freq.value} value={freq.value}>
                  {freq.label}
                </SelectItem>
              ))}
            </Select>
          </div>
        </div>
      </Card>

      {/* Missing Value Settings */}
      <Card className="p-4">
        <Title className="text-sm mb-4">Missing Value Handling</Title>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-700">Fill Missing Media Values</label>
            <NumberInput
              value={draftConfig.fill_missing_media ?? 0}
              onValueChange={(val) => updateDraft({ fill_missing_media: val ?? 0 })}
              step={0.01}
              className="mt-1"
            />
            <Text className="text-xs text-gray-500 mt-1">
              Value to use for missing media channel data (typically 0)
            </Text>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-700">Fill Missing Control Values</label>
            <NumberInput
              value={draftConfig.fill_missing_controls ?? undefined}
              onValueChange={(val) => updateDraft({ fill_missing_controls: val || undefined })}
              step={0.01}
              placeholder="Leave empty to error on missing"
              className="mt-1"
            />
            <Text className="text-xs text-gray-500 mt-1">
              Value to use for missing control data (leave empty to require all data)
            </Text>
          </div>
        </div>
      </Card>

      {/* Help Section */}
      <Card className="p-4 bg-blue-50 border-blue-200">
        <Title className="text-sm text-blue-900">About MFF Format</Title>
        <div className="mt-2 text-sm text-blue-800 space-y-2">
          <p>
            The Master Flat File (MFF) format is a long-format data structure where each row
            represents a single observation of a variable at a specific time and dimension.
          </p>
          <p>
            <strong>Required columns:</strong> Period (date), VariableName, VariableValue
          </p>
          <p>
            <strong>Optional dimensions:</strong> Geography, Product, Campaign, Outlet, Creative
          </p>
        </div>
      </Card>
    </div>
  );
}

export default MFFColumnsStep;
