import { useEffect } from 'react';
import { Card, Title, Text, Button } from '@tremor/react';
import { PlusIcon } from '@heroicons/react/24/outline';
import { useConfigWizardStore } from '../../../../stores/configWizardStore';
import { useDatasetVariables } from '../../../../api/hooks';
import { useProjectStore } from '../../../../stores/projectStore';
import { ControlVariableCard } from '../shared';
import { LoadingSpinner } from '../../../../components/common/LoadingSpinner';
import { DEFAULT_CONTROL_VARIABLE } from '../../types/wizardSchema';
import type { ControlVariableConfig } from '../../../../api/types';

export function ControlVariablesStep() {
  const { draftConfig, updateDraft, setStepValidation } = useConfigWizardStore();
  const { selectedDataId } = useProjectStore();
  const { data: variablesData, isLoading: variablesLoading } = useDatasetVariables(selectedDataId || undefined);

  const variables = variablesData?.map((v) => v.name) || [];
  const controls = draftConfig.controls || [];

  // This step is optional - always valid
  useEffect(() => {
    // Only invalid if controls exist but some are unnamed
    const allControlsNamed = controls.every((c) => c.name && c.name.trim() !== '');
    const isValid = controls.length === 0 || allControlsNamed;

    const errors: string[] = [];
    if (!allControlsNamed) {
      errors.push('All control variables must have a variable selected');
    }

    setStepValidation('controls', isValid, errors);
  }, [controls, setStepValidation]);

  const handleAddControl = () => {
    const newControl: ControlVariableConfig = {
      ...DEFAULT_CONTROL_VARIABLE,
      name: '',
    };
    updateDraft({ controls: [...controls, newControl] });
  };

  const handleUpdateControl = (index: number, updated: ControlVariableConfig) => {
    const newControls = [...controls];
    newControls[index] = updated;
    updateDraft({ controls: newControls });
  };

  const handleRemoveControl = (index: number) => {
    const newControls = controls.filter((_, i) => i !== index);
    updateDraft({ controls: newControls });
  };

  if (variablesLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner />
        <Text className="ml-2">Loading variables...</Text>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <Title className="text-sm">Control Variables (Optional)</Title>
            <Text className="text-gray-600">
              Add non-marketing variables that affect your KPI (e.g., seasonality, promotions, macro factors).
            </Text>
          </div>
          <Button icon={PlusIcon} variant="secondary" onClick={handleAddControl}>
            Add Control
          </Button>
        </div>
      </Card>

      {/* Control List */}
      {controls.length === 0 ? (
        <Card className="p-8 text-center border-2 border-dashed">
          <Title className="text-gray-500">No control variables added</Title>
          <Text className="text-gray-400 mt-2">
            Control variables are optional. They help account for factors outside of marketing.
          </Text>
          <Text className="text-gray-400 mt-1 text-sm">
            Examples: holidays, promotions, economic indicators, competitor actions
          </Text>
          <Button variant="secondary" icon={PlusIcon} onClick={handleAddControl} className="mt-4">
            Add Control Variable
          </Button>
        </Card>
      ) : (
        <div className="space-y-4">
          {controls.map((control, index) => (
            <ControlVariableCard
              key={index}
              value={control}
              onChange={(updated) => handleUpdateControl(index, updated)}
              onRemove={() => handleRemoveControl(index)}
              variables={variables}
              index={index}
            />
          ))}
        </div>
      )}

      {/* Add More Button */}
      {controls.length > 0 && (
        <Button variant="secondary" icon={PlusIcon} onClick={handleAddControl}>
          Add Another Control
        </Button>
      )}

      {/* Help Section */}
      <Card className="p-4 bg-blue-50 border-blue-200">
        <Title className="text-sm text-blue-900">About Control Variables</Title>
        <div className="mt-2 text-sm text-blue-800 space-y-2">
          <p>
            <strong>Allow Negative:</strong> Enable this if the variable can have a negative effect
            on your KPI (e.g., competitor spending might reduce your sales).
          </p>
          <p>
            <strong>Shrinkage Prior:</strong> Use this for variables where you expect small effects.
            The shrinkage prior pulls coefficients toward zero.
          </p>
          <p>
            <strong>Common Controls:</strong> Holiday indicators, price changes, distribution
            changes, economic indicators, weather, competitor activity.
          </p>
        </div>
      </Card>
    </div>
  );
}

export default ControlVariablesStep;
