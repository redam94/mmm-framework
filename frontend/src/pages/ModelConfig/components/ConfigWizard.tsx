import { Title, Text, Button } from '@tremor/react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { useConfigWizardStore, STEP_LABELS } from '../../../stores/configWizardStore';
import { useProjectStore } from '../../../stores/projectStore';
import { WizardProgress } from './WizardProgress';
import { WizardNavigation } from './WizardNavigation';
import {
  KPIStep,
  MediaChannelsStep,
  ControlVariablesStep,
  ModelSettingsStep,
  MFFColumnsStep,
  DimensionAlignmentStep,
  ReviewStep,
} from './steps';

export function ConfigWizard() {
  const { isOpen, currentStep, closeWizard, editingConfigId } = useConfigWizardStore();
  const { selectedDataId } = useProjectStore();

  if (!isOpen) return null;

  // Render current step
  const renderStep = () => {
    switch (currentStep) {
      case 'kpi':
        return <KPIStep />;
      case 'media_channels':
        return <MediaChannelsStep />;
      case 'controls':
        return <ControlVariablesStep />;
      case 'model_settings':
        return <ModelSettingsStep />;
      case 'mff_columns':
        return <MFFColumnsStep />;
      case 'alignment':
        return <DimensionAlignmentStep />;
      case 'review':
        return <ReviewStep />;
      default:
        return null;
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={closeWizard}
      />

      {/* Modal */}
      <div className="flex min-h-full items-center justify-center p-4">
        <div className="relative w-full max-w-4xl bg-white rounded-xl shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b">
            <div>
              <Title>
                {editingConfigId ? 'Edit Configuration' : 'New Configuration'}
              </Title>
              <Text className="text-gray-600">
                Step: {STEP_LABELS[currentStep]}
              </Text>
            </div>
            <Button
              variant="light"
              icon={XMarkIcon}
              onClick={closeWizard}
              className="text-gray-500 hover:text-gray-700"
            >
              Close
            </Button>
          </div>

          {/* Dataset check */}
          {!selectedDataId && currentStep === 'kpi' && (
            <div className="p-6 bg-yellow-50 border-b border-yellow-200">
              <Text className="text-yellow-800">
                <strong>Note:</strong> No dataset is currently selected. Please select a dataset
                from the Data Upload page to populate variable dropdowns.
              </Text>
            </div>
          )}

          {/* Progress */}
          <div className="px-6 pt-6">
            <WizardProgress />
          </div>

          {/* Content */}
          <div className="p-6 max-h-[60vh] overflow-y-auto">
            {renderStep()}
          </div>

          {/* Navigation */}
          <div className="px-6 pb-6">
            <WizardNavigation onClose={closeWizard} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ConfigWizard;
